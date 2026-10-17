#!/usr/bin/env python3
"""
Model discovery & compatibility scoring for benchtest.

Two discovery sources per framework:
  1. served — the framework's live /v1/models (authoritative while it runs)
  2. local  — repos already in the Hugging Face cache (serve-ready, no download)

Every candidate is scored for compatibility with the benchmark:
  RAM fit (weights + KV-cache headroom), context window vs the configured
  window, local availability, and MTP draft availability for speculative
  decoding. The UI uses this to let users pick a model per framework.

Stdlib only.
"""

import glob
import json
import os
import re
import subprocess
import urllib.request

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
# MTPLX keeps its own model store, separate from the HF cache.
MTPLX_MODELS = os.path.expanduser("~/.mtplx/models")
_GB = 1073741824
# Weights + KV cache + engine overhead: 25% headroom over raw weight size.
RAM_HEADROOM = 1.25


def _get_json(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def cache_dirname(model_id):
    """Resolve an HF-cache directory for a model id. Handles both id styles:
    repo ids ("org/name") and cache-style ids ("org--name" as served by OMLX).
    Returns the directory path or None."""
    candidates = []
    if "/" in model_id:
        candidates.append("models--" + model_id.replace("/", "--"))
    candidates.append("models--" + model_id)
    for c in candidates:
        d = os.path.join(HF_CACHE, c)
        if os.path.isdir(d):
            return d
    return None


def snapshot_dir(model_id):
    """Latest snapshot dir inside the cache entry (CLI --model flags need a
    real path; repo names only work as request ids)."""
    d = cache_dirname(model_id)
    if not d:
        return None
    hits = sorted(glob.glob(os.path.join(d, "snapshots", "*")))
    return hits[-1].rstrip("/") if hits else None


def _snapshot_size_gb(cache_dir):
    """Largest snapshot's total size in GB (models may have several)."""
    best = 0
    for snap in glob.glob(os.path.join(cache_dir, "snapshots", "*")):
        total = 0
        for dirpath, _dirnames, filenames in os.walk(snap):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        best = max(best, total)
    return round(best / _GB, 1) if best else None


def _context_from_cfg(cfg):
    """Context window from a config dict, when the model declares one. Checks
    the top level and the nested text_config (multimodal models nest the LM
    config there)."""
    for node in (cfg, cfg.get("text_config") or {}):
        if not isinstance(node, dict):
            continue
        for key in ("max_position_embeddings", "seq_len", "model_max_length"):
            v = node.get(key)
            if isinstance(v, int) and v > 0:
                return v
    return None


def _file_context(path):
    """Context window from a single config.json file (None if absent/invalid)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return _context_from_cfg(json.load(f))
    except (OSError, json.JSONDecodeError):
        return None


def _cached_context(cache_dir):
    """Context window from the cached config.json (largest snapshot)."""
    for p in sorted(glob.glob(os.path.join(cache_dir, "snapshots", "*", "config.json"))):
        ctx = _file_context(p)
        if ctx:
            return ctx
    return None


def _dir_size_gb(d):
    """Total size of a directory tree in GB (MTPLX models have no snapshots/)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(d):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return round(total / _GB, 1) if total else None


def normalize_key(model_id):
    """Canonical dedup key for a model id, in either style. Repo ids
    ("org/name") and cache-style ids ("org--name", as OMLX serves them) both
    map to the HF-cache directory name, so the same model de-dups across the
    served and local sources."""
    if "/" in model_id:
        return "models--" + model_id.replace("/", "--")
    return "models--" + model_id


def discover_served(port, timeout=4):
    """Live model list from a running OpenAI-compatible server."""
    data = _get_json(f"http://127.0.0.1:{port}/v1/models", timeout)
    if not data:
        return []
    out = []
    for m in data.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        out.append({
            "id": mid,
            "context_length": m.get("context_length"),
            "served": True,
            "in_cache": cache_dirname(mid) is not None,
        })
    return out


def _snapshot_complete(cache_dir):
    """True when ANY snapshot's shards match its weight index (None when no
    snapshot has an index to check). Catches half-downloaded models —
    observed twice: a 38 GB partial Speed build, and an interrupted re-
    download whose stub snapshot (index present, zero shards) HID the
    complete snapshot next to it from discovery — check every snapshot, a
    complete one anywhere means the model is usable."""
    any_index = False
    for p in sorted(glob.glob(os.path.join(cache_dir, "snapshots", "*",
                                           "model.safetensors.index.json"))):
        any_index = True
        try:
            with open(p) as f:
                index = json.load(f)
            needed = sorted(set(index.get("weight_map", {}).values()))
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        base = os.path.dirname(p)
        if all(os.path.isfile(os.path.join(base, s)) for s in needed):
            return True
    return False if any_index else None


# public aliases — server.py's extra-path scan shares these instead of
# duplicating them locally
dir_size_gb = _dir_size_gb
file_context = _file_context
cached_context = _cached_context


def discover_local():
    """All repos in the local HF cache, with size + context where known."""
    out = []
    if not os.path.isdir(HF_CACHE):
        return out
    for entry in sorted(os.listdir(HF_CACHE)):
        if not entry.startswith("models--"):
            continue
        d = os.path.join(HF_CACHE, entry)
        repo = entry[len("models--"):].replace("--", "/")
        # half-downloaded models (shards missing vs their weight index) must
        # not be offered as if runnable
        if _snapshot_complete(d) is False:
            continue
        size = _snapshot_size_gb(d)
        # skip helper artifacts: draft heads / embedders are not runnable
        # language models (observed: the 0.4 GB MTP-8bit draft head was
        # selectable and produced zero-token completions)
        if size is not None and size < 1.0:
            continue
        name_l = repo.lower()
        if "draft" in name_l or name_l.endswith(("-embed", "-embedder")):
            continue
        out.append({
            "id": repo,
            "served": False,
            "in_cache": True,
            "size_gb": size,
            "context_length": _cached_context(d),
            "moe": _cached_moe(d)[0],
            "moe_detail": _cached_moe(d)[1],
        })
    return out


def discover_mtplx():
    """Models in MTPLX's own store (~/.mtplx/models). Each dir is a cache-style
    name (org--name) holding the weights plus .mtplx-source.json (the original
    repo_id) and config.json (context). Only MTPLX can serve these — they are a
    different format (MTP sidecar + runtime) than plain HF/MLX repos."""
    out = []
    if not os.path.isdir(MTPLX_MODELS):
        return out
    for entry in sorted(os.listdir(MTPLX_MODELS)):
        d = os.path.join(MTPLX_MODELS, entry)
        if not os.path.isdir(d):
            continue
        repo = None
        src = os.path.join(d, ".mtplx-source.json")
        if os.path.isfile(src):
            try:
                with open(src) as f:
                    repo = json.load(f).get("repo_id")
            except (OSError, json.JSONDecodeError):
                pass
        if not repo:
            # store dirs may be cache-style (org--name) or HF-cache-style
            # (models--org--name, as mtplx pull creates) — parse both
            stripped = entry[len("models--"):] if entry.startswith("models--") else entry
            repo = stripped.replace("--", "/")
        out.append({
            "id": repo,
            "source": "mtplx",
            "served": False,
            "in_cache": True,
            "size_gb": _dir_size_gb(d),
            "context_length": _file_context(os.path.join(d, "config.json")),
        })
    return out


def mtp_draft_for(model_id, local_ids):
    """Heuristic: a cached MTP draft matching the base model (speculative
    decoding). Returns the draft repo id or None."""
    base = model_id.split("/")[-1]
    for lid in local_ids:
        short = lid.split("/")[-1]
        if short != base and short.startswith(base) and "mtp" in short.lower():
            return lid
    return None



def machine_profile():
    """What THIS machine can run: chip name, total RAM and the practical
    Metal wired-limit estimate for model + KV residency. Deliberately
    machine-independent: verdicts scale from a 36 GB M1 to a 512 GB studio
    because every verdict is computed against the detected ceiling."""
    chip, total_gb = "Apple Silicon", None
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=5
                              ).stdout.strip() or chip
        total_gb = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                            capture_output=True, text=True,
                                            timeout=5).stdout.strip()) / _GB, 1)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    if not total_gb:
        return {"chip": chip, "total_gb": None, "usable_gb": None}
    # macOS wired-limit defaults: ~75% of RAM on small machines, rising with
    # capacity (observed: 121.6 GB on a 128 GB M5 Max). Estimate, not gospel.
    frac = 0.95 if total_gb >= 64 else (0.85 if total_gb >= 48 else 0.75)
    return {"chip": chip, "total_gb": total_gb,
            "usable_gb": round(total_gb * frac, 1),
            "note": "usable = estimated Metal wired limit for model + KV"}


_QUANT_TOKENS = re.compile(
    r"(?i)(\d+bit|\d+bit-?mlx|oq\d+e?|bf16|fp16|fp8|int\d+|mlx|mlxl|mtp|optimized|"
    r"instruct|it\b|speed|quality|bare|gguf|exl\d|nvfp4|gsq|rcq|heretic|uncensored|"
    r"turbo|splash|draft|snapshot\d*)")


def family_for(model_id):
    """Base family name: the model id with quant/runtime packaging tokens
    removed, so every quant of one model groups together ('Qwen3.8-Flash-Next
    4bit/5bit/Speed' all land in one family)."""
    name = model_id.split("/")[-1]
    name = re.sub(r"(?i)^[a-z0-9]+[-—]?(Qwen|Llama|Gemma|Mistral)", r"\1", name)
    tokens = re.split(r"[-_ ]", name)
    keep = [t for t in tokens if not _QUANT_TOKENS.match(t)]
    core = "-".join(keep) or name
    return re.sub(r"(?i)[-_]?(MLX|MTPLX)$", "", core) or name


def _moe_from_cfg(cfg):
    """MoE detection from a config dict: returns (is_moe, detail)."""
    for node in (cfg, cfg.get("text_config") or {}):
        if not isinstance(node, dict):
            continue
        for key in ("num_experts", "n_routed_experts", "num_local_experts",
                    "n_experts", "moe_num_experts"):
            v = node.get(key)
            if isinstance(v, int) and v > 1:
                return True, f"{v} experts"
    return False, None


def _cached_moe(cache_dir):
    for p in sorted(glob.glob(os.path.join(cache_dir, "snapshots", "*", "config.json"))):
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        moe, detail = _moe_from_cfg(cfg)
        if moe:
            return moe, detail
    return False, None



def compatibility(cand, free_ram_gb, ctx_tokens, machine=None):
    """Score one candidate model. Returns {verdict, reasons[]}.
    verdict: ready | tight | too-large | unknown
      ready     — fits RAM, context OK, locally available
      tight     — fits, but declared context < configured window
      too-large — weights + KV headroom exceed free RAM
      unknown   — not locally cached and not currently served (download needed)
    """
    reasons = []
    verdict = "ready"
    size = cand.get("size_gb")
    if size and free_ram_gb:
        need = size * RAM_HEADROOM
        if need > free_ram_gb:
            verdict = "too-large"
            reasons.append(f"~{size} GB weights (+KV headroom) > {free_ram_gb:.0f} GB free")
    ctx = cand.get("context_length")
    if ctx and ctx_tokens and ctx < ctx_tokens:
        if verdict == "ready":
            verdict = "tight"
        reasons.append(f"context {ctx:,} < configured {ctx_tokens:,}")
    if not cand.get("in_cache") and not cand.get("served"):
        if verdict == "ready":
            verdict = "unknown"
        reasons.append("not in local cache — framework must download on start")
    # machine fit: verdict against the machine's wired ceiling — stable across
    # sessions, unlike free RAM (which drops whenever another model is loaded)
    machine_fit = None
    size = cand.get("size_gb")
    if machine and machine.get("usable_gb") and size:
        need = round(size * 1.1 + 2.0, 1)   # weights + KV/overhead estimate
        cap = machine["usable_gb"]
        machine_fit = ("fits" if need <= cap * 0.85
                       else "tight" if need <= cap else "wont-fit")
        cand["need_gb"] = need
        if machine_fit == "wont-fit":
            reasons.append(f"~{need} GB needed > {cap} GB usable on this machine")
        elif machine_fit == "tight":
            reasons.append(f"tight fit: ~{need} GB vs {cap} GB usable")
    return {"verdict": verdict, "reasons": reasons, "machine_fit": machine_fit}


def all_candidates(free_ram_gb, frameworks, machine=None):
    """Unified, de-duplicated candidate list across ALL sources (HF cache +
    MTPLX store + anything currently served), each tagged with the list of
    frameworks it is compatible with. Compatibility is by model source: a
    framework's "model_source" ("hf" or "mtplx") must match the model's source.
    The UI uses this to show every model and grey out the ones a selected
    framework can't serve (e.g. MTPLX-only models for OMLX/MLX-VLM/MLX-Serve).

    frameworks: the full {id: cfg} map (so sources can be computed globally).
    Returns a list of candidate dicts, each with: id, source, frameworks[],
    served, in_cache, size_gb, context_length, compat, mtp_draft."""
    hf_fws = [fw for fw, c in frameworks.items()
              if c.get("model_source", "hf") == "hf"]
    mtp_fws = [fw for fw, c in frameworks.items()
               if c.get("model_source") == "mtplx"]
    by_key = {}
    for m in discover_local():
        m["source"] = "hf"
        m["frameworks"] = list(hf_fws)
        by_key[normalize_key(m["id"])] = m
    for m in discover_mtplx():
        m["source"] = "mtplx"
        k = normalize_key(m["id"])
        e = by_key.get(k)
        if e is None:
            m["frameworks"] = list(mtp_fws)
            by_key[k] = m
        else:
            # The same weights are present in BOTH registries (e.g. a model
            # pulled by mtplx and downloaded to the HF cache). Union the
            # framework tags so neither side greys it out, keep the existing
            # entry's metadata, and mark the dual presence.
            e["frameworks"] = sorted(set(e.get("frameworks", [])) | set(mtp_fws))
            e["in_mtplx_store"] = True
    # Merge in whatever each framework is serving right now (authoritative
    # while it runs). A served id that matches a local model just flags it;
    # an unknown served id is added, compatible with the serving framework.
    for fw, cfg in frameworks.items():
        port = cfg.get("port")
        if not port:
            continue
        for m in discover_served(port):
            k = normalize_key(m["id"])
            e = by_key.get(k)
            if e is None:
                e = dict(m)
                e["source"] = "served"
                e["frameworks"] = [fw]
                by_key[k] = e
            else:
                e["served"] = True
                if fw not in e["frameworks"]:
                    e["frameworks"].append(fw)
    # Score each candidate. Context window is per-framework; use the first
    # compatible framework's configured window (they normally agree).
    local_ids = [m["id"] for m in discover_local()]
    out = []
    for m in by_key.values():
        ctx_tokens = None
        for fw in m.get("frameworks", []):
            ctx_tokens = frameworks[fw].get("ctx_tokens")
            if ctx_tokens:
                break
        m["compat"] = compatibility(m, free_ram_gb, ctx_tokens, machine)
        m["family"] = family_for(m["id"])
        m["mtp_draft"] = mtp_draft_for(m["id"], local_ids) if m.get("source") == "hf" else None
        out.append(m)
    rank = {"ready": 0, "tight": 1, "unknown": 2, "too-large": 3}
    out.sort(key=lambda m: (0 if m.get("served") else 1,
                            rank.get(m["compat"]["verdict"], 9), m["id"]))
    return out
