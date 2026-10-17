"""Regression tests for benchtest's pure functions.

The four confirmed review bugs (double HTTP response per POST, the
start_framework reuse/restart mis-wiring, the proxy_reset KeyError that ate
truncated cells' stats, and the pipe-decode hang) all lived in code these
tests now pin down. BENCHTEST_NO_STARTUP=1 — set BELOW, before importing
server — makes the import side-effect-free (no directories, no config.json
seeding or rewriting), so the suite is safe to run while a live backend
owns config.json.

    python3 -m unittest tests.test_pure -v     (or: make check)
"""
import json
import os
import sys
import tempfile
import time
import unittest

os.environ["BENCHTEST_NO_STARTUP"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import server          # noqa: E402  (env var must be set first)
import discovery       # noqa: E402
import qa              # noqa: E402


class TestHtmlExtraction(unittest.TestCase):
    def test_labeled_fence(self):
        doc = ("<!DOCTYPE html><html><body><h1>hi</h1>"
               + "<p>" + "padding to clear the 120-char plausibility floor " * 2
               + "</p></body></html>")
        out = server.extract_html(f"prose\n```html\n{doc}\n```\nmore prose")
        self.assertTrue(out and out.lstrip().startswith("<!DOCTYPE"))

    def test_bare_html_block(self):
        doc = ("<html><body><canvas id='c'></canvas>"
               + "<script>var x=1;" + "/* padding */" * 12 + "</script>"
               + "</body></html>")
        out = server.extract_html(f"here you go:\n{doc}")
        self.assertTrue(out and out.startswith("<html"))

    def test_prose_only_returns_none(self):
        self.assertIsNone(server.extract_html("just words, no markup at all"))


class TestSeamOverlap(unittest.TestCase):
    def test_detects_repeat(self):
        # raw+ continuation: the model restarted the last line — trim it
        # overlap must exceed min_overlap (8): shorter repeats are noise
        self.assertEqual(server._seam_overlap("xxOVERLAP99", "OVERLAP99yy"), 9)

    def test_no_repeat_is_zero(self):
        self.assertEqual(server._seam_overlap("abcdefgh", "zzzzzzzz"), 0)


class TestHarnessStream(unittest.TestCase):
    def test_pi_iterations_count_assistant_messages(self):
        events = "\n".join([
            json.dumps({"type": "message_start"}),
            json.dumps({"type": "message_end", "message": {"role": "user"}}),
            json.dumps({"type": "message_end",
                        "message": {"role": "assistant", "content": []}}),
            json.dumps({"type": "message_end",
                        "message": {"role": "assistant", "content": []}}),
            json.dumps({"type": "agent_end", "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}],
                 "usage": {"input": 10, "output": 2}}]}),
        ])
        text, tokens, met = server.finalize_harness_stream("pi", events)
        self.assertEqual(met.get("calls"), 2)   # assistant responses only

    def test_pi_agent_end(self):
        ev = {"type": "agent_end", "messages": [
            {"role": "user", "content": "build it"},
            {"role": "assistant",
             "content": [{"type": "text", "text": "done building"}],
             "usage": {"input": 120, "output": 34}}]}
        text, tokens, met = server.finalize_harness_stream(
            "pi", json.dumps(ev))
        self.assertEqual(text, "done building")
        self.assertEqual(tokens, 34)

    def test_opencode_stream(self):
        lines = "\n".join([
            json.dumps({"type": "text", "part": {"text": "hello "}}),
            json.dumps({"type": "text", "part": {"text": "world"}}),
            json.dumps({"type": "step_finish",
                        "part": {"tokens": {"input": 50, "output": 9}}}),
        ])
        text, tokens, met = server.finalize_harness_stream("opencode", lines)
        self.assertEqual(text, "hello world")
        self.assertEqual(tokens, 9)

    def test_non_json_passthrough(self):
        text, tokens, met = server.finalize_harness_stream("hart", "plain")
        self.assertEqual((text, tokens, met), ("plain", None, {}))


class TestServerCellDelta(unittest.TestCase):
    def test_omlx_delta(self):
        before = {"kind": "omlx", "prompt": 100, "completion": 10,
                  "pre_s": 5.0, "gen_s": 4.0}
        after = {"kind": "omlx", "prompt": 2100, "completion": 410,
                 "pre_s": 7.0, "gen_s": 9.0}
        d = server.server_cell_delta(None, before, after)
        self.assertEqual(d["server_pp"], 1000.0)   # 2000 tok / 2.0 s
        self.assertEqual(d["server_tgs"], 80.0)    # 400 tok / 5.0 s
        self.assertNotIn("server_pp_flagged", d)

    def test_omlx_implausible_value_kept_but_flagged(self):
        before = {"kind": "omlx", "prompt": 0, "completion": 0,
                  "pre_s": 0.0, "gen_s": 0.0}
        after = {"kind": "omlx", "prompt": 500_000, "completion": 10,
                 "pre_s": 2.0, "gen_s": 3.0}
        d = server.server_cell_delta(None, before, after)
        self.assertEqual(d["server_pp"], 250000.0)   # kept, not discarded
        self.assertTrue(d["server_pp_flagged"])

    def test_mlxvlm_token_weighted_beats_mean(self):
        # one tiny fast request + one big slow one: the old mean-of-rates
        # overvalued the tiny one; tokens/seconds weights correctly
        before = {"kind": "mlxvlm", "prompt": 0, "completion": 0, "recent": []}
        after = {"kind": "mlxvlm", "prompt": 2200, "completion": 60,
                 "recent": [
                     {"ts": 1, "prefill_tok_s": 5000.0, "prompt_tokens": 100,
                      "prefill_s": 0.02},
                     {"ts": 2, "prefill_tok_s": 1000.0, "prompt_tokens": 2000,
                      "prefill_s": 2.0}]}
        d = server.server_cell_delta(None, before, after)
        self.assertEqual(d["server_pp"], 1039.6)    # 2100 tok / 2.02 s
        self.assertNotEqual(d["server_pp"], 3000.0)  # the old mean-of-rates
        self.assertEqual(d["server_pp_method"], "weighted")

    def test_mlxvlm_mean_fallback_without_counters(self):
        before = {"kind": "mlxvlm", "prompt": 0, "completion": 0, "recent": []}
        after = {"kind": "mlxvlm", "prompt": 100, "completion": 10,
                 "recent": [{"ts": 1, "prefill_tok_s": 800.0},
                            {"ts": 2, "prefill_tok_s": 1200.0}]}
        d = server.server_cell_delta(None, before, after)
        self.assertEqual(d["server_pp"], 1000.0)
        self.assertEqual(d["server_pp_method"], "mean-of-requests")

    def test_fresh_requests_identity_by_timestamp(self):
        # identical back-to-back requests must not collapse when the server
        # exposes a per-request timestamp
        before = [{"ts": 1, "decode_tok_s": 70.0}]
        after = [{"ts": 1, "decode_tok_s": 70.0},
                 {"ts": 2, "decode_tok_s": 70.0}]
        fresh = server._fresh_requests(before, after)
        self.assertEqual(len(fresh), 1)

    def test_mtplx_delta(self):
        before = {"kind": "mtplx", "recent": [], "latest": None}
        after = {"kind": "mtplx", "recent": [
            {"ts": 1, "prompt_tps": 900.0, "generation_tps": 60.0,
             "ttft_s": 0.4}], "latest": None}
        d = server.server_cell_delta(None, before, after)
        self.assertEqual(d["server_requests"], 1)
        self.assertEqual(d["server_tgs"], 60.0)


class TestProxyStats(unittest.TestCase):
    def setUp(self):
        server.proxy_reset()

    def test_reset_then_length_finish_no_keyerror(self):
        # C3 regression: proxy_reset once dropped length_hits from the dict,
        # so the FIRST truncated request after it raised KeyError inside the
        # lock and lost that cell's stats
        server._proxy_record(None, time.monotonic(), None, None,
                             "m", "/v1/chat/completions", True, 200, "length")
        st = server.proxy_read()
        self.assertEqual(st["requests"], 1)
        self.assertEqual(st["length_hits"], 1)

    def test_cached_prompt_tokens_recorded(self):
        usage = {"prompt_tokens": 500, "completion_tokens": 10,
                 "prompt_tokens_details": {"cached_tokens": 400}}
        server._proxy_record(usage, time.monotonic(), None, None,
                             "m", "/v1/chat/completions", False, 200, "stop")
        st = server.proxy_read()
        self.assertEqual(st["cached_prompt_tokens"], 400)

    def test_proxy_zero_covers_all_stats_keys(self):
        # every key the record path touches must survive a reset
        server.proxy_reset()
        self.assertEqual(set(server.PROXY_STATS), set(server.PROXY_ZERO))
        self.assertEqual(server.PROXY_STATS["cached_prompt_tokens"], 0)


class TestStartCmd(unittest.TestCase):
    def test_model_and_reasoning_placeholders(self):
        cfg = {"start_cmd": ["srv", "--model", "{model}", "{reasoning}"],
               "model": "org--name", "repo": "org/name",
               "reasoning_flags": {"low": ["--effort", "low"]}}
        cmd = server.resolve_start_cmd(cfg, "low")
        self.assertEqual(cmd[:3], ["srv", "--model", "org--name"])
        self.assertIn("--effort", cmd)
        self.assertNotIn("{reasoning}", cmd)

    def test_draft_expansion(self):
        cfg = {"start_cmd": ["srv", "--model", "{model}", "{draft}"],
               "model": "m", "draft_model": "tiny-draft"}
        cmd = server.resolve_start_cmd(cfg)
        self.assertIn("--draft-model", cmd)
        self.assertIn("tiny-draft", cmd)


class TestNormalizeKey(unittest.TestCase):
    def test_repo_and_cache_styles_dedup(self):
        self.assertEqual(discovery.normalize_key("org/name"),
                         discovery.normalize_key("org--name"))

    def test_plain_name_maps_to_cache_form(self):
        self.assertEqual(discovery.normalize_key("model"),
                         "models--model")


class TestNoDuplicateDefs(unittest.TestCase):
    def test_server_has_no_duplicate_top_level_defs(self):
        # a patch once left two free_ram definitions; Python's last-wins
        # silently ran the OLD one. Ban duplicate top-level defs entirely.
        import ast as _ast
        tree = _ast.parse(open(os.path.join(ROOT, "server.py")).read())
        names = [n.name for n in tree.body
                 if isinstance(n, (_ast.FunctionDef, _ast.ClassDef))]
        dupes = {n for n in names if names.count(n) > 1}
        self.assertEqual(dupes, set())


class TestQa(unittest.TestCase):
    def _html(self, tmp, body, name="app.html"):
        import tempfile as _t
        p = os.path.join(tmp, name)
        with open(p, "w") as f:
            f.write(body)
        return p

    def test_full_pass_artifact(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as d:
            p = self._html(d, "<!DOCTYPE html><html><body>" + "x" * 6000
                           + "<div>12/12 passed</div></body></html>")
            r = qa.evaluate({"status": "done"}, p)
        self.assertEqual(r["qa_score"], 100)
        self.assertIn("12/12 passed", r["qa_notes"])

    def test_truncated_html_loses_complete_and_untruncated(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as d:
            p = self._html(d, "<!DOCTYPE html><html><body>" + "x" * 6000)
            r = qa.evaluate({"status": "done", "truncated": True}, p)
        self.assertEqual(r["qa_score"], 57)   # 40 earned / 70 normalized
        self.assertTrue(any("cut off" in n for n in r["qa_notes"].split("; ")))

    def test_missing_artifact_scores_zero_delivery(self):
        r = qa.evaluate({"status": "done"}, "/nonexistent/x.html")
        self.assertLessEqual(r["qa_score"], 30)
        self.assertIn("missing on disk", r["qa_notes"])

    def test_error_row_is_zero(self):
        r = qa.evaluate({"status": "error", "error": "timeout"}, None)
        self.assertEqual(r["qa_score"], 0)
        self.assertIn("timeout", r["qa_notes"])

    def test_text_artifact_without_selftest_normalizes_to_full(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as d:
            p = self._html(d, "# Report\n\n" + "word " * 500, "report.md")
            r = qa.evaluate({"status": "done"}, p)
        # clean artifact without a self-test line: 70 earned / 70 → 100
        self.assertEqual(r["qa_score"], 100)
        self.assertIn("normalized over 70", r["qa_notes"])


class TestSnapshotComplete(unittest.TestCase):
    def _mk(self, root, snap, shards=2):
        d = os.path.join(root, "snapshots", snap)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {f"w{i}": f"model-0000{i}-of-0000{shards}.safetensors"
                                      for i in range(shards)}}, f)
        return d

    def test_stub_snapshot_does_not_hide_complete_one(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as root:
            self._mk(root, "a-stub")                    # index, no shards
            d = self._mk(root, "b-complete")
            for i in range(2):
                open(os.path.join(d, f"model-0000{i}-of-00002.safetensors"), "w").close()
            self.assertTrue(discovery._snapshot_complete(root))

    def test_only_stub_is_incomplete(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as root:
            self._mk(root, "a-stub")
            self.assertFalse(discovery._snapshot_complete(root))

    def test_no_index_is_unknown(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as root:
            self.assertIsNone(discovery._snapshot_complete(root))


class TestEnsureSingleModel(unittest.TestCase):
    def setUp(self):
        self.cfg = server.FRAMEWORKS["omlx"]
        self._orig = self.cfg["model"]

    def tearDown(self):
        self.cfg["model"] = self._orig

    def test_non_omlx_is_always_true(self):
        self.assertTrue(server.ensure_single_model("mtplx"))

    def test_resident_selection_short_circuits(self):
        from unittest.mock import patch
        m = "Org--Sel-Model"
        self.cfg["model"] = m
        with patch.object(server, "_omlx_resident_ids", return_value=[m]), \
             patch.object(server, "_omlx_discovered_ids", return_value=[]):
            self.assertTrue(server.ensure_single_model("omlx"))

    def test_undiscovered_selection_aborts_loudly(self):
        from unittest.mock import patch
        self.cfg["model"] = "Org--Ghost-Model"
        with patch.object(server, "_omlx_resident_ids", return_value=["Other--x"]), \
             patch.object(server, "_omlx_discovered_ids", return_value=["Other--x"]):
            self.assertFalse(server.ensure_single_model("omlx"))


class TestWeightedRateHeterogeneous(unittest.TestCase):
    def test_mixed_records_fall_back_to_mean_not_crash(self):
        records = [
            {"prompt_tokens": 100, "prefill_s": 0.2, "prefill_tok_s": 500.0},
            {"prefill_tok_s": 300.0},   # no counters — old code raised
        ]
        rate, method = server._weighted_rate(
            records, ("prefill_tok_s",),
            ("prompt_tokens", "prefill_tokens"), ("prefill_s", "prefill_seconds"))
        self.assertEqual(method, "mean-of-requests")
        self.assertEqual(rate, 400.0)

    def test_all_counters_still_weight(self):
        records = [
            {"prompt_tokens": 100, "prefill_s": 0.2},
            {"prompt_tokens": 300, "prefill_s": 1.0},
        ]
        rate, method = server._weighted_rate(
            records, ("prefill_tok_s",),
            ("prompt_tokens",), ("prefill_s",))
        self.assertEqual(method, "weighted")
        self.assertEqual(rate, round(400 / 1.2, 1))


class TestHartGuards(unittest.TestCase):
    """The sibling-prefix escape (SEC-003) — hart's execute_action guards."""

    def _exec(self, workdir, action, args):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hart_mod", os.path.join(ROOT, "hart", "hart.py"))
        hart = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hart)
        return hart.execute_action({"workdir": workdir, "artifacts": [],
                                   "observations": [], "_step_writes": {}},
                                   action, args)

    def test_write_rejects_prefix_sibling(self):
        import tempfile as _t
        with _t.TemporaryDirectory(suffix="_wd") as wd:
            sib = wd + "_evil"
            os.makedirs(sib, exist_ok=True)
            r = self._exec(wd, "write_file",
                           {"path": os.path.join("..", os.path.basename(sib), "x.txt"),
                            "content": "x" * 20})
        self.assertIn("error", r)

    def test_write_accepts_inside(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as wd:
            r = self._exec(wd, "write_file",
                           {"path": "ok.txt", "content": "x" * 20})
            self.assertNotIn("error", r)
            self.assertTrue(os.path.isfile(os.path.join(wd, "ok.txt")))


class TestMigrateMlxlm(unittest.TestCase):
    def test_renames_framework_and_set_keys_preserving_order(self):
        cfg = {"frameworks": {"omlx": {}, "mlxlm": {"port": 7003}, "mtplx": {}},
               "model_sets": {"s": {"models": {"mlxlm": "org/name",
                                               "omlx": "other"}}}}
        out = server._migrate_mlxlm(cfg)
        self.assertEqual(list(out["frameworks"]), ["omlx", "mlxvlm", "mtplx"])
        self.assertEqual(out["frameworks"]["mlxvlm"]["port"], 7003)
        self.assertEqual(out["model_sets"]["s"]["models"]["mlxvlm"], "org/name")

    def test_noop_without_mlxlm(self):
        cfg = {"frameworks": {"omlx": {}}}
        self.assertEqual(server._migrate_mlxlm(cfg)["frameworks"], {"omlx": {}})


class TestTasks(unittest.TestCase):
    def test_unique_task_ids(self):
        ids = [t["id"] for t in server.TASKS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_fixture_files_exist(self):
        for task, (src, _dst) in server.TASK_FIXTURES.items():
            path = os.path.join(server.ROOT, "fixtures", src)
            self.assertTrue(os.path.isfile(path),
                            f"fixture for {task!r} missing: {path}")


class TestNewestHtml(unittest.TestCase):
    def test_newer_stub_does_not_hide_real_artifact(self):
        # P0-4 regression: only the newest .html was plausibility-checked, so
        # a stray stub newer than the real artifact failed a built cell
        import tempfile as _t
        with _t.TemporaryDirectory() as d:
            real = os.path.join(d, "app.html")
            with open(real, "w") as f:
                f.write("<!DOCTYPE html><html><body>" + "x" * 200
                        + "</body></html>")
            os.utime(real, (1000, 1000))
            stub = os.path.join(d, "zz_stub.html")
            with open(stub, "w") as f:
                f.write("<html>stub</html>")
            os.utime(stub, (2000, 2000))
            self.assertEqual(server.newest_html(d, 0), real)


class TestScanFolderModels(unittest.TestCase):
    def _mk(self, root, name, with_cfg=True):
        import tempfile as _t
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        if with_cfg:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"max_position_embeddings": 262144}, f)
        return d

    def test_plain_dirs(self):
        import tempfile as _t
        from unittest.mock import patch
        with _t.TemporaryDirectory() as root:
            self._mk(root, "my-model-a")
            self._mk(root, "my-model-b")
            self._mk(root, "notes", with_cfg=False)   # no config.json → skip
            with patch.object(discovery, "dir_size_gb", return_value=5.0), \
                 patch.object(discovery, "file_context", return_value=262144):
                out = server._scan_folder_models(root)
        self.assertEqual([m["id"] for m in out], ["my-model-a", "my-model-b"])

    def test_hf_hub_layout(self):
        import tempfile as _t
        from unittest.mock import patch
        with _t.TemporaryDirectory() as root:
            entry = self._mk(root, "models--org--name")
            snap = os.path.join(entry, "snapshots", "abc123")
            os.makedirs(snap)
            with open(os.path.join(snap, "config.json"), "w") as f:
                json.dump({}, f)
            with patch.object(discovery, "_snapshot_size_gb", return_value=5.0):
                out = server._scan_folder_models(root)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "org/name")
        self.assertEqual(out[0]["path"], os.path.realpath(snap))

    def test_folder_that_is_itself_a_model(self):
        import tempfile as _t
        from unittest.mock import patch
        with _t.TemporaryDirectory() as root:
            # the root itself is a model: config.json is a FILE here
            with open(os.path.join(root, "config.json"), "w") as f:
                json.dump({}, f)
            with patch.object(discovery, "dir_size_gb", return_value=5.0):
                out = server._scan_folder_models(root)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], os.path.basename(root))

    def test_draft_sized_entries_filtered(self):
        import tempfile as _t
        from unittest.mock import patch
        with _t.TemporaryDirectory() as root:
            self._mk(root, "tiny-draft-head")
            with patch.object(discovery, "dir_size_gb", return_value=0.4):
                self.assertEqual(server._scan_folder_models(root), [])


class TestPersistenceHelpers(unittest.TestCase):
    def test_atomic_write_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            server._atomic_write_json(path, {"a": 1})
            server._atomic_write_json(path, {"b": 2})
            with open(path) as f:
                self.assertEqual(json.load(f), {"b": 2})
            self.assertEqual(os.listdir(d), ["settings.json"])  # no tmp litter

    def test_jsonl_cell_writes_without_text(self):
        with tempfile.TemporaryDirectory() as d:
            server._RUN_JSONL["path"] = os.path.join(d, "cells.jsonl")
            try:
                server._jsonl_cell({"framework": "omlx", "status": "done",
                                    "_text": "secret"})
                server._jsonl_cell({"framework": "mtplx", "status": "error"})
                with open(server._RUN_JSONL["path"]) as f:
                    rows = [json.loads(l) for l in f]
            finally:
                server._RUN_JSONL["path"] = None
            self.assertEqual(len(rows), 2)
            self.assertNotIn("_text", rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
