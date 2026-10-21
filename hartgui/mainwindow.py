"""The hart main window.

Layout:
  ┌ left panel ──────────┐  ┌ center ────────────────────────────────┐
  │ BACKENDS (+/discover)│  │ GOAL (prompt)                          │
  │ STAGE MODELS (per stg)│ │ PLANNING: single | parallel(N) + judge │
  │ CACHE (stage I/O)    │  │ DOUBLE-VALIDATE: [stages]              │
  └──────────────────────┘  │ [ RUN ] [ Stop ]                       │
                            │ PIPELINE: ● connection → ● planning …  │
                            │ PLAN COMPARISON (side-by-side, ★ best) │
                            │ OUTPUT / LOG (live narration)          │
                            └────────────────────────────────────────┘
"""

import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSplitter, QStatusBar,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from hartgui import theme
from hartgui.pipelinebar import PipelineBar

from hart import providers
from hartgui import best_of_n
from hartgui.model_config import Backend, BackendManager
from hartgui.workers import DiscoverWorker, PipelineWorker

STAGES = ["planning", "building", "validation", "evaluation"]


class MainWindow(QMainWindow):
    def __init__(self, backends_path=None):
        super().__init__()
        self.setWindowTitle("hart — multi-agent harness")
        self.resize(1240, 800)
        self.setMinimumSize(940, 600)

        self.backends = BackendManager(backends_path)
        self.workdir = os.path.join(os.getcwd(), "hart-gui-work")
        self._model_lookup = {}      # "backend/model" -> (Backend, model_id)
        self._worker = None
        self._discover_workers = []
        self._last_states = {}         # stage → last state (theme repaint)

        self._build_ui()
        self._populate_model_combos()
        self._set_status("ready")

    # ── UI construction ──────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(10, 8, 10, 8)
        root.addLayout(body, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._wrap_scroll(self._build_left_panel(), 300))
        splitter.addWidget(self._build_center())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 880])
        splitter.setChildrenCollapsible(False)
        body.addWidget(splitter)
        self.setStatusBar(QStatusBar())

    def _wrap_scroll(self, widget, minw):
        """Scrollable wrapper — the fix for the not-resizable UI: at any
        window size the panel content scrolls instead of clipping."""
        sc = QScrollArea()
        sc.setWidget(widget)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.NoFrame)
        sc.setMinimumWidth(minw)
        return sc

    def _build_header(self):
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(16, 10, 16, 10)
        title = QLabel("⚡ hart")
        title.setProperty("h1", True)
        self.header_stage = QLabel("")
        self.header_stage.setStyleSheet(
            f"color: {theme.FG}; font-weight: 700; background: {theme.PANEL2};"
            f"border: 1px solid {theme.BORDER}; border-radius: 9px;"
            f"padding: 3px 12px;")
        sub = QLabel("multi-agent build harness — plan · build · validate · evaluate")
        sub.setProperty("muted", True)
        self.header_status = QLabel("● ready")
        self.header_status.setStyleSheet(
            f"color: {theme.current()['green']}; font-weight: 700; background: transparent;")
        self.theme_btn = QPushButton("☀ light")
        self.theme_btn.setToolTip("Toggle dark / light theme")
        self.theme_btn.clicked.connect(self._on_toggle_theme)
        row.addWidget(title)
        row.addWidget(self.header_stage)
        row.addSpacing(10)
        row.addWidget(sub)
        row.addStretch(1)
        row.addWidget(self.header_status)
        row.addWidget(self.theme_btn)
        return w

    def _build_left_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._build_backends_group())
        lay.addWidget(self._build_stage_group())
        lay.addStretch(1)
        return w

    def _build_backends_group(self):
        g = QGroupBox("Backends (model hosts)")
        lay = QVBoxLayout(g)
        hint = QLabel("Add a backend, click it in the list, then Discover.")
        hint.setProperty("muted", True)
        lay.addWidget(hint)
        self.backend_list = QListWidget()
        self.backend_list.setMinimumHeight(76)
        self.backend_list.currentItemChanged.connect(self._on_backend_selected)
        lay.addWidget(self.backend_list)

        form = QGridLayout()
        self.backend_name = QLineEdit()
        self.backend_name.setPlaceholderText("name (e.g. omlx / openai)")
        self.backend_url = QLineEdit()
        self.backend_url.setPlaceholderText("http://127.0.0.1:7001/v1")
        self.backend_provider = QComboBox()
        self.backend_provider.addItems(["openai (OpenAI-compatible)", "anthropic"])
        self.backend_provider.currentIndexChanged.connect(self._on_provider_changed)
        self.backend_model = QLineEdit()
        self.backend_model.setPlaceholderText("model id (optional)")
        self.backend_key = QLineEdit()
        self.backend_key.setEchoMode(QLineEdit.Password)
        self.backend_key.setPlaceholderText("API key (optional)")
        self.backend_key_env = QLineEdit()
        self.backend_key_env.setPlaceholderText("…or env var name (e.g. OPENAI_API_KEY)")
        form.addWidget(QLabel("Name"), 0, 0)
        form.addWidget(self.backend_name, 0, 1)
        form.addWidget(QLabel("URL"), 1, 0)
        form.addWidget(self.backend_url, 1, 1)
        form.addWidget(QLabel("Provider"), 2, 0)
        form.addWidget(self.backend_provider, 2, 1)
        form.addWidget(QLabel("Model"), 3, 0)
        form.addWidget(self.backend_model, 3, 1)
        form.addWidget(QLabel("API key"), 4, 0)
        form.addWidget(self.backend_key, 4, 1)
        form.addWidget(QLabel("Key env"), 5, 0)
        form.addWidget(self.backend_key_env, 5, 1)
        lay.addLayout(form)

        btns = QHBoxLayout()
        save = QPushButton("Add / Update")
        save.clicked.connect(self._on_save_backend)
        disc = QPushButton("Discover")
        disc.clicked.connect(self._on_discover)
        rm = QPushButton("Remove")
        rm.clicked.connect(self._on_remove_backend)
        btns.addWidget(save)
        btns.addWidget(disc)
        btns.addWidget(rm)
        lay.addLayout(btns)
        self._on_provider_changed()
        self._refresh_backend_list()
        return g

    def _build_stage_group(self):
        g = QGroupBox("Stage models")
        lay = QVBoxLayout(g)
        self.same_model_cb = QCheckBox("Use one model for every stage")
        self.same_model_cb.toggled.connect(self._on_same_model_toggled)
        lay.addWidget(self.same_model_cb)
        self.primary_combo = QComboBox()
        lay.addWidget(QLabel("Primary (all stages):"))
        lay.addWidget(self.primary_combo)
        self.stage_combos = {}
        for stage in STAGES:
            c = QComboBox()
            c.setEnabled(False)      # enabled only when "same model" is off
            self.stage_combos[stage] = c
            lay.addWidget(QLabel(f"{stage}:"))
            lay.addWidget(c)
        return g

    def _build_center(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # live tail — TOP, dominant panel
        tail_head = QHBoxLayout()
        tail_head.addWidget(QLabel("<b>Live tail</b> — plan · build · validate · evaluate"))
        self.autoscroll_cb = QCheckBox("follow")
        self.autoscroll_cb.setChecked(True)
        tail_head.addStretch(1)
        tail_head.addWidget(self.autoscroll_cb)
        outer.addLayout(tail_head)
        from PySide6.QtWidgets import QTextBrowser
        self.log = QTextBrowser()
        self.log.setOpenExternalLinks(False)
        logfont = QFont("Menlo", 11)
        self.log.setFont(logfont)
        self.log.setPlaceholderText("narration streams here while the pipeline runs")
        self.log.anchorClicked.connect(self._on_tail_link)
        outer.addWidget(self.log, 1)

        # config card — prompt · workdir · controls · pipeline bar
        config = QWidget()
        config.setObjectName("hartConfigPanel")
        self._config_panel = config                # strong ref — GC-safe across theme swaps
        lay = QVBoxLayout(config)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(QLabel("<b>Prompt</b> — the goal for the pipeline"))
        self.goal_edit = QTextEdit()
        self.goal_edit.setPlaceholderText(
            "e.g. Build a single-file HTML page with a heading, a working "
            "button, and an animated canvas.")
        self.goal_edit.setMinimumHeight(72)
        self.goal_edit.setMaximumHeight(200)   # internal scrollbar for long prompts
        lay.addWidget(self.goal_edit)

        wd_row = QHBoxLayout()
        self.workdir_edit = QLineEdit(self.workdir)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_workdir)
        wd_row.addWidget(QLabel("Workdir:"))
        wd_row.addWidget(self.workdir_edit, 1)
        wd_row.addWidget(browse)
        lay.addLayout(wd_row)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run pipeline")
        self.run_btn.setProperty("primary", True)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setProperty("danger", True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.pause_btn = QPushButton("⏸  Pause at next stage")
        self.pause_btn.setEnabled(False)      # only meaningful mid-run
        self.pause_btn.setToolTip(
            "Pause strictly BETWEEN stages — never interrupts the running "
            "stage. Click while a run is on: the review dialog opens at the "
            "next stage boundary (and every boundary after, until the run "
            "ends). Review and edit the stage's output, optionally redirect "
            "the next stage, then Continue.")
        self.pause_btn.clicked.connect(self._on_arm_pause)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addWidget(self.pause_btn)
        run_row.addStretch(1)
        lay.addLayout(run_row)

        self.pipeline_bar = PipelineBar()
        lay.addWidget(self.pipeline_bar)

        sc = QScrollArea()
        sc.setWidget(config)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.NoFrame)
        sc.setMinimumHeight(210)
        outer.addWidget(sc)
        outer.setStretch(0, 1)            # tail dominates
        outer.setStretch(1, 0)            # config compact at the bottom
        return w

    def _populate_model_combos(self):
        self._model_lookup = {}
        entries = []
        for b in self.backends.all():
            for m in b.models:
                key = f"{b.name}/{m}"
                self._model_lookup[key] = (b, m)
                entries.append(key)
        if not entries:
            entries = ["(add a backend and Discover models)"]
        for combo in [self.primary_combo] + list(self.stage_combos.values()):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(entries)
            if current in entries:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self._on_same_model_toggled(self.same_model_cb.isChecked())

    def _resolve_model(self, combo):
        key = combo.currentText()
        if key == best_of_n.BUILTIN:
            return best_of_n.BUILTIN    # sentinel — worker runs the built-in judge
        if key in self._model_lookup:
            b, m = self._model_lookup[key]
            return best_of_n.make_model(b, m)
        return None

    def _on_same_model_toggled(self, on):
        for c in self.stage_combos.values():
            c.setEnabled(not on)

    # ── stage-gate review dialog ─────────────────────────────────────────
    def _on_stage_gate(self, stage, output):
        """Modal review between stages: shows the stage's exact output as
        editable JSON. Continue passes the edit downstream; Abort kills the
        run. The worker thread is blocked on a threading.Event meanwhile."""
        import json as _json
        dlg = QDialog(self)
        dlg.setWindowTitle(f"⏸ {stage} — review before continuing")
        dlg.resize(780, 560)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            f"<b>{stage}</b> produced the output below — this is the exact "
            f"input the next stage will receive."))
        edit = QTextEdit()
        try:
            edit.setPlainText(_json.dumps(output, indent=2, ensure_ascii=False))
        except (TypeError, ValueError):
            edit.setPlainText(str(output))
        edit.setFont(QFont("Menlo", 11))
        lay.addWidget(edit, 1)
        note = QTextEdit()
        note.setMaximumHeight(80)
        note.setPlaceholderText(
            "Optional prompt note redirecting the next stage — e.g. 'skip "
            "the scoreboard, focus on game-over logic' …")
        lay.addWidget(QLabel("Prompt note for the next stage (optional):"))
        lay.addWidget(note)
        btns = QHBoxLayout()
        cont = QPushButton("▶  Continue")
        cont.setProperty("primary", True)
        abort = QPushButton("■  Abort run")
        abort.setProperty("danger", True)
        btns.addStretch(1)
        btns.addWidget(abort)
        btns.addWidget(cont)
        lay.addLayout(btns)
        state = {"ok": False}

        def _cont():
            state["ok"] = True
            txt = edit.toPlainText().strip()
            edited = output
            if txt:
                try:
                    edited = _json.loads(txt)
                except ValueError:
                    self._log(stage, "edited output was not valid JSON — "
                                     "keeping the original")
            self._worker.deliver_gate(edited,
                                      note.toPlainText().strip() or None)
            dlg.accept()

        def _abort():
            providers.ABORT["flag"] = True
            self._worker.deliver_gate(output, None)
            dlg.reject()

        cont.clicked.connect(_cont)
        abort.clicked.connect(_abort)
        dlg.exec()

    # ── backend actions ──────────────────────────────────────────────────
    def _refresh_backend_list(self):
        current = self._selected_backend_name()
        self.backend_list.blockSignals(True)
        self.backend_list.clear()
        for b in self.backends.all():
            models = ", ".join(b.models) if b.models else "(not discovered)"
            keyhint = " 🔑" if (b.api_key or b.api_key_env) else ""
            item = QListWidgetItem(f"{b.name}{keyhint}  —  {b.base_url}\n{models}")
            item.setData(Qt.UserRole, b.name)
            self.backend_list.addItem(item)
        if self.backend_list.count():
            if current:
                self._select_backend(current)
            else:
                self.backend_list.setCurrentRow(0)
        self.backend_list.blockSignals(False)

    def _selected_backend_name(self):
        item = self.backend_list.currentItem()
        if item is not None:
            return item.data(Qt.UserRole)
        name = self.backend_name.text().strip()
        return name if name in self.backends.backends else None

    def _selected_backend(self):
        name = self._selected_backend_name()
        return self.backends.get(name) if name else None

    def _select_backend(self, name):
        for i in range(self.backend_list.count()):
            if self.backend_list.item(i).data(Qt.UserRole) == name:
                self.backend_list.setCurrentRow(i)
                return

    def _provider(self):
        return "anthropic" if self.backend_provider.currentIndex() == 1 else "openai"

    def _on_provider_changed(self, *_):
        if self.backend_provider.currentIndex() == 1:      # anthropic
            self.backend_url.setPlaceholderText("https://api.anthropic.com")
            self.backend_model.setPlaceholderText("model id (required for anthropic)")
            self.backend_key_env.setPlaceholderText("…or env var (e.g. ANTHROPIC_API_KEY)")
        else:                                              # openai-compatible
            self.backend_url.setPlaceholderText(
                "http://127.0.0.1:7001/v1  or  https://api.openai.com/v1")
            self.backend_model.setPlaceholderText("model id (optional)")
            self.backend_key_env.setPlaceholderText("…or env var (e.g. OPENAI_API_KEY)")

    def _on_backend_selected(self, current, _previous):
        """Load the selected backend into the form for editing (never echo the
        secret key — show a hint instead)."""
        if current is None:
            return
        b = self.backends.get(current.data(Qt.UserRole))
        if b is None:
            return
        self.backend_name.setText(b.name)
        self.backend_url.setText(b.base_url)
        self.backend_provider.blockSignals(True)
        self.backend_provider.setCurrentIndex(1 if b.provider == "anthropic" else 0)
        self.backend_provider.blockSignals(False)
        self._on_provider_changed()
        self.backend_model.setText(b.models[0] if len(b.models) == 1 else "")
        self.backend_key.clear()
        if b.api_key_env:
            self.backend_key_env.setText(b.api_key_env)
            self.backend_key.setPlaceholderText("(key comes from env var)")
        elif b.api_key:
            self.backend_key_env.clear()
            self.backend_key.setPlaceholderText("(existing key kept — type to replace)")
        else:
            self.backend_key_env.clear()
            self.backend_key.setPlaceholderText("API key (optional)")

    def _on_save_backend(self):
        name = self.backend_name.text().strip()
        url = self.backend_url.text().strip()
        if not name or not url:
            QMessageBox.warning(self, "hart", "Enter a name and a URL.")
            return
        provider = self._provider()
        model_id = self.backend_model.text().strip()
        key = self.backend_key.text().strip()
        key_env = self.backend_key_env.text().strip()
        existing = self.backends.get(name)
        # preserve an existing key if the user didn't type a new one
        if existing and not key and not key_env:
            key, key_env = existing.api_key, existing.api_key_env
        b = Backend(name, url, provider, key, key_env,
                    ctx_tokens=existing.ctx_tokens if existing else 262144,
                    models=existing.models if existing else [])
        if model_id and model_id not in b.models:
            b.models.append(model_id)
        self.backends.add(b)
        self.backend_key.clear()      # don't keep a typed secret sitting in the field
        self._refresh_backend_list()
        self._select_backend(name)
        self._populate_model_combos()
        self._log("backends", f"saved {name} [{provider}] {url}"
                  + (f"  (key via env {key_env})" if key_env
                     else ("  (key set)" if key else "")))

    def _on_remove_backend(self):
        b = self._selected_backend()
        if b is None:
            QMessageBox.warning(self, "hart", "Select a backend in the list first.")
            return
        self.backends.remove(b.name)
        self._refresh_backend_list()
        self._populate_model_combos()
        self._log("backends", f"removed {b.name}")

    def _on_discover(self):
        b = self._selected_backend()
        if b is None:
            QMessageBox.warning(self, "hart",
                                "Add a backend (name + URL), click it in the list, "
                                "then Discover.")
            return
        self._log("discovery", f"discovering models on {b.base_url}…")
        w = DiscoverWorker(b, self)
        w.done.connect(self._on_discovered)
        self._discover_workers.append(w)
        w.start()

    def _on_discovered(self, name, ok, result):
        b = self.backends.get(name)
        if ok:
            self._log("discovery", f"{name}: {len(result)} model(s) — "
                                   f"{', '.join(result[:8])}")
        else:
            self._log("discovery", f"{name}: discovery failed — {result}")
        self.backends.save()
        self._refresh_backend_list()
        self._populate_model_combos()

    # ── cache actions ────────────────────────────────────────────────────
    def _on_browse_workdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose workdir", self.workdir)
        if d:
            self.workdir_edit.setText(d)

    # ── run ──────────────────────────────────────────────────────────────
    def _on_run(self):
        goal = self.goal_edit.toPlainText().strip()
        if not goal:
            QMessageBox.warning(self, "hart", "Enter a goal first.")
            return
        self.workdir = self.workdir_edit.text().strip() or self.workdir
        os.makedirs(self.workdir, exist_ok=True)

        # resolve planning models
        primary = self._resolve_model(self.primary_combo)
        if primary is None:
            QMessageBox.warning(self, "hart",
                                "No model selected — add a backend and Discover.")
            return
        # resolve stage models
        if self.same_model_cb.isChecked():
            stage_models = {s: primary for s in STAGES}
        else:
            stage_models = {}
            for s in STAGES:
                m = self._resolve_model(self.stage_combos[s]) or primary
                stage_models[s] = m

        self._log("run", f"goal: {goal[:80]}")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pipeline_bar.reset()

        self._worker = PipelineWorker(
            goal, self.workdir, stage_models, parallel=4,
            test_command=None, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage_gate.connect(self._on_stage_gate)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _apply_header_palette(self):
        """Re-style the header for the current palette (dark/light) so the
        logo and text stay visible after a theme toggle."""
        p = theme.current()
        self.header_stage.setStyleSheet(
            f"color: {p['fg']}; font-weight: 700; background: {p['panel2']};"
            f"border: 1px solid {p['border']}; border-radius: 9px;"
            f"padding: 3px 12px;")
        self.theme_btn.setText("☀ light" if theme.current() is theme.DARK
                               else "☾ dark")
        if hasattr(self, "_header_widget"):
            self._header_widget.setStyleSheet(
                f"background: {p['panel']};"
                f"border-bottom: 1px solid {p['border']};")

    def _on_toggle_theme(self):
        from PySide6.QtWidgets import QApplication
        nxt = theme.LIGHT if theme.current() is theme.DARK else theme.DARK
        theme.apply(QApplication.instance(), nxt)
        self.theme_btn.setText("☾ dark" if nxt is theme.LIGHT else "☀ light")
        self._apply_header_palette()
        self.pipeline_bar.refresh_palette()
        for name, state in self._last_states.items():
            self.pipeline_bar.set_stage(name, state)
        self._log("theme", "switched to "
                  + ("light" if nxt is theme.LIGHT else "dark"))

    def _on_tail_link(self, url):
        """Clickable artifact paths in the tail: directories reveal in
        Finder, files open in the default app."""
        import subprocess
        p = url.toLocalFile() if hasattr(url, "toLocalFile") else url.toString()
        p = p.replace("file://", "")
        if os.path.isdir(p):
            subprocess.run(["open", "-R", p], check=False)
        elif os.path.isfile(p):
            subprocess.run(["open", p], check=False)
        else:
            self._log("run", f"link target not found: {p}")

    def _on_stop(self):
        providers.ABORT["flag"] = True
        self._log("run", "stop requested — aborting in-flight calls…")

    # ── worker signal handlers ───────────────────────────────────────────
    def _on_arm_pause(self):
        if self._worker is not None:
            self._worker.arm_gate()
            self._log("run", "\u23f8 pause armed \u2014 opens at the next "
                             "stage boundary")
    def _on_progress(self, stage, msg):
        self._log(stage, msg)
        self._set_stage(stage, "active")

    def _on_finished(self, result):
        self.pipeline_bar.set_all("done")
        self._set_status(f"done — {result.get('status')}")
        self._log("done", f"status={result.get('status')}  "
                          f"artifacts={result.get('artifacts')}  "
                          f"calls={ (result.get('metrics') or {}).get('calls') }")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        providers.ABORT["flag"] = False

    def _on_stopped(self):
        self._set_status("stopped")
        self._log("run", "stopped by user")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        providers.ABORT["flag"] = False

    def _on_failed(self, err):
        for name, chip in self.pipeline_bar.chips.items():
            if chip._state == "active":
                self.pipeline_bar.set_stage(name, "fail")
        self._set_status("failed")
        self._log("error", err)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        providers.ABORT["flag"] = False

    # ── helpers ──────────────────────────────────────────────────────────
    def _format_plan(self, p):
        lines = [f"model: {p.model_name}   valid: {p.valid}   "
                 f"calls: {p.calls}   wall: {p.wall}s", ""]
        for s in p.plan.get("steps", []):
            lines.append(f"  {s.get('id')}. {s.get('title')}")
            if s.get("detail"):
                lines.append(f"      {s.get('detail')}")
            if s.get("outputs"):
                lines.append(f"      outputs: {', '.join(s['outputs'])}")
            if s.get("acceptance"):
                lines.append("      acceptance:")
                for a in s["acceptance"]:
                    lines.append(f"        - {a}")
            lines.append("")
        if p.self_check.get("problems"):
            lines.append("self-check problems: " + "; ".join(p.self_check["problems"]))
        if p.error:
            lines.append(f"ERROR: {p.error}")
        return "\n".join(lines)

    def _log(self, stage, msg):
        import html as _h
        import time as _t
        color = theme.STAGE_COLORS.get(stage, theme.MUTED)
        ts = _t.strftime("%H:%M:%S")
        safe = _h.escape(str(msg))
        self.log.append(
            f'<span style="color:{theme.MUTED}">{ts}</span> '
            f'<span style="color:{color};font-weight:700">[{_h.escape(stage)}]</span> '
            f'<span style="color:{theme.FG}">{safe}</span>')
        if self.autoscroll_cb.isChecked():
            sb = self.log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _set_status(self, text):
        color = theme.GREEN
        for word, c in (("fail", theme.RED), ("stop", theme.AMBER),
                        ("running", theme.ACCENT), ("planning", theme.ACCENT),
                        ("discover", theme.ACCENT)):
            if word in text:
                color = c
                break
        self.header_status.setText(f"● {text}")
        self.header_status.setStyleSheet(
            f"color: {color}; font-weight: 700; background: transparent;")
        self.statusBar().showMessage(text)

    def _set_stage(self, stage, state):
        self.pipeline_bar.set_stage(stage, state)

    def _set_all_stages(self, state):
        self.pipeline_bar.set_all(state)
