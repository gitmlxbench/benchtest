"""Qt worker threads — run the (blocking) agent pipeline off the GUI thread.

The agents make blocking HTTP calls to the models, so they must never run on
the Qt main thread. `PipelineWorker` runs the whole flow (parallel planning →
(single-model planning → build → validate → evaluate → summary) and emits
signals the main window connects to for live updates.
"""

import threading

from PySide6.QtCore import QThread, Signal

from hart.providers import RunAborted
from hartgui import best_of_n
from hartgui.pipeline import GuiPipeline


class PipelineWorker(QThread):
    stage_gate = Signal(str, object)               # (stage, output) — pause for review
    progress = Signal(str, str)                    # (stage, message)
    finished_ok = Signal(object)                   # result dict
    stopped = Signal()                             # user-initiated stop
    failed = Signal(str)                           # error message

    def __init__(self, goal, workdir, stage_models,
                 parallel=4, sandbox=False,
                 test_command=None, pause_gates=False, parent=None):
        super().__init__(parent)
        self.pause_gates = pause_gates
        self._armed = False            # set by arm_gate() mid-run (GUI thread)
        self._gate_reply = {}          # {"edited":…, "note":…} filled by GUI
        self._gate_evt = threading.Event()
        self.goal = goal
        self.workdir = workdir
        self.stage_models = stage_models
        self.parallel = parallel
        self.sandbox = sandbox
        self.test_command = test_command

    def run(self):
        try:
            all_models = [m for m in self.stage_models.values()
                          if m is not None]
            if any(getattr(m, "provider", "") == "openai" for m in all_models):
                best_of_n.ensure_single_model(all_models,
                                              on_event=self.progress.emit)
            planner = self.stage_models.get("planning")
            if planner is None:
                self.failed.emit("no planning model selected — pick one on "
                                 "the Setup page")
                return
            self.progress.emit("planning", f"planning with {planner.name}…")
            chosen = best_of_n.plan_with_model(
                self.goal, self.workdir, planner,
                on_event=self.progress.emit)
            if not chosen.valid:
                probs = "; ".join(chosen.self_check.get("problems") or [])
                self.failed.emit("planning produced no valid plan: "
                                 + (chosen.error or probs or "invalid plan"))
                return
            self.progress.emit("planning", f"plan ready from {planner.name} "
                                           f"({chosen.n_steps} steps)")

            def _gate(stage, output):
                """STRICTLY between stages: no-op unless pausing was armed
                (the Pause-at-next-stage button). Blocks the worker thread
                while the GUI shows the review dialog."""
                if not (self.pause_gates or self._armed):
                    return (output, None)
                self._gate_evt.clear()
                self._gate_reply.clear()
                self.stage_gate.emit(stage, output)
                self._gate_evt.wait()
                return (self._gate_reply.get("edited", output),
                        self._gate_reply.get("note"))

            result = GuiPipeline(
                self.goal, self.workdir, self.stage_models, chosen.plan,
                on_event=self.progress.emit, parallel=self.parallel,
                sandbox=self.sandbox, test_command=self.test_command,
                gate=_gate).run()
            self.finished_ok.emit(result)
        except RunAborted:
            self.stopped.emit()
        except Exception as e:      # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


    def arm_gate(self):
        """Called by the GUI when the user clicks Pause-at-next-stage while a
        run is on: the NEXT stage boundary (and each one after, until the
        run ends) opens the review dialog. Never interrupts mid-stage."""
        self._armed = True
        self.progress.emit("run", "⏸ pause armed — the review dialog will "
                                   "open at the NEXT stage boundary")

    def deliver_gate(self, edited, note):
        """Called (queued, thread-safe via signal) by the GUI when the user
        closes the review dialog."""
        self._gate_reply = {"edited": edited, "note": note}
        self._gate_evt.set()


class DiscoverWorker(QThread):
    """Probe a backend's /models endpoint off the GUI thread."""
    done = Signal(str, bool, object)   # (backend_name, ok, models|error)

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend

    def run(self):
        ok, result = self.backend.discover()
        self.done.emit(self.backend.name, ok, result)
