"""The GUI pipeline: best-of-N plan → build → validate → evaluate → summary.

Unlike the CLI's `Run` (which does single-model planning + critic), the GUI
pipeline:
  * takes an already-chosen plan (from the GUI's planning stage model),
  * runs each downstream stage on its own configured model (same or different).

It reuses the hart agents directly (in-process), so the GUI *is* the harness.
"""

import os

from hart.agents import AGENTS
from hart.agents.base import RunContext
from hart.blackboard import Blackboard


class GuiPipeline:
    def __init__(self, goal, workdir, stage_models, chosen_plan,
                 on_event=None,
                 parallel=4, sandbox=False, shell_timeout=600, test_command=None,
                 gate=None):
        """
        stage_models:    {stage: Model} incl. planning/building/validation/…
        chosen_plan:     the plan dict to carry into building
        """
        self.goal = goal
        self.workdir = workdir
        self.stage_models = stage_models
        self.chosen_plan = chosen_plan
        self.on_event = on_event or (lambda *a, **k: None)
        self.parallel = parallel
        self.sandbox = sandbox
        self.shell_timeout = shell_timeout
        self.test_command = test_command
        # gate(stage_name, output_dict) -> (edited_output, prompt_note|None)
        # called BETWEEN stages when pausing is on; blocks the worker thread
        self.gate = gate

        # build a registry from the per-stage models
        models = {}
        stage_map = {}
        for stage, m in stage_models.items():
            models[m.name] = m
            stage_map[stage] = m.name
        from hart.config import ModelRegistry
        default = next(iter(stage_models.values()), None)
        self.registry = ModelRegistry(models, stage_map,
                                      default=default.name if default else None)
        extra = {"test_command": test_command, "quiet": True}
        self.bb = Blackboard(workdir, Blackboard.new_run_id(), goal,
                             self.registry, extra)
        self.ctx = RunContext(self.bb, self.registry, goal, extra,
                              parallel=parallel, sandbox=sandbox,
                              shell_timeout=shell_timeout)
        self.agents = {name: AGENTS[name](self.ctx) for name in AGENTS}

    # ── entry ────────────────────────────────────────────────────────────
    def run(self):
        self.on_event("run", f"workdir: {os.path.abspath(self.workdir)} "
                             f"(artifacts land here; stage I/O in .hart/stages/)")
        self._write_chosen_plan()
        if self.gate:
            self.chosen_plan = self._gate("planning", self.chosen_plan)
            self.bb.write_stage("planning", self.chosen_plan)
        self._build()
        self._validate()
        self._evaluation_loop()
        self._summary()
        self.bb.save()
        return self.bb.state.get("result")

    def _plan_model_name(self):
        return (self.chosen_plan or {}).get("_selected_by", "gui")

    def _write_chosen_plan(self):
        plan = dict(self.chosen_plan or {"steps": []})
        self.bb.write_stage("planning", plan)
        self.bb.state["plan"] = plan
        self.bb.set_stage("planning", "done", steps=len(plan.get("steps", [])))
        self.bb.save()
        self.on_event("planning", f"using selected plan "
                                  f"({len(plan.get('steps', []))} steps)")

    # ── stages ───────────────────────────────────────────────────────────
    def _build(self):
        self.on_event("building",
                      f"building with {self.stage_models['building'].name}…")
        self.agents["building"].run()
        build = self.bb.read_stage("building", {}) or {}
        arts = build.get("artifacts") or []
        if arts:
            for a in arts:
                p = a if os.path.isabs(a) else os.path.join(self.workdir, a)
                self.on_event("building", f"wrote {p}")
        else:
            self.on_event("building", "no artifacts declared — check .hart/stages/building.json")
        if self.gate:
            build = self._gate("building", build)

    def _validate(self):
        self.on_event("validation", "validating…")
        self.agents["validation"].run()
        val = self.bb.read_stage("validation", {}) or {}
    def _evaluation_loop(self):
        max_fix = int(self.ctx.budget("eval_fix_rounds", 3))
        rnd = 0
        while True:
            self.on_event("evaluation", "evaluating…")
            self.agents["evaluation"].run()
            ev = self.bb.read_stage("evaluation", {}) or {}
            if ev.get("verdict") == "pass":
                self.bb.set_stage("evaluation", "done", verdict="pass")
                break
            if rnd >= max_fix:
                self.bb.set_stage("evaluation", "done", verdict="fail", fix_round=rnd)
                break
            rnd += 1
            self.bb.state["pipeline"]["evaluation"]["fix_round"] = rnd
            self.bb.save()
            self.on_event("evaluation", f"failed — fix round {rnd}/{max_fix}")
            fix_mode = {"source": "evaluation",
                        "failures": ev.get("failures", []),
                        "fix_actions": ev.get("fix_actions", [])}
            self.agents["building"].run(fix_mode=fix_mode)
            self.agents["validation"].run()

    def _summary(self):
        # gate before the final summary (evaluation output reviewable)
        if self.gate:
            ev = self.bb.read_stage("evaluation", {}) or {}
            self._gate("evaluation", ev)
        self.on_event("summary", "summarizing…")
        self.agents["summary"].run()
        res = self.bb.state.get("result") or {}
        for a in res.get("artifacts") or []:
            p = a if os.path.isabs(a) else os.path.join(self.workdir, a)
            self.on_event("summary", f"deliverable: {p}")

    def _gate(self, stage, output):
        self.on_event(stage, "⏸ paused for review — edit the output in the "
                             "dialog, then Continue")
        edited, note = (self.gate or (lambda s, o: (o, None)))(stage, output)
        if edited is not output and isinstance(edited, dict):
            self.bb.write_stage(stage, edited)
        if note:
            self.ctx.extra["prompt_notes"] = \
                (self.ctx.extra.get("prompt_notes") or []) + \
                [{f"stage": stage, "note": note}]
            self.on_event(stage, f"prompt note attached for downstream: {note[:80]}")
        self.on_event(stage, "gate passed — continuing")
        return edited if isinstance(edited, dict) else output

    # ── helpers ──────────────────────────────────────────────────────────
