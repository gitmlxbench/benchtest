"""The pipeline stage bar: chips with live state, connected by a rail.

States: pending (gray dot), active (pulsing amber dot + elapsed timer),
done (green check), fail (red cross). Double-validation stages glow with a
second ring while a second model reviews the stage.
"""
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from hartgui import theme

STATES = ("connection", "planning", "building", "validation", "evaluation")

# one-line meaning, shown as the chip's tooltip
TOOLTIPS = {
    "connection": "probe the backend — confirm the model is reachable",
    "planning": "the planning model turns the goal into an ordered, "
                "acceptance-tested build plan",
    "building": "the building model executes the plan and writes the "
                "artifacts to the workdir",
    "validation": "the validation model reviews the artifacts against the "
                  "acceptance criteria",
    "evaluation": "the evaluation model scores the result and fixes any "
                  "test failure (repair loop)",
}


class _StageChip(QWidget):
    """One stage: a dot + name + (when active) elapsed time."""

    def __init__(self, name, color):
        super().__init__()
        self.name = name
        self.color = color
        self._state = "pending"
        self._t0 = None
        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._on_tick)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(1)
        self.dot = QLabel("●")
        self.dot.setAlignment(Qt.AlignCenter)
        self.dot.setFixedHeight(18)
        self.name_lbl = QLabel(name)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.time_lbl = QLabel("")
        self.time_lbl.setAlignment(Qt.AlignCenter)
        self.time_lbl.setProperty("muted", True)
        lay.addWidget(self.dot)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.time_lbl)
        self.setToolTip(TOOLTIPS.get(name, ""))
        self._render()

    # ── state machine ───────────────────────────────────────────────────
    def set_state(self, state):
        # the underlying C++ widget can outlive a Qt parent teardown during
        # window rebuilds — never touch it after deletion
        import shiboken6
        if not shiboken6.isValid(self):
            return
        self._state = state
        if state == "active":
            self._t0 = time.monotonic()
            self._tick.start()
        else:
            self._tick.stop()
            if state in ("done", "fail") and self._t0 is not None:
                self.time_lbl.setText(self._fmt(time.monotonic() - self._t0))
            elif state == "pending":
                self.time_lbl.setText("")
        self._render()

    def _on_tick(self):
        self.time_lbl.setText(self._fmt(time.monotonic() - (self._t0 or 0)))
        # pulse: alternate the dot between two shades while active
        self._pulse = not getattr(self, "_pulse", False)
        self._render()

    @staticmethod
    def _fmt(s):
        s = int(s)
        return f"{s // 60}:{s % 60:02d}"

    def _render(self):
        light = theme.current() is theme.LIGHT
        self._pulse = getattr(self, "_pulse", False)
        pending_dot = "#b9b2a2" if light else "#484f58"
        if self._state == "active":
            dot = (theme.AMBER if self._pulse else "#8a6d1f")
            glow = "#d2992233"
        elif self._state == "done":
            dot, glow = theme.GREEN, "#3fb95022"
        else:  # pending / fail
            dot, glow = pending_dot, "transparent"
        self.dot.setStyleSheet(f"color: {dot}; font-size: 15px; background: transparent;")
        self.name_lbl.setStyleSheet(
            f"color: {theme.FG if self._state != 'pending' else theme.MUTED};"
            f"font-weight: {'700' if self._state == 'active' else '400'};"
            f"font-size: 11.5px; background: transparent;")
        self.setStyleSheet(
            f"background: {glow if glow != 'transparent' else theme.PANEL};"
            f"border: 1px solid {theme.BORDER}; border-radius: 9px;")


FG = theme.FG


class PipelineBar(QWidget):
    """The full horizontal bar: chip → rail → chip …"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(4)
        self.chips = {}
        for i, name in enumerate(STATES):
            if i:
                rail = QLabel("──")
                rail.setStyleSheet(f"color: {theme.BORDER}; background: transparent;")
                rail.setFixedWidth(20)
                rail.setAlignment(Qt.AlignCenter)
                lay.addWidget(rail)
            chip = _StageChip(name, theme.STAGE_COLORS.get(name, theme.ACCENT))
            self.chips[name] = chip
            lay.addWidget(chip)
        lay.addStretch(1)

    def set_stage(self, name, state):
        if name in self.chips:
            self.chips[name].set_state(state)

    def set_all(self, state):
        for chip in self.chips.values():
            chip.set_state(state)

    def refresh_palette(self):
        """Re-render every chip against the current palette (theme toggle)."""
        for chip in self.chips.values():
            chip._render()

    def reset(self):
        self.set_all("pending")
