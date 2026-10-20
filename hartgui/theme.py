"""Global dark theme for the hart GUI (opencode-inspired).

One QSS stylesheet + a small palette module. Applied once in main.py via
apply(app). Stage/pipeline colors live here so every widget agrees.
"""

BG = "#0d1117"          # window
PANEL = "#161b22"       # cards / groups
PANEL2 = "#1c2330"      # inputs, hover
BORDER = "#2d333b"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
AMBER = "#d29922"
RED = "#f85149"

STAGE_COLORS = {
    "connection": ACCENT,
    "planning": "#db61a2",
    "building": ACCENT,
    "validation": AMBER,
    "evaluation": "#a371f7",
    "summary": GREEN,
    "run": FG,
    "done": GREEN,
    "error": RED,
    "backends": MUTED,
    "discovery": MUTED,
    "cache": MUTED,
}

QSS = f"""
* {{ font-family: -apple-system, 'Helvetica Neue', sans-serif; font-size: 13px; color: {FG}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QLabel {{ color: {FG}; background: transparent; }}
QLabel[muted="true"] {{ color: {MUTED}; font-size: 11px; }}
QLabel[h1="true"] {{ font-size: 20px; font-weight: 800; }}

QGroupBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    margin-top: 14px; padding: 10px 10px 8px 10px; font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: {ACCENT}; background: {PANEL}; border-radius: 4px;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {{
    background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 7px;
    padding: 5px 8px; selection-background-color: #264f78;
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {PANEL2}; border: 1px solid {BORDER};
    selection-background-color: #264f78; outline: none;
}}

QPushButton {{
    background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 7px;
    padding: 6px 14px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: #22293a; }}
QPushButton:disabled {{ color: #555e6b; border-color: {BORDER}; }}
QPushButton[primary="true"] {{
    background: #238636; border: 1px solid #2ea043; color: #fff; font-weight: 700;
    padding: 8px 22px; font-size: 14px;
}}
QPushButton[primary="true"]:hover {{ background: #2ea043; }}
QPushButton[primary="true"]:disabled {{ background: #1b3a25; color: #4a6b55; border-color: #274534; }}
QPushButton[danger="true"] {{
    background: #da3633; border: 1px solid #f8514988; color: #fff; font-weight: 700; padding: 8px 18px;
}}
QPushButton[danger="true"]:hover {{ background: #f85149; }}
QPushButton[danger="true"]:disabled {{ background: #3a1d1d; color: #6b4a4a; }}

QListWidget {{
    background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 7px;
    padding: 4px; outline: none;
}}
QListWidget::item {{ padding: 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: #264f78; color: #fff; }}

QCheckBox {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER}; border-radius: 4px;
    background: {PANEL2};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QRadioButton {{ spacing: 7px; background: transparent; }}
QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER}; border-radius: 8px;
    background: {PANEL2};
}}
QRadioButton::indicator:checked {{ border: 4px solid {ACCENT}; background: {BG}; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {PANEL}; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 6px 14px; margin-right: 2px;
    border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{ color: {FG}; background: {PANEL}; border-color: {BORDER}; }}
QTabBar::tab:hover {{ color: {FG}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #444c56; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; }}

QSplitter::handle {{ background: {BORDER}; width: 1px; }}
QToolTip {{ background: {PANEL2}; color: {FG}; border: 1px solid {BORDER}; padding: 4px; }}

QListWidget#nav {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 6px; outline: none; font-size: 13.5px;
}}
QListWidget#nav::item {{
    padding: 10px 10px; border-radius: 9px; margin: 2px 3px; color: {MUTED};
}}
QListWidget#nav::item:selected {{
    background: #264f78; color: {FG};
}}
QListWidget#nav::item:hover {{ background: {PANEL2}; color: {FG}; }}

QDialog {{ background: {BG}; }}
"""


def _qss(p):
    """Render the stylesheet for a palette dict (see DARK/LIGHT)."""
    return f"""
* {{ font-family: -apple-system, 'Helvetica Neue', sans-serif; font-size: 13px; color: {p['fg']}; }}
QMainWindow, QDialog {{ background: {p['bg']}; }}
QWidget {{ background: transparent; }}
QLabel {{ color: {p['fg']}; background: transparent; }}
QLabel[muted="true"] {{ color: {p['muted']}; font-size: 11px; }}
QLabel[h1="true"] {{ font-size: 20px; font-weight: 800; }}

QGroupBox {{
    background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 10px;
    margin-top: 14px; padding: 10px 10px 8px 10px; font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: {p['accent']}; background: {p['panel']}; border-radius: 4px;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {{
    background: {p['panel2']}; border: 1px solid {p['border']}; border-radius: 7px;
    padding: 5px 8px; selection-background-color: {p['select']};
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{ border: 1px solid {p['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {p['panel2']}; border: 1px solid {p['border']};
    selection-background-color: {p['select']}; outline: none;
}}

QPushButton {{
    background: {p['panel2']}; border: 1px solid {p['border']}; border-radius: 7px;
    padding: 6px 14px; font-weight: 600; color: {p['fg']};
}}
QPushButton:hover {{ border-color: {p['accent']}; color: {p['accent']}; }}
QPushButton:pressed {{ background: {p['pressed']}; }}
QPushButton:disabled {{ color: {p['disabled_fg']}; border-color: {p['border']}; }}
QPushButton[primary="true"] {{
    background: {p['primary_bg']}; border: 1px solid {p['primary_bd']}; color: #fff; font-weight: 700;
    padding: 8px 22px; font-size: 14px;
}}
QPushButton[primary="true"]:hover {{ background: {p['primary_hover']}; }}
QPushButton[primary="true"]:disabled {{ background: {p['primary_dis']}; color: {p['primary_dis_fg']}; border-color: {p['primary_dis_bd']}; }}
QPushButton[danger="true"] {{
    background: {p['danger_bg']}; border: 1px solid {p['danger_bd']}; color: #fff; font-weight: 700; padding: 8px 18px;
}}
QPushButton[danger="true"]:hover {{ background: {p['danger_hover']}; }}
QPushButton[danger="true"]:disabled {{ background: {p['danger_dis']}; color: {p['muted']}; }}

QListWidget {{
    background: {p['panel2']}; border: 1px solid {p['border']}; border-radius: 7px;
    padding: 4px; outline: none;
}}
QListWidget::item {{ padding: 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {p['select']}; color: {p['fg']}; }}

QCheckBox {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {p['border']}; border-radius: 4px;
    background: {p['panel2']};
}}
QCheckBox::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}
QRadioButton {{ spacing: 7px; background: transparent; }}
QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {p['border']}; border-radius: 8px;
    background: {p['panel2']};
}}
QRadioButton::indicator:checked {{ border: 4px solid {p['accent']}; background: {p['bg']}; }}

QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 8px; background: {p['panel']}; }}
QTabBar::tab {{
    background: transparent; color: {p['muted']}; padding: 6px 14px; margin-right: 2px;
    border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{ color: {p['fg']}; background: {p['panel']}; border-color: {p['border']}; }}
QTabBar::tab:hover {{ color: {p['fg']}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p['scroll_hover']}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p['border']}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {p['panel']}; border-top: 1px solid {p['border']}; color: {p['muted']}; }}
QStatusBar QLabel {{ color: {p['muted']}; }}

QSplitter::handle {{ background: {p['border']}; width: 1px; }}
QToolTip {{ background: {p['panel2']}; color: {p['fg']}; border: 1px solid {p['border']}; padding: 4px; }}

QListWidget#nav {{
    background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 12px;
    padding: 6px; outline: none; font-size: 13.5px;
}}
QListWidget#nav::item {{
    padding: 10px 10px; border-radius: 9px; margin: 2px 3px; color: {p['muted']};
}}
QListWidget#nav::item:selected {{ background: {p['select']}; color: {p['fg']}; }}
QListWidget#nav::item:hover {{ background: {p['panel2']}; color: {p['fg']}; }}
"""


DARK = dict(bg=BG, panel=PANEL, panel2=PANEL2, border=BORDER, fg=FG,
            muted=MUTED, accent=ACCENT, select="#264f78", pressed="#22293a",
            disabled_fg="#555e6b", primary_bg="#238636", primary_bd="#2ea043",
            primary_hover="#2ea043", primary_dis="#1b3a25",
            primary_dis_fg="#4a6b55", primary_dis_bd="#274534",
            danger_bg="#da3633", danger_bd="#f8514988", danger_hover="#f85149",
            danger_dis="#3a1d1d", scroll_hover="#444c56",
            green=GREEN, amber=AMBER, red=RED)

# warm paper tones: soft on the eyes, no harsh pure-white/pure-black
LIGHT = dict(bg="#faf7f2", panel="#fffdf8", panel2="#f3efe6", border="#e2dccd",
             fg="#3b3a36", muted="#8a857a", accent="#8a7a4a",
             select="#e8e0cc", pressed="#ece7d9", disabled_fg="#c0b9ab",
             primary_bg="#7a9a6d", primary_bd="#8aa87d",
             primary_hover="#8aa87d", primary_dis="#e9e4d6",
             primary_dis_fg="#b3ac9c", primary_dis_bd="#e0dac9",
             danger_bg="#c96f5e", danger_bd="#c96f5e88", danger_hover="#b25a49",
             danger_dis="#f0e4e0", scroll_hover="#d5cec0",
             green="#7a9a6d", amber="#b89a5a", red="#c96f5e")

# light-theme stage colors: muted, easy-on-the-eyes versions
LIGHT_STAGES = dict(STAGE_COLORS)
LIGHT_STAGES.update({"connection": "#6b8aad", "planning": "#a586b8",
                     "building": "#7a9a6d", "validation": "#b89a5a",
                     "evaluation": "#9a7ab8", "summary": "#7a9a6d",
                     "error": "#c96f5e", "run": "#3b3a36"})

_current = {"palette": DARK}


def current():
    return _current["palette"]


def stages():
    return LIGHT_STAGES if current() is LIGHT else STAGE_COLORS


def apply(app, palette=None):
    app.setStyle("Fusion")
    p = palette or DARK
    _current["palette"] = p
    app.setStyleSheet(_qss(p))
