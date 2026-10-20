"""Application entry point for the hart GUI.

    python3 -m hartgui
    python3 hartgui/main.py
"""

import sys

from PySide6.QtWidgets import QApplication

from . import theme
from .mainwindow import MainWindow


def main(argv=None):
    app = QApplication(argv if argv is not None else sys.argv)
    theme.apply(app)
    app.setApplicationName("hart")
    app.setOrganizationName("hart")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
