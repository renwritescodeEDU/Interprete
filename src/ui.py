import multiprocessing
import sys
from PyQt6.QtWidgets import QApplication, QLabel, QWidget


def run_ui(ui_queue: multiprocessing.Queue):
    app = QApplication(sys.argv)
    window = QWidget()
    label = QLabel("Waiting...", parent=window)
    window.show()
    # In full implementation, a QTimer checks the ui_queue periodically
    # sys.exit(app.exec())
