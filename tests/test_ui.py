import multiprocessing
from unittest.mock import patch, MagicMock
from src.ui import run_ui, ControlPanelWindow, OverlayWindow

def test_ui_callable():
    assert callable(run_ui)

@patch("src.ui.QApplication")
@patch("src.ui.ControlPanelWindow")
@patch("src.ui.sys")
def test_run_ui(mock_sys, mock_window_cls, mock_app_cls):
    """Verify run_ui initializes QApplication and ControlPanelWindow properly."""
    ui_queue = multiprocessing.Queue()
    mock_app = MagicMock()
    mock_app_cls.return_value = mock_app
    
    mock_window = MagicMock()
    mock_window_cls.return_value = mock_window
    
    start_cb = MagicMock()
    stop_cb = MagicMock()
    run_ui(ui_queue, start_cb, stop_cb)
    
    mock_app_cls.assert_called_once()
    mock_window_cls.assert_called_once_with(ui_queue, start_cb, stop_cb)
    mock_window.show.assert_called_once()
    mock_app.exec.assert_called_once()
    mock_sys.exit.assert_called_once()

def test_ui_classes_exist():
    """Verify that ControlPanelWindow and OverlayWindow exist and have required methods."""
    assert hasattr(ControlPanelWindow, "start_interpreter")
    assert hasattr(ControlPanelWindow, "stop_interpreter")
    assert hasattr(OverlayWindow, "poll_queue")
