import multiprocessing
from unittest.mock import patch, MagicMock
from src.ui import run_ui, MainWindow

def test_ui_callable():
    assert callable(run_ui)

@patch("src.ui.list_audio_devices", return_value=[])
@patch("src.ui.QApplication")
@patch("src.ui.MainWindow")
@patch("src.ui.sys")
def test_run_ui(mock_sys, mock_window_cls, mock_app_cls, mock_list_devices):
    """Verify run_ui initializes QApplication and MainWindow properly."""
    ui_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()
    mock_app = MagicMock()
    mock_app_cls.return_value = mock_app
    
    mock_window = MagicMock()
    mock_window_cls.return_value = mock_window
    
    start_cb = MagicMock()
    stop_cb = MagicMock()
    run_ui(ui_queue, control_queue, start_cb, stop_cb)
    
    mock_app_cls.assert_called_once()
    start_cb.assert_called_once()
    mock_window_cls.assert_called_once_with(ui_queue, control_queue, stop_cb, None)
    mock_window.show.assert_called_once()
    mock_app.exec.assert_called_once()
    mock_sys.exit.assert_called_once()

def test_ui_classes_exist():
    """Verify that MainWindow exists and has required methods."""
    assert hasattr(MainWindow, "on_action_clicked")
    assert hasattr(MainWindow, "poll_queue")
    assert hasattr(MainWindow, "update_system_readiness")
    assert hasattr(MainWindow, "closeEvent")
    # New device-related methods
    assert hasattr(MainWindow, "_refresh_devices")
    assert hasattr(MainWindow, "_on_device_changed")
    assert hasattr(MainWindow, "_restore_saved_device")
    assert hasattr(MainWindow, "_re_enable_device_controls")

