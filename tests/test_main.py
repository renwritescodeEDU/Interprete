from unittest.mock import patch, MagicMock
from src.main import main, Orchestrator


def test_main_callable():
    assert callable(main)


@patch("src.main.run_ui")
@patch("src.main.multiprocessing.set_start_method")
def test_main_execution(mock_set_start, mock_run_ui):
    main()
    mock_run_ui.assert_called_once()


@patch("src.main.multiprocessing.Process")
def test_orchestrator_start_stop(mock_process_cls, caplog):
    mock_process = MagicMock()
    mock_process_cls.return_value = mock_process
    
    import logging
    caplog.set_level(logging.INFO)
    
    orchestrator = Orchestrator()
    orchestrator.start_processes()
    
    assert "Starting background processes..." in caplog.text
    
    assert mock_process_cls.call_count == 3
    assert mock_process.start.call_count == 3
    
    orchestrator.stop_processes()
    assert "Background processes stopped." in caplog.text

