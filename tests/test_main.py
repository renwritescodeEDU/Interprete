from unittest.mock import patch, MagicMock
from src.main import main


def test_main_callable():
    assert callable(main)


@patch("src.main.run_ui")
@patch("src.main.multiprocessing.Process")
@patch("src.main.multiprocessing.set_start_method")
def test_main_execution(mock_set_start, mock_process_cls, mock_run_ui, capsys):
    mock_process = MagicMock()
    mock_process_cls.return_value = mock_process

    main()
    
    captured = capsys.readouterr()
    assert "Starting background processes..." in captured.out
    
    assert mock_process_cls.call_count == 3
    assert mock_process.start.call_count == 3
    mock_run_ui.assert_called_once()

