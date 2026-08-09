from src.main import main


def test_main_stub():
    assert callable(main)


def test_main_execution(capsys):
    main()
    captured = capsys.readouterr()
    assert "Orchestrator ready." in captured.out

