import unittest
import multiprocessing
from unittest.mock import patch, MagicMock

from src.main import Orchestrator, main

class TestMain(unittest.TestCase):
    def test_main_callable(self):
        """Verify main is callable"""
        self.assertTrue(callable(main))

    @patch('src.main.run_ui')
    @patch('src.main.multiprocessing.set_start_method')
    @patch('src.main.multiprocessing.get_start_method')
    def test_main_execution(self, mock_get_start, mock_set_start, mock_run_ui):
        """Mock run_ui and set_start_method, verify run_ui called"""
        mock_get_start.return_value = 'fork'
        main()
        mock_set_start.assert_called_once_with('spawn', force=True)
        mock_run_ui.assert_called_once()
        args, _ = mock_run_ui.call_args
        self.assertEqual(len(args), 5)

    @patch('src.main.multiprocessing.Process')
    @patch('src.main.os.makedirs')
    def test_orchestrator_start_stop(self, mock_makedirs, mock_process):
        """Mock Process, verify 3 processes started and stopped with logs"""
        mock_p1 = MagicMock()
        mock_p2 = MagicMock()
        mock_p3 = MagicMock()
        mock_process.side_effect = [mock_p1, mock_p2, mock_p3]
        
        orch = Orchestrator()
        orch.start_processes()
        
        self.assertEqual(mock_process.call_count, 3)
        mock_p1.start.assert_called_once()
        mock_p2.start.assert_called_once()
        mock_p3.start.assert_called_once()
        
        mock_p1.is_alive.return_value = False
        mock_p2.is_alive.return_value = False
        mock_p3.is_alive.return_value = False
        
        orch.stop_processes()
        # They were all assigned, and _terminate_process will be called on each
        mock_p1.is_alive.assert_called()

    @patch('src.main.os.makedirs')
    def test_orchestrator_clear_queue(self, mock_makedirs):
        """Put items in queue, call _clear_queue, verify empty"""
        import time
        orch = Orchestrator()
        q = multiprocessing.Queue()
        q.put("item1")
        q.put("item2")
        time.sleep(0.1) # Wait for multiprocessing queue to propagate
        orch._clear_queue(q)
        time.sleep(0.1)
        self.assertTrue(q.empty())

    @patch('src.main.os.makedirs')
    def test_orchestrator_terminate_process_alive(self, mock_makedirs):
        """Mock process that is_alive=True, won't join, verify terminate called"""
        orch = Orchestrator()
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        
        orch._terminate_process(mock_process, "test_proc")
        
        mock_process.join.assert_any_call(timeout=3.0)
        mock_process.terminate.assert_called_once()
        mock_process.join.assert_any_call(timeout=1.0)

    @patch('src.main.os.makedirs')
    def test_orchestrator_terminate_process_dead(self, mock_makedirs):
        """Mock process that is_alive=False, verify terminate NOT called"""
        orch = Orchestrator()
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        
        orch._terminate_process(mock_process, "test_proc")
        
        mock_process.terminate.assert_not_called()

    @patch('src.main.os.makedirs')
    def test_orchestrator_poison_pills(self, mock_makedirs):
        """Call stop_processes, verify poison pills sent to queues"""
        orch = Orchestrator()
        orch.control_queue = MagicMock()
        orch.asr_queue = MagicMock()
        orch.translation_queue = MagicMock()
        
        orch.audio_process = None
        orch.asr_process = None
        orch.translator_process = None
        
        orch.stop_processes()
        
        orch.control_queue.put_nowait.assert_called_once_with("QUIT")
        orch.asr_queue.put_nowait.assert_called_once_with(None)
        orch.translation_queue.put_nowait.assert_called_once_with(None)

    @patch('src.main.os.makedirs')
    def test_orchestrator_creates_log_dir(self, mock_makedirs):
        """Verify os.makedirs called for logs dir"""
        orch = Orchestrator()
        mock_makedirs.assert_called_once_with("logs", exist_ok=True)
