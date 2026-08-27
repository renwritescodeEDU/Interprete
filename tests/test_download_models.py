import unittest
from unittest.mock import patch

from src.download_models import download_models

class TestDownloadModels(unittest.TestCase):
    def test_download_models_callable(self):
        """Verify download_models is callable"""
        self.assertTrue(callable(download_models))

    @patch('src.download_models.subprocess.run')
    def test_download_models_success(self, mock_run):
        """Mock subprocess.run, verify called with correct args"""
        download_models()
        mock_run.assert_called_once_with(["ollama", "pull", "qwen2.5:1.5b"], check=True)

    @patch('src.download_models.sys.exit')
    @patch('src.download_models.subprocess.run')
    def test_download_models_failure(self, mock_run, mock_exit):
        """Mock subprocess.run to raise, verify sys.exit called"""
        mock_run.side_effect = Exception("Mock failure")
        download_models()
        mock_exit.assert_called_once_with(1)
