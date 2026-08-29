import json
from pathlib import Path
import tempfile
import unittest

from benchmark.core.checkpoint import CheckpointManager
from benchmark.core.retry import run_id_builder
from benchmark.core.timeout import InferenceTimeoutError


class TestBenchmarkSmoke(unittest.TestCase):
    def test_run_id_builder(self):
        self.assertEqual(
            run_id_builder("phi3:mini", "extraction_01", "cold"),
            "phi3:mini_extraction_01_cold",
        )
        self.assertEqual(
            run_id_builder("phi3:mini", "extraction_01", "warm", 2),
            "phi3:mini_extraction_01_warm_02",
        )

    def test_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_results.json"
            mgr = CheckpointManager(ckpt_path)

            self.assertEqual(mgr.get_completed_run_ids(), set())

            sample_run = {
                "run_id": "phi3:mini_generation_01_cold",
                "model": "phi3:mini",
                "task_id": "generation_01",
                "category": "general_generation",
                "run_type": "cold",
                "run_number": None,
                "attempts": 1,
                "status": "success",
                "error": None,
                "metrics": {"latency_seconds": 1.23},
            }

            mgr.record_and_save(sample_run)

            self.assertTrue(ckpt_path.exists())
            self.assertTrue(mgr.is_completed("phi3:mini_generation_01_cold"))

            # Re-instantiate checkpoint manager to test reading persisted checkpoint
            mgr2 = CheckpointManager(ckpt_path)
            self.assertTrue(mgr2.is_completed("phi3:mini_generation_01_cold"))
            self.assertEqual(len(mgr2.get_results()), 1)

    def test_checkpoint_status_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "status_test_results.json"
            mgr = CheckpointManager(ckpt_path)

            success_run = {
                "run_id": "phi3:mini_task1_cold",
                "status": "success",
            }
            timeout_run = {
                "run_id": "phi3:mini_task2_cold",
                "status": "timeout",
            }
            error_run = {
                "run_id": "phi3:mini_task3_cold",
                "status": "error",
            }

            mgr.record_and_save(success_run)
            mgr.record_and_save(timeout_run)
            mgr.record_and_save(error_run)

            completed_ids = mgr.get_completed_run_ids()

            # Success run MUST be considered completed (skipped on restart)
            self.assertIn("phi3:mini_task1_cold", completed_ids)
            self.assertTrue(mgr.is_completed("phi3:mini_task1_cold"))

            # Timeout and error runs MUST NOT be considered completed (retried on restart)
            self.assertNotIn("phi3:mini_task2_cold", completed_ids)
            self.assertFalse(mgr.is_completed("phi3:mini_task2_cold"))

            self.assertNotIn("phi3:mini_task3_cold", completed_ids)
            self.assertFalse(mgr.is_completed("phi3:mini_task3_cold"))

            # All 3 records should still be preserved in stored results
            self.assertEqual(len(mgr.get_results()), 3)


if __name__ == "__main__":
    unittest.main()
