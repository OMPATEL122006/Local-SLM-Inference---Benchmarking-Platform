import json
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
DEFAULT_CHECKPOINT_FILE = RESULTS_DIR / "modular_results.json"


class CheckpointManager:
    """
    Manages atomic persistence of benchmark run results immediately after
    each inference, and handles resuming by checking existing completed runs.
    """

    def __init__(self, checkpoint_file: Path = DEFAULT_CHECKPOINT_FILE):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._results: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.checkpoint_file.exists():
            return []
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return []

    def get_results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    def get_completed_run_ids(self) -> Set[str]:
        return {
            res["run_id"]
            for res in self._results
            if "run_id" in res and res.get("status") == "success"
        }

    def is_completed(self, run_id: str) -> bool:
        return run_id in self.get_completed_run_ids()

    def record_and_save(self, result: Dict[str, Any]) -> None:
        """
        Appends or updates the result in memory and immediately persists it
        atomically using a temporary file replacement.
        """
        existing_index = None
        for idx, res in enumerate(self._results):
            if res.get("run_id") == result.get("run_id"):
                existing_index = idx
                break

        if existing_index is not None:
            self._results[existing_index] = result
        else:
            self._results.append(result)

        self._atomic_save()

    def _atomic_save(self) -> None:
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(self._results, file, indent=2, ensure_ascii=False)
        temp_file.replace(self.checkpoint_file)
