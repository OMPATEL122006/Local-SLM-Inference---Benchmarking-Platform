import time
from typing import Any, Dict, List, Optional

from benchmark.core.inference import run_single_inference
from benchmark.core.timeout import DEFAULT_TIMEOUT_SECONDS

MAX_ATTEMPTS = 3
RETRY_DELAYS = [2, 5]  # Delay in seconds after attempt 1, attempt 2


def run_id_builder(
    model: str,
    task_id: str,
    run_type: str,
    run_number: Optional[int] = None,
) -> str:
    if run_type == "cold":
        return f"{model}_{task_id}_cold"
    return f"{model}_{task_id}_warm_{run_number:02d}"


def execute_run_with_retry(
    model: str,
    task: Dict[str, Any],
    run_type: str,
    run_number: Optional[int] = None,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Executes a benchmark run with bounded retries and per-inference timeout.
    """
    current_run_id = run_id_builder(model, task["id"], run_type, run_number)
    attempt_history: List[Dict[str, Any]] = []

    final_status = "error"
    final_error: Optional[str] = None
    final_metrics: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        start_attempt = time.perf_counter()
        try:
            metrics = run_single_inference(
                model=model,
                prompt=task["prompt"],
            )
            elapsed = round(time.perf_counter() - start_attempt, 4)
            attempt_history.append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "elapsed_seconds": elapsed,
                }
            )
            final_status = "success"
            final_metrics = metrics
            final_error = None
            break

        except Exception as exc:
            elapsed = round(time.perf_counter() - start_attempt, 4)
            err_msg = str(exc)
            attempt_history.append(
                {
                    "attempt": attempt,
                    "status": "error",
                    "error": err_msg,
                    "elapsed_seconds": elapsed,
                }
            )
            final_status = "error"
            final_error = err_msg

        # If we failed and have remaining attempts, wait according to backoff schedule
        if attempt < max_attempts:
            delay = (
                RETRY_DELAYS[attempt - 1]
                if (attempt - 1) < len(RETRY_DELAYS)
                else 5
            )
            time.sleep(delay)

    total_attempts = len(attempt_history)

    return {
        "run_id": current_run_id,
        "model": model,
        "task_id": task["id"],
        "category": task["category"],
        "run_type": run_type,
        "run_number": run_number,
        "attempts": total_attempts,
        "status": final_status,
        "error": final_error,
        "metrics": final_metrics,
        "attempt_history": attempt_history,
    }
