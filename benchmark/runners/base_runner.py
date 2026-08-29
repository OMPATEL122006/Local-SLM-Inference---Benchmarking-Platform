import json
from pathlib import Path
import subprocess
import time
from typing import Optional

from benchmark.core.checkpoint import CheckpointManager
from benchmark.core.retry import execute_run_with_retry, run_id_builder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_FILE = PROJECT_ROOT / "benchmark" / "benchmark_tasks.json"
WARM_RUNS = 2


def load_tasks():
    with open(TASKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)["tasks"]


def unload_model(model: str):
    """Unload a model when starting/finishing a model group."""
    try:
        subprocess.run(
            ["ollama", "stop", model],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        print(f"Warning: timed out while stopping {model}.")
    except Exception as exc:
        print(f"Warning: could not stop {model}: {exc}")

    time.sleep(1)


def model_is_loaded(model: str) -> bool:
    """Check whether the target model is currently reported by ollama ps."""
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

        lines = result.stdout.splitlines()

        return any(
            line.strip().startswith(model)
            for line in lines[1:]
        )

    except Exception:
        return False


def run_model_benchmark(
    model: str,
    checkpoint_file: Optional[Path] = None,
    warm_runs: int = WARM_RUNS,
    timeout_seconds: float = 180.0,
    max_attempts: int = 3,
):
    """
    Shared model runner orchestrating the benchmark lifecycle for a single model:
    - Unload model once before starting
    - First inference = COLD
    - All remaining inferences = WARM
    - Context remains independent per task
    - Unload model after finishing
    - Results persisted immediately after every inference (task-level checkpointing)
    """
    tasks = load_tasks()
    checkpoint_mgr = (
        CheckpointManager(checkpoint_file)
        if checkpoint_file
        else CheckpointManager()
    )

    print("=" * 70)
    print(f"STARTING BENCHMARK FOR MODEL: {model}")
    print("=" * 70)

    completed_ids = checkpoint_mgr.get_completed_run_ids()
    print(
        f"Checkpoint status: {len(completed_ids)} runs previously completed."
    )

    print("Preparing model for cold start: unloading model...")
    unload_model(model)

    if model_is_loaded(model):
        print(
            f"Warning: {model} still appears loaded before the first request."
        )
    else:
        print("Model unloaded successfully. Ready for cold start.")

    first_request = True

    for task in tasks:
        print(f"\n[{task['id']}] Category: {task['category']}")

        # First request for this model lifecycle is cold
        if first_request:
            cold_id = run_id_builder(model, task["id"], "cold")
            if not checkpoint_mgr.is_completed(cold_id):
                print("Executing COLD inference for this model...")
                result = execute_run_with_retry(
                    model=model,
                    task=task,
                    run_type="cold",
                    max_attempts=max_attempts,
                    timeout_seconds=timeout_seconds,
                )
                checkpoint_mgr.record_and_save(result)
                print(
                    f"Saved checkpoint: {cold_id} "
                    f"(status: {result['status']}, attempts: {result['attempts']})"
                )
            else:
                print(f"Cold inference already completed: {cold_id}")

            first_request = False
        else:
            print("Model remains loaded; task starts with fresh context.")

        # Warm inferences
        for run_number in range(1, warm_runs + 1):
            warm_id = run_id_builder(
                model, task["id"], "warm", run_number
            )
            if checkpoint_mgr.is_completed(warm_id):
                print(
                    f"Warm inference {run_number}/{warm_runs} "
                    f"already completed: {warm_id}"
                )
                continue

            print(
                f"Executing warm inference {run_number}/{warm_runs}..."
            )
            result = execute_run_with_retry(
                model=model,
                task=task,
                run_type="warm",
                run_number=run_number,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )
            checkpoint_mgr.record_and_save(result)
            print(
                f"Saved checkpoint: {warm_id} "
                f"(status: {result['status']}, attempts: {result['attempts']})"
            )

    print(f"\nFinished benchmark for model: {model}")
    print("Unloading model...")
    unload_model(model)
    if model_is_loaded(model):
        print(f"Warning: {model} still appears loaded after benchmark.")
    else:
        print("Model unloaded successfully.")
