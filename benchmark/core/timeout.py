import multiprocessing
import threading
import time
from typing import Any, Dict

import ollama

from benchmark.system_monitor import (
    collect_snapshot,
    summarize_snapshots,
)

DEFAULT_TIMEOUT_SECONDS = 180.0
RESOURCE_SAMPLE_INTERVAL = 0.5


class InferenceTimeoutError(Exception):
    """Raised when an inference call exceeds the allotted timeout."""

    pass


class ResourceMonitor:
    def __init__(self, interval=RESOURCE_SAMPLE_INTERVAL):
        self.interval = interval
        self.snapshots = []
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            self.snapshots.append(collect_snapshot())
            time.sleep(self.interval)

    def start(self):
        self.snapshots = []
        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join()
            self.thread = None

        return summarize_snapshots(self.snapshots)


def _inference_worker(model: str, prompt: str, queue: multiprocessing.Queue):
    """Child process worker target for executing an Ollama generate request."""
    try:
        client = ollama.Client()
        baseline = collect_snapshot()

        monitor = ResourceMonitor()
        monitor.start()

        start = time.perf_counter()

        response = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            think=False,
        )

        end = time.perf_counter()
        resource_metrics = monitor.stop()

        latency_seconds = end - start
        resource_metrics["baseline"] = baseline

        eval_duration = getattr(response, "eval_duration", None) or 0
        eval_count = getattr(response, "eval_count", None) or 0

        tokens_per_second = (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_duration > 0
            else 0.0
        )

        metrics = {
            "response": getattr(response, "response", ""),
            "latency_seconds": round(latency_seconds, 4),
            "total_duration_ns": getattr(response, "total_duration", None),
            "load_duration_ns": getattr(response, "load_duration", None),
            "prompt_eval_count": getattr(response, "prompt_eval_count", None),
            "prompt_eval_duration_ns": getattr(
                response, "prompt_eval_duration", None
            ),
            "eval_count": eval_count,
            "eval_duration_ns": eval_duration,
            "tokens_per_second": round(tokens_per_second, 4),
            "resource_metrics": resource_metrics,
        }

        queue.put({"success": True, "metrics": metrics})

    except Exception as exc:
        queue.put({"success": False, "error": str(exc)})


def run_inference_with_timeout(
    model: str,
    prompt: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Executes an inference request in an isolated child process protected by a timeout.
    If the timeout expires, the process is forcibly killed to ensure no orphaned
    requests or sockets remain hanging.
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_inference_worker,
        args=(model, prompt, queue),
    )

    start_time = time.perf_counter()
    process.start()

    process.join(timeout=timeout_seconds)
    elapsed_time = round(time.perf_counter() - start_time, 4)

    if process.is_alive():
        # Process hung or exceeded timeout -> kill process cleanly
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()

        raise InferenceTimeoutError(
            f"Inference call timed out after {timeout_seconds} seconds (elapsed: {elapsed_time}s)"
        )

    try:
        result = queue.get(timeout=1.0)
        if result["success"]:
            result["metrics"]["elapsed_seconds"] = elapsed_time
            return result["metrics"]
        else:
            raise RuntimeError(result.get("error", "Unknown inference error"))
    except Exception as exc:
        raise RuntimeError(
            f"Inference process exited without returning a result (exitcode={process.exitcode}): {exc}"
        )
