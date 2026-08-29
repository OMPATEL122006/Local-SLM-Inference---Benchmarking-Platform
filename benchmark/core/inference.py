import threading
import time
from typing import Any, Dict

import ollama

from benchmark.system_monitor import (
    collect_snapshot,
    summarize_snapshots,
)


RESOURCE_SAMPLE_INTERVAL = 0.5


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


def run_single_inference(
    model: str,
    prompt: str,
) -> Dict[str, Any]:
    """
    Run one Ollama inference directly in the current process
    while collecting system resource metrics.
    """

    client = ollama.Client()

    baseline = collect_snapshot()

    monitor = ResourceMonitor()
    monitor.start()

    start = time.perf_counter()

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            think=False,
        )
    finally:
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

    return {
        "response": getattr(response, "response", ""),
        "latency_seconds": round(latency_seconds, 4),
        "total_duration_ns": getattr(response, "total_duration", None),
        "load_duration_ns": getattr(response, "load_duration", None),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
        "prompt_eval_duration_ns": getattr(
            response,
            "prompt_eval_duration",
            None,
        ),
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration,
        "tokens_per_second": round(tokens_per_second, 4),
        "resource_metrics": resource_metrics,
    }