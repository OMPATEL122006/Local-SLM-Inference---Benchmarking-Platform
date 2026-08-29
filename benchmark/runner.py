import json
import time
from pathlib import Path

import ollama


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = PROJECT_ROOT / "benchmark" / "benchmark_tasks.json"

MODELS = [
    "phi3:mini",
    "qwen3:4b",
    "llama3.2:3b",
]

WARM_RUNS = 3


def load_tasks():
    with open(TASKS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["tasks"]


def run_inference(client, model, prompt):
    start = time.perf_counter()

    response = client.generate(
        model=model,
        prompt=prompt,
        stream=False,
        think=False,
    )

    end = time.perf_counter()

    latency_seconds = end - start

    return {
        "response": response.response,
        "latency_seconds": round(latency_seconds, 4),
        "total_duration_ns": response.total_duration,
        "load_duration_ns": response.load_duration,
        "prompt_eval_count": response.prompt_eval_count,
        "prompt_eval_duration_ns": response.prompt_eval_duration,
        "eval_count": response.eval_count,
        "eval_duration_ns": response.eval_duration,
        "tokens_per_second": (
            response.eval_count
            / (response.eval_duration / 1_000_000_000)
            if response.eval_duration
            else 0
        ),
    }


def main():
    tasks = load_tasks()
    client = ollama.Client()

    results = []

    print("=" * 70)
    print("LOCAL SLM BENCHMARK")
    print("=" * 70)

    for model in MODELS:
        print(f"\nMODEL: {model}")

        for task in tasks:
            print(
                f"\n[{task['id']}] "
                f"{task['category']}"
            )

            print("Running cold inference...")

            cold_result = run_inference(
                client,
                model,
                task["prompt"],
            )

            warm_results = []

            for run_number in range(1, WARM_RUNS + 1):
                print(
                    f"Running warm inference "
                    f"{run_number}/{WARM_RUNS}..."
                )

                result = run_inference(
                    client,
                    model,
                    task["prompt"],
                )

                warm_results.append(result)

            results.append(
                {
                    "model": model,
                    "task_id": task["id"],
                    "category": task["category"],
                    "cold": cold_result,
                    "warm": warm_results,
                }
            )

    output_dir = PROJECT_ROOT / "benchmark" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "raw_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print(f"Benchmark complete.")
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()