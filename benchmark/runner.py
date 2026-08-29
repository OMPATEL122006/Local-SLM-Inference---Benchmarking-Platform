from benchmark.runners.base_runner import run_model_benchmark

MODELS = [
    "phi3:mini",
    "qwen3:4b",
    "llama3.2:3b",
]


def main():
    print("=" * 70)
    print("LOCAL SLM BENCHMARK ORCHESTRATOR")
    print("=" * 70)

    for model in MODELS:
        run_model_benchmark(model)

    print("\n" + "=" * 70)
    print("ALL MODEL BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
