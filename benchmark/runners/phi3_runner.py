from benchmark.runners.base_runner import run_model_benchmark

MODEL_NAME = "phi3:mini"


def main():
    run_model_benchmark(MODEL_NAME)


if __name__ == "__main__":
    main()
