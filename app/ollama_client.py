import ollama

from app.config import OLLAMA_HOST


client = ollama.Client(host=OLLAMA_HOST)


def list_models() -> list[str]:
    response = client.list()

    return [
        model.model
        for model in response.models
    ]


def generate_response(
    prompt: str,
    model: str,
) -> dict:
    response = client.generate(
        model=model,
        prompt=prompt,
    )

    eval_duration = response["eval_duration"]

    if eval_duration > 0:
        tokens_per_second = (
            response["eval_count"]
            / (eval_duration / 1_000_000_000)
        )
    else:
        tokens_per_second = 0.0

    return {
        "response": response["response"],
        "model": model,
        "metrics": {
            "total_duration_ns": response["total_duration"],
            "load_duration_ns": response["load_duration"],
            "prompt_eval_count": response["prompt_eval_count"],
            "prompt_eval_duration_ns": response["prompt_eval_duration"],
            "eval_count": response["eval_count"],
            "eval_duration_ns": response["eval_duration"],
            "tokens_per_second": round(tokens_per_second, 2),
        },
    }