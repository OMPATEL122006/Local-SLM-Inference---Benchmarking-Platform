import ollama
import instructor
import time
from instructor.core.hooks import Hooks

from app.config import OLLAMA_HOST


client = ollama.Client(host=OLLAMA_HOST)

instructor_client = instructor.from_provider(
    "ollama/phi3:mini",
    base_url=f"{OLLAMA_HOST}/v1",
    mode=instructor.Mode.JSON,
)

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

def generate_structured_response(
    prompt: str,
    model: str,
    schema: dict,
) -> dict:
    response = client.generate(
        model=model,
        prompt=prompt,
        format=schema,
    )

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
            "tokens_per_second": round(
                response["eval_count"]
                / (response["eval_duration"] / 1_000_000_000),
                2,
            )
            if response["eval_duration"] > 0
            else 0.0,
        },
    }
    
def generate_instructor_response(
    prompt: str,
    model: str,
    response_model,
):
    attempts = 0
    validation_errors = []

    hooks = Hooks()

    def on_completion_kwargs(*args, **kwargs):
        nonlocal attempts
        attempts += 1

    def on_parse_error(error, **kwargs):
        validation_errors.append(str(error))

    hooks.on("completion:kwargs", on_completion_kwargs)
    hooks.on("parse:error", on_parse_error)

    client = instructor.from_provider(
        f"ollama/{model}",
        base_url=f"{OLLAMA_HOST}/v1",
        mode=instructor.Mode.JSON,
    )

    start_time = time.perf_counter()

    result = client.create(
        response_model=response_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_retries=2,
        hooks=hooks,
    )

    end_time = time.perf_counter()

    return {
        "result": result,
        "model": model,
        "latency_seconds": round(
            end_time - start_time,
            4,
        ),
        "attempts": attempts,
        "retries": max(0, attempts - 1),
        "validation_errors": validation_errors,
    }