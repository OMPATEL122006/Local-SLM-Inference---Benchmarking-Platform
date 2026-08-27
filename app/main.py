from fastapi import FastAPI, HTTPException

from app.config import OLLAMA_HOST
from app.ollama_client import generate_response, list_models
from app.schemas import GenerateRequest, GenerateResponse


app = FastAPI(
    title="Local SLM Inference API",
    description="API for running and benchmarking local SLM inference",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Local SLM Inference API is running",
        "ollama_host": OLLAMA_HOST,
    }


@app.get("/models")
def models():
    return {
        "models": list_models()
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    available_models = list_models()

    if request.model not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' is not installed.",
        )

    return generate_response(
        prompt=request.prompt,
        model=request.model,
    )