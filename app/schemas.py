from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Prompt to send to the local model",
    )
    model: str = Field(
        default="phi3:mini",
        description="Ollama model to use",
    )


class InferenceMetrics(BaseModel):
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    eval_count: int
    eval_duration_ns: int
    tokens_per_second: float


class GenerateResponse(BaseModel):
    response: str
    model: str
    metrics: InferenceMetrics
    
class StructuredAnalysis(BaseModel):
    summary: str
    key_points: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    
class StructuredResponse(BaseModel):
    result: StructuredAnalysis
    model: str
    metrics: InferenceMetrics