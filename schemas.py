from pydantic import BaseModel, ConfigDict, Field


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    probabilities: list[ClassProbability]
    inference_ms: float


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    summary: dict[str, int] = Field(
        description="Count of predictions per class within this batch"
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    classes: list[str]
