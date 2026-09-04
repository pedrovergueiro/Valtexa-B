from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    ALLOWED_CONTENT_TYPES,
    CLASS_NAMES,
    CORS_ORIGINS,
    MAX_BATCH_SIZE,
    MAX_UPLOAD_MB,
)
from app.inference.preprocess import InvalidImageError
from app.model import corn_disease_model
from app.schemas import BatchPredictionResponse, HealthResponse, PredictionResponse

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    corn_disease_model.load()
    yield


app = FastAPI(
    title="Corn Disease Detection API",
    description="Backend de inferência para detecção de doenças em folhas de milho, "
    "usado no pipeline de análise de imagens capturadas por drone.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_and_validate(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. "
            f"Use um destes: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo excede o limite de {MAX_UPLOAD_MB}MB.",
        )
    return data


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=corn_disease_model.is_loaded,
        classes=CLASS_NAMES,
    )


@app.get("/classes")
def classes() -> list[str]:
    return CLASS_NAMES


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not corn_disease_model.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado no servidor.")

    data = await _read_and_validate(file)
    try:
        return corn_disease_model.predict(file.filename or "upload", data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictionResponse:
    if not corn_disease_model.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado no servidor.")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Lote excede o limite de {MAX_BATCH_SIZE} imagens por requisição.",
        )

    results: list[PredictionResponse] = []
    for file in files:
        data = await _read_and_validate(file)
        try:
            results.append(corn_disease_model.predict(file.filename or "upload", data))
        except InvalidImageError as exc:
            raise HTTPException(
                status_code=400, detail=f"{file.filename}: {exc}"
            ) from exc

    summary = Counter(r.predicted_class for r in results)
    return BatchPredictionResponse(
        results=results,
        summary={name: summary.get(name, 0) for name in CLASS_NAMES},
    )
