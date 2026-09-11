"""Aplicação FastAPI: rotas de saúde, catálogo de modelos e classificação."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import ALLOWED_CONTENT_TYPES, UPLOAD_CHUNK_BYTES, settings
from app.inference.preprocess import InvalidImageError
from app.model import CornDiseaseModel, ModelNotLoadedError, Upload
from app.registry import MODEL_SPECS, Platform, registry
from app.review import InvalidClassNameError, ReviewQueue, UnknownReviewItemError
from app.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    ReviewItem,
    ReviewStats,
    ReviewVerdict,
)
from app.severity import severity_model

logger = logging.getLogger(__name__)

HTTP_413_TOO_LARGE = 413
"""Payload grande demais.

Usado como literal porque o Starlette renomeou a constante ao adotar os nomes da
RFC 9110, e o nome antigo passou a emitir DeprecationWarning.
"""

ERROR_RESPONSES: dict[int | str, dict[str, type[ErrorResponse]]] = {
    status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    HTTP_413_TOO_LARGE: {"model": ErrorResponse},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
}

review_queue = ReviewQueue(settings.review_dir)
"""Fila do agrônomo. Desligada quando ``REVIEW_DIR`` não está definido."""

ModelQuery = Query(
    default=None,
    description="Identificador do modelo, ex.: 'rover-understory'. Veja GET /models.",
)
PlatformQuery = Query(
    default=None,
    description="Alternativa ao 'model': escolhe o modelo padrão da plataforma.",
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Carrega os pesos de todos os modelos do catálogo, uma vez só."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    registry.load_all()
    severity_model.load()
    yield


app = FastAPI(
    title="Corn & Coffee Disease Detection API",
    description=(
        "Backend de inferência para detecção de doenças em milho e café. Serve "
        "um modelo por combinação de cultura e enquadramento: drone (triagem de "
        "dossel), rover terrestre (diagnóstico sob o dossel) e folha em close "
        "(milho e café). Ver MODELS.md."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(InvalidImageError)
async def _handle_invalid_image(_: Request, exc: InvalidImageError) -> JSONResponse:
    """Imagem ilegível é erro do cliente, não do servidor."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.exception_handler(ModelNotLoadedError)
async def _handle_model_not_loaded(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
    """Sem pesos carregados o serviço está indisponível, não quebrado."""
    logger.error("Previsão solicitada sem modelo carregado: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
    )


@app.get("/health", response_model=HealthResponse, summary="Estado da API")
def health() -> HealthResponse:
    """Responde ``200`` enquanto o processo estiver de pé.

    ``loaded_models`` é o que diz quais previsões estão disponíveis — útil para
    separar *liveness* de *readiness* num orquestrador.
    """
    return HealthResponse(
        status="ok",
        model_loaded=registry.get(settings.default_model).is_loaded,
        default_model=settings.default_model,
        loaded_models=registry.loaded_keys,
    )


@app.get("/models", response_model=list[ModelInfo], summary="Catálogo de modelos")
def models() -> list[ModelInfo]:
    """Lista todos os modelos do catálogo e quais estão prontos para prever."""
    return [
        ModelInfo(
            key=spec.key,
            title=spec.title,
            platform=spec.platform.value,
            summary=spec.summary,
            classes=list(spec.class_names),
            img_size=spec.img_size,
            datasets=list(spec.datasets),
            loaded=registry.get(spec.key).is_loaded,
            needs_field_data=spec.needs_field_data,
        )
        for spec in MODEL_SPECS.values()
    ]


@app.get(
    "/classes",
    response_model=list[str],
    responses=ERROR_RESPONSES,
    summary="Classes de um modelo",
)
def classes(
    model: str | None = ModelQuery,
    platform: Platform | None = PlatformQuery,
) -> list[str]:
    """Lista as classes na mesma ordem em que o modelo as indexa."""
    return _resolve_model(model, platform).class_names


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses=ERROR_RESPONSES,
    summary="Classifica uma imagem",
)
async def predict(
    file: UploadFile = File(description="Imagem JPEG, PNG ou WebP."),
    model: str | None = ModelQuery,
    platform: Platform | None = PlatformQuery,
) -> PredictionResponse:
    """Classifica uma única imagem e devolve as probabilidades por classe."""
    selected = _resolve_ready_model(model, platform)
    data = await _read_validated_upload(file)
    filename = file.filename or "upload"

    # A inferência é síncrona e pesada: rodá-la no event loop travaria todas as
    # outras requisições enquanto o TensorFlow trabalha.
    prediction = await run_in_threadpool(selected.predict, filename, data)

    if prediction.needs_review and review_queue.enabled:
        await run_in_threadpool(
            review_queue.enqueue, data, prediction, file.content_type or "image/jpeg"
        )
    return prediction


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses=ERROR_RESPONSES,
    summary="Classifica um lote de imagens",
)
async def predict_batch(
    files: list[UploadFile] = File(description="Imagens JPEG, PNG ou WebP."),
    model: str | None = ModelQuery,
    platform: Platform | None = PlatformQuery,
) -> BatchPredictionResponse:
    """Classifica várias imagens de uma vez — ex.: todas as fotos de um voo.

    O lote inteiro passa pelo modelo numa única chamada. A validação é
    tudo-ou-nada: se uma das imagens for ilegível, a requisição falha com ``400``
    identificando o arquivo, em vez de devolver um resultado parcial que o
    cliente teria de conferir item a item.
    """
    selected = _resolve_ready_model(model, platform)

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Envie ao menos uma imagem no campo 'files'.",
        )
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=(
                f"Lote de {len(files)} imagens excede o limite de "
                f"{settings.max_batch_size} por requisição."
            ),
        )

    uploads: list[Upload] = [
        (file.filename or "upload", await _read_validated_upload(file)) for file in files
    ]
    results = await run_in_threadpool(selected.predict_many, uploads)

    if review_queue.enabled:
        await run_in_threadpool(_enqueue_flagged, uploads, results, files)

    counts = Counter(result.predicted_class for result in results)
    return BatchPredictionResponse(
        model_key=selected.key,
        results=results,
        summary={name: counts.get(name, 0) for name in selected.class_names},
        abstained_count=sum(1 for result in results if result.abstained),
        needs_review_count=sum(1 for result in results if result.needs_review),
    )


@app.get(
    "/review/stats",
    response_model=ReviewStats,
    summary="Tamanho da fila de revisão",
    tags=["revisão"],
)
def review_stats() -> ReviewStats:
    """Quanto material o modelo separou por não saber explicar."""
    return ReviewStats(**review_queue.stats())


@app.get(
    "/review",
    response_model=list[ReviewItem],
    summary="Imagens aguardando diagnóstico",
    tags=["revisão"],
)
def review_pending(
    limit: int = Query(default=50, ge=1, le=500, description="Itens por página."),
    offset: int = Query(default=0, ge=0, description="Quantos pular, para paginar."),
) -> list[ReviewItem]:
    """Lista o que a detecção de novidade separou, do mais recente ao mais antigo.

    Cada item traz o palpite do modelo e a pontuação de novidade: o agrônomo
    decide melhor sabendo o que a máquina respondeu e o quanto ela estranhou.
    """
    _require_review_queue()
    return [ReviewItem(**item) for item in review_queue.pending(limit=limit, offset=offset)]


@app.get(
    "/review/{item_id}/image",
    summary="A imagem de um item da fila",
    tags=["revisão"],
    responses={404: {"model": ErrorResponse, "description": "Item não encontrado."}},
)
def review_image(item_id: str) -> FileResponse:
    """Devolve a foto para o agrônomo olhar antes de diagnosticar."""
    _require_review_queue()
    try:
        return FileResponse(review_queue.image_path(item_id))
    except UnknownReviewItemError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@app.post(
    "/review/{item_id}",
    response_model=ReviewItem,
    summary="Registrar o diagnóstico do agrônomo",
    tags=["revisão"],
    responses={
        404: {"model": ErrorResponse, "description": "Item não encontrado."},
        422: {"model": ErrorResponse, "description": "Rótulo inválido."},
    },
)
def review_label(item_id: str, verdict: ReviewVerdict = Body(...)) -> ReviewItem:
    """Aplica o rótulo e move a imagem para o dataset de revisados.

    O rótulo **não** precisa existir no catálogo: um nome novo cria uma classe
    nova, e é exatamente assim que uma doença que o modelo não conhecia entra no
    dataset próprio. A partir daqui o ``prepare_dataset`` consome o diretório
    como qualquer outro, e a imagem entra no próximo treino.
    """
    _require_review_queue()
    try:
        metadata = review_queue.label(item_id, verdict.class_name, verdict.reviewed_by)
    except UnknownReviewItemError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except InvalidClassNameError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return ReviewItem(**metadata)


def _require_review_queue() -> None:
    """A fila é opt-in: sem ``REVIEW_DIR`` não há o que listar nem onde gravar."""
    if not review_queue.enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Fila de revisão desligada. Defina REVIEW_DIR no .env para ativá-la.",
        )


def _enqueue_flagged(
    uploads: list[Upload], results: list[PredictionResponse], files: list[UploadFile]
) -> None:
    """Arquiva as imagens do lote que precisam de revisão."""
    for (_, data), result, file in zip(uploads, results, files, strict=True):
        if result.needs_review:
            review_queue.enqueue(data, result, file.content_type or "image/jpeg")


def _resolve_model(key: str | None, platform: Platform | None) -> CornDiseaseModel:
    """Escolhe o modelo pedido, traduzindo um identificador inválido em ``404``."""
    try:
        return registry.resolve(key, platform)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc.args[0])) from exc


def _resolve_ready_model(key: str | None, platform: Platform | None) -> CornDiseaseModel:
    """Idem, recusando cedo os modelos que ainda não têm pesos carregados."""
    selected = _resolve_model(key, platform)
    if not selected.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Modelo {selected.key!r} ainda não carregado no servidor. "
                f"Modelos prontos: {', '.join(registry.loaded_keys) or 'nenhum'}."
            ),
        )
    return selected


async def _read_validated_upload(file: UploadFile) -> bytes:
    """Valida o tipo declarado e lê o arquivo respeitando o limite de tamanho."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Tipo de arquivo não suportado: {file.content_type}. "
                f"Use um destes: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}."
            ),
        )
    return await _read_within_limit(file)


async def _read_within_limit(file: UploadFile) -> bytes:
    """Lê o upload em blocos, abortando assim que o limite é ultrapassado.

    Ler em blocos evita carregar um arquivo enorme inteiro na memória só para
    depois descobrir que ele excede ``MAX_UPLOAD_MB``.
    """
    limit = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0

    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=HTTP_413_TOO_LARGE,
                detail=(
                    f"{file.filename or 'upload'}: arquivo excede o limite de "
                    f"{settings.max_upload_mb} MB."
                ),
            )
        chunks.append(chunk)

    return b"".join(chunks)
