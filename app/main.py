"""Aplicação FastAPI: rotas de saúde, catálogo de modelos e classificação."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import (
    Body,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app import metrics
from app.config import ALLOWED_CONTENT_TYPES, HTTP_422, UPLOAD_CHUNK_BYTES, settings
from app.contorno_api import router as contorno_router
from app.fields_api import router as fields_router
from app.framing import framing_model
from app.inference.preprocess import InvalidImageError
from app.model import CornDiseaseModel, ModelNotLoadedError, UnknownClassError, Upload
from app.registry import MODEL_SPECS, Platform, default_for_crop, registry, specs_by_crop
from app.review import (
    TRAINING_CONSENT_HEADER,
    InvalidClassNameError,
    ReviewQueue,
    UnknownReviewItemError,
    consent_allows,
)
from app.rover_api import router as rover_router
from app.schemas import (
    BatchPredictionResponse,
    CropChoices,
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    ModelChoice,
    ModelInfo,
    PredictionResponse,
    ReviewItem,
    ReviewStats,
    ReviewVerdict,
)
from app.service_auth import ServiceTokenMiddleware
from app.severity import severity_model
from app.terrain_api import router as terrain_router

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
TrainingConsentHeader = Header(
    default=None,
    alias=TRAINING_CONSENT_HEADER,
    description=(
        "1 = o cliente autorizou o uso da imagem no treino; 0 = não. Com 0 a imagem "
        "não vai para a fila de revisão. Ausente = decisão de quem opera (REVIEW_DIR)."
    ),
)
TtaQuery = Query(
    default=None,
    description=(
        "Prevê em quatro orientações e tira a média — é como as acurácias do "
        "MODELS.md foram medidas. Custa 4x. Vazio segue INFERENCE_TTA."
    ),
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
    framing_model.load()
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

# Lido a cada requisição (e não fixado aqui) para os testes poderem trocar o token.
app.add_middleware(ServiceTokenMiddleware, token_getter=lambda: settings.service_token)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Por fora de tudo: mede a requisição inteira, inclusive a recusa do token.
app.add_middleware(metrics.MetricasMiddleware)


app.include_router(terrain_router)
app.include_router(contorno_router)
app.include_router(rover_router)
app.include_router(fields_router)


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


@app.exception_handler(RequestValidationError)
async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 legível mesmo quando o valor recusado não cabe em JSON.

    O corpo de erro do FastAPI devolve o valor que o cliente mandou. Quando esse
    valor é ``NaN`` ou infinito — que os modelos deste projeto recusam de
    propósito, porque toda comparação com NaN é falsa —, serializar a resposta
    estoura e o 422 vira 500. Aqui o valor ofensivo vira texto antes de sair.
    """
    return JSONResponse(
        status_code=HTTP_422,
        content={"detail": jsonable_encoder(exc.errors(), custom_encoder={float: _finito})},
    )


def _finito(value: float) -> float | str:
    return value if math.isfinite(value) else repr(value)


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


@app.get(
    "/metrics",
    summary="Métricas da IA (formato Prometheus)",
    response_class=Response,
    responses={200: {"content": {metrics.CONTENT_TYPE: {}}}},
)
def prometheus_metrics() -> Response:
    """Inferências, erros, latência e classes por modelo, e o HTTP por rota.

    Exige o ``SERVICE_TOKEN`` quando ele está definido (o middleware do token vale
    aqui como em toda rota): o Prometheus manda ``Authorization: Bearer``.
    """
    carregados = {key: model.is_loaded for key, model in registry.models.items()}
    return Response(
        metrics.render([metrics.modelos_carregados(carregados)]),
        media_type=metrics.CONTENT_TYPE,
    )


@app.get("/models", response_model=list[ModelInfo], summary="Catálogo de modelos")
def models() -> list[ModelInfo]:
    """Lista todos os modelos do catálogo, quais estão prontos e com que salvaguardas.

    ``safeguards`` é o que teria mostrado, no primeiro dia, que a novidade do
    ``leaf-foliar`` estava desligada: a referência era recusada no carregamento e
    o único rastro era uma linha de log.
    """
    infos = []
    for spec in MODEL_SPECS.values():
        model = registry.get(spec.key)
        infos.append(
            ModelInfo(
                key=spec.key,
                title=spec.title,
                platform=spec.platform.value,
                summary=spec.summary,
                classes=list(spec.class_names),
                img_size=spec.img_size,
                datasets=list(spec.datasets),
                loaded=model.is_loaded,
                needs_field_data=spec.needs_field_data,
                runtime=model.runtime,
                safeguards=model.safeguards if model.is_loaded else {},
                display_name=spec.label,
                crop=spec.crop,
                variant=spec.variant,
                license=spec.license,
                field_accuracy=spec.field_accuracy,
                test_accuracy=spec.test_accuracy,
                recommended=spec.recommended,
            )
        )
    return infos


@app.get(
    "/models/choices",
    response_model=list[CropChoices],
    summary="Versões de modelo por cultura, para escolher qual usar",
)
def model_choices() -> list[CropChoices]:
    """As versões de cada cultura, como um seletor de modelos.

    Para cada cultura: as versões disponíveis (a recomendada primeiro, depois
    pelo número de campo), com a licença dos dados de treino, e a que o rover
    usa quando manda só ``crop=``. Quem escolhe passa a ``key`` em ``model=`` —
    no ``/predict`` de um upload manual ou no ``/predict/rover-frame``.
    """
    result = []
    for crop, options in sorted(specs_by_crop().items()):
        result.append(
            CropChoices(
                crop=crop,
                rover_default=default_for_crop(crop).key,
                options=[
                    ModelChoice(
                        key=spec.key,
                        display_name=spec.label,
                        variant=spec.variant,
                        summary=spec.summary,
                        license=spec.license,
                        field_accuracy=spec.field_accuracy,
                        test_accuracy=spec.test_accuracy,
                        recommended=spec.recommended,
                        loaded=registry.get(spec.key).is_loaded,
                    )
                    for spec in options
                ],
            )
        )
    return result


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
    tta: bool | None = TtaQuery,
    training_consent: str | None = TrainingConsentHeader,
) -> PredictionResponse:
    """Classifica uma única imagem e devolve as probabilidades por classe."""
    selected = _resolve_ready_model(model, platform)
    data = await _read_validated_upload(file)
    filename = file.filename or "upload"

    # A inferência é síncrona e pesada: rodá-la no event loop travaria todas as
    # outras requisições enquanto o TensorFlow trabalha.
    prediction = await run_in_threadpool(selected.predict, filename, data, tta)

    if prediction.needs_review and review_queue.enabled and consent_allows(training_consent):
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
    tta: bool | None = TtaQuery,
    training_consent: str | None = TrainingConsentHeader,
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
    results = await run_in_threadpool(selected.predict_many, uploads, tta)

    if review_queue.enabled and consent_allows(training_consent):
        await run_in_threadpool(_enqueue_flagged, uploads, results, files)

    counts = Counter(result.predicted_class for result in results)
    return BatchPredictionResponse(
        model_key=selected.key,
        results=results,
        summary={name: counts.get(name, 0) for name in selected.class_names},
        abstained_count=sum(1 for result in results if result.abstained),
        needs_review_count=sum(1 for result in results if result.needs_review),
        quality_issue_count=sum(
            1 for result in results if result.quality is not None and not result.quality.ok
        ),
        out_of_scope_count=sum(
            1 for result in results if result.framing is not None and not result.framing.in_scope
        ),
        ambiguous_count=sum(
            1 for result in results if result.prediction_set and len(result.prediction_set) > 1
        ),
        empty_set_count=sum(1 for result in results if result.prediction_set == []),
    )


@app.post(
    "/predict/explain",
    response_model=ExplanationResponse,
    responses={
        **ERROR_RESPONSES,
        422: {"model": ErrorResponse, "description": "Classe inexistente no modelo."},
    },
    summary="Mostra onde o modelo olhou para decidir",
)
async def explain(
    file: UploadFile = File(description="Imagem JPEG, PNG ou WebP."),
    model: str | None = ModelQuery,
    platform: Platform | None = PlatformQuery,
    target_class: str | None = Query(
        default=None,
        description=(
            "Classe a explicar. Vazio explica a prevista; informar outra responde "
            "'por que não esta?'."
        ),
    ),
    grid: int = Query(
        default=7,
        ge=3,
        le=14,
        description="Células por lado. Custa grid² + 1 passadas pelo modelo.",
    ),
    overlay: bool = Query(default=True, description="Inclui a foto com o mapa em PNG base64."),
    method: Literal["occlusion", "rise"] = Query(
        default="occlusion",
        description=(
            "occlusion: uma célula por vez, grid² + 1 passadas. rise: metade da imagem "
            "coberta ao acaso, samples + 1 passadas — enxerga lesões espalhadas, que a "
            "oclusão não vê."
        ),
    ),
    samples: int = Query(default=256, ge=32, le=1024, description="Máscaras do rise."),
) -> ExplanationResponse:
    """Cobre a imagem célula por célula e mede quanto a confiança cai em cada uma.

    É ferramenta de console e de auditoria, não de triagem de voo: custa dezenas
    de passadas por imagem. Nada aqui entra na fila de revisão — quem pede uma
    explicação já está olhando para a imagem.
    """
    selected = _resolve_ready_model(model, platform)
    data = await _read_validated_upload(file)
    try:
        return await run_in_threadpool(
            selected.explain,
            file.filename or "upload",
            data,
            target_class,
            grid,
            overlay,
            method,
            samples,
        )
    except UnknownClassError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc


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
    order: Literal["recentes", "valor"] = Query(
        default="recentes",
        description=(
            "'recentes': do mais novo ao mais antigo. 'valor': o que mais ensina ao próximo "
            "treino primeiro (aprendizado ativo), com o motivo de cada item."
        ),
    ),
) -> list[ReviewItem]:
    """Lista o que a detecção de novidade separou.

    Cada item traz o palpite do modelo e a pontuação de novidade: o agrônomo
    decide melhor sabendo o que a máquina respondeu e o quanto ela estranhou.
    Com ``order=valor``, a fila vem ordenada pelo que mais ensina — incerteza,
    novidade, classe rara entre as já rotuladas —, e o agrônomo que só tem tempo
    para 50 fotos rotula as 50 que mais valem.
    """
    _require_review_queue()
    if order == "valor":
        ranked = review_queue.prioritized(budget=offset + limit)[offset : offset + limit]
        return [
            ReviewItem(**item, priority=choice.position, priority_reasons=list(choice.reasons))
            for item, choice in ranked
        ]
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
        raise HTTPException(HTTP_422, str(exc)) from exc
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
