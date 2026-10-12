"""Rota do quadro do rover: o quadro inteiro entra, os achados por zona saem.

O rover não manda folha recortada: manda o quadro de 12 MP de uma câmera
lateral, com a altura dela no mastro e a distância até a planta (a mediana da
profundidade da OAK-D). Aqui o quadro vira recortes de tamanho físico fixo
(:mod:`app.rover`), cada recorte passa pelo modelo foliar da cultura, e a
resposta diz o que foi achado **e a que altura do chão** — pseudocaule, folha
baixa, saia, copa.

É o mesmo código que roda na Jetson do rover com ``INFERENCE_RUNTIME=onnx``: a
API e o robô não têm duas implementações do mesmo cálculo.
"""

from __future__ import annotations

import io
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import HTTP_422, settings
from app.fields_api import fields
from app.inference.preprocess import load_image_from_bytes
from app.registry import default_for_crop
from app.review import TRAINING_CONSENT_HEADER, consent_allows
from app.rover import DEFAULT_TILE_MM, plan_tiles
from app.schemas import RoverFrameResponse, RoverTile
from app.sigatoka_severity import estimate_plant_severity

router = APIRouter(tags=["rover"])

MAX_TILES = 2000
"""Teto de recortes por quadro. A 1 m de distância um quadro de 12 MP dá ~800;
acima disso alguém mandou distância errada, e a requisição levaria minutos."""

TILES_PER_CALL = 128
"""Recortes por chamada ao modelo. Limita a memória **do computador**, não da GPU: o TTA
empilha as quatro vistas do lote (~600 KB cada recorte por vista), e sem o corte o pico
crescia com o quadro — 5,2 GB para 828 recortes na borda, em CPU. Cada recorte é previsto
independente dos outros, então fatiar não muda nenhuma resposta."""

HEALTHY = "Healthy"

SIGATOKA_BOOTSTRAP = 200
"""Reamostragens do intervalo da severidade no quadro — 200 bastam para um intervalo
de 95% estável na primeira casa, e cabem na latência de um quadro."""


@router.post(
    "/predict/rover-frame",
    response_model=RoverFrameResponse,
    summary="Quadro de uma câmera lateral do rover, diagnosticado por zona da planta",
)
async def rover_frame(
    file: UploadFile = File(description="Quadro da câmera lateral (JPEG, PNG ou WebP)."),
    model: str | None = Query(
        default=None,
        description=(
            "Versão do modelo, ex.: banana-foliar, citrus-foliar-cc (ver GET /models/choices). "
            "Tem precedência sobre 'crop'."
        ),
    ),
    crop: str | None = Query(
        default=None,
        description="Só a cultura (banana, citrus): usa a versão configurada em ROVER_MODELS.",
    ),
    camera_height_mm: float = Query(gt=0, le=3000, description="Altura da câmera no mastro."),
    distance_mm: float = Query(
        gt=50, le=5000, description="Distância até a planta — mediana da profundidade da OAK-D."
    ),
    tile_mm: float = Query(default=DEFAULT_TILE_MM, gt=20, le=1000),
    overlap: float = Query(default=0.25, ge=0.0, lt=0.9),
    only_findings: bool = Query(
        default=True, description="Devolve só os recortes que não são folha sadia."
    ),
    tta: bool | None = Query(
        default=None,
        description="Média das 4 orientações em cada recorte. Padrão: ROVER_TTA (ligado).",
    ),
    field_id: str | None = Query(
        default=None,
        description=(
            "Talhão. Com ele (e row_id, side, x_m, pass_id) o quadro alimenta a fusão por "
            "planta e a prevalência do talhão — ver /fields. Exige FIELDS_DIR."
        ),
    ),
    row_id: str | None = Query(default=None, description="Linha de plantio."),
    side: Literal["left", "right"] | None = Query(
        default=None, description="Lado do rover em que está a câmera."
    ),
    x_m: float | None = Query(
        default=None,
        # NaN embaralharia a associação por planta: toda comparação com ele é falsa.
        allow_inf_nan=False,
        description="Posição do rover ao longo da linha (odometria), em metros.",
    ),
    pass_id: str | None = Query(default=None, description="Passada do rover pelo talhão."),
    lat: float | None = Query(default=None, ge=-90, le=90, description="GNSS do rover."),
    lon: float | None = Query(default=None, ge=-180, le=180, description="GNSS do rover."),
    heading_deg: float | None = Query(
        default=None, allow_inf_nan=False, description="Rumo do rover, em graus."
    ),
    camera_id: str | None = Query(default=None, description="Câmera (ex.: esq-1000)."),
    frame_id: str | None = Query(default=None, description="Quadro, para rastrear revisões."),
    captured_at: datetime | None = Query(
        default=None,
        description=(
            "Instante da captura; padrão: agora. O rover que sincroniza depois de ficar "
            "sem enlace manda o da captura — é ele que data o histórico da planta."
        ),
    ),
    training_consent: str | None = Header(
        default=None,
        alias=TRAINING_CONSENT_HEADER,
        description="1/0: o cliente autorizou o treino. Com 0, nada vai para a fila de revisão.",
    ),
) -> RoverFrameResponse:
    """Corta o quadro na escala das fotos de treino e classifica recorte a recorte.

    Um recorte entra em ``findings`` quando a classe prevista não é
    ``Healthy``, quando o conjunto conformal não é uma resposta única, ou quando
    a imagem é estranha para o modelo (``needs_review``). ``zone_summary`` diz,
    por zona da planta, quantos recortes foram vistos e quantos têm achado.

    Recorte que o portão de enquadramento recusa (pseudocaule, tronco, fruto)
    **não** é achado: o modelo foliar dá uma doença de folha para qualquer
    coisa, com confiança, e um pseudocaule sadio sairia como "Fusariose". Ele
    é contado à parte, em ``out_of_scope``, pela cena que o portão viu.
    """
    from app.main import _read_validated_upload, _resolve_ready_model, review_queue

    if model is None:
        if crop is None:
            raise HTTPException(
                HTTP_422,
                "Informe 'model' (a versão, ver GET /models/choices) ou 'crop' (a cultura).",
            )
        try:
            model = default_for_crop(crop).key
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc.args[0])) from exc
    selected = _resolve_ready_model(model, None)
    crop = selected.spec.crop
    data = await _read_validated_upload(file)
    image = load_image_from_bytes(data, file.filename or "quadro")

    try:
        tiles = plan_tiles(
            image.size, distance_mm, camera_height_mm, crop=crop, tile_mm=tile_mm, overlap=overlap
        )
    except ValueError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc
    if len(tiles) > MAX_TILES:
        raise HTTPException(
            HTTP_422,
            f"{len(tiles)} recortes passam do teto de {MAX_TILES} — confira distance_mm e tile_mm.",
        )

    crops = [image.crop(tile.box) for tile in tiles]
    names = [f"recorte_{i}" for i in range(len(tiles))]
    use_tta = settings.rover_tta if tta is None else tta
    results = await run_in_threadpool(_predict_in_chunks, selected, names, crops, use_tta)

    all_tiles = [
        RoverTile(
            box=list(tile.box),
            height_mm=tile.center_height_mm,
            zones=list(tile.zones),
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            prediction_set=result.prediction_set,
            needs_review=result.needs_review,
            in_scope=None if result.framing is None else result.framing.in_scope,
            scene=None if result.framing is None else result.framing.scene,
        )
        for tile, result in zip(tiles, results, strict=True)
    ]
    findings = [tile for tile in all_tiles if _is_finding(tile)]
    # A resposta traz o conjunto honesto; a fusão por planta recebe o da calibração
    # original (ver CornDiseaseModel.fusion_prediction_sets).
    fusion_sets = selected.fusion_prediction_sets(results, use_tta)
    observations = [
        _observation(tile, result, fusion_set)
        for tile, result, fusion_set in zip(all_tiles, results, fusion_sets, strict=True)
    ]

    queued = 0
    if (
        review_queue.enabled
        and settings.rover_review_per_frame
        and consent_allows(training_consent)
    ):
        chosen = _review_candidates(all_tiles, settings.rover_review_per_frame)
        context = {
            "source": "rover",
            "field_id": field_id,
            "row_id": row_id,
            "side": side,
            "x_m": x_m,
            "pass_id": pass_id,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "camera_height_mm": camera_height_mm,
            "distance_mm": distance_mm,
            "captured_at": _utc(captured_at).isoformat(),
        }
        queued = await run_in_threadpool(
            _enqueue_tiles,
            review_queue,
            chosen,
            crops,
            results,
            all_tiles,
            context,
            frame_id or file.filename or "quadro",
        )

    plants, prevalence = None, None
    if field_id is not None:
        missing = [
            name
            for name, value in (
                ("row_id", row_id),
                ("side", side),
                ("x_m", x_m),
                ("pass_id", pass_id),
            )
            if value is None
        ]
        if missing:
            raise HTTPException(
                HTTP_422,
                f"Com field_id, informe também: {', '.join(missing)}.",
            )
        if not fields.enabled:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Talhões desligados: configure FIELDS_DIR para guardar a fusão por planta.",
            )
        frame = {
            "field_id": field_id,
            "row_id": row_id,
            "side": side,
            "camera_height_mm": camera_height_mm,
            "distance_mm": distance_mm,
            "x_m": x_m,
            "pass_id": pass_id,
            "timestamp": _utc(captured_at).isoformat(),
            "crop": crop,
            "tiles": observations,
            "lat": lat,
            "lon": lon,
            "heading_deg": heading_deg,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "image_width_px": image.size[0],
        }
        in_scope = np.array(
            [
                [obs["probabilities"][name] for name in selected.class_names]
                for obs in observations
                if obs["in_scope"] is not False
            ]
        ).reshape(-1, len(selected.class_names))
        try:
            plants, prevalence = await run_in_threadpool(
                fields.record_frame, selected.spec, frame, in_scope
            )
        except ValueError as exc:
            # Passada já fechada, talhão de outra cultura: o quadro está certo, o
            # pedido de juntá-lo ao talhão é que conflita com o estado.
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    sigatoka = None
    if crop == "banana":
        sigatoka = _sigatoka_severity(observations)

    zone_total: Counter[str] = Counter()
    zone_findings: Counter[str] = Counter()
    zone_refused: Counter[str] = Counter()
    refused_scenes: dict[str, Counter[str]] = {}
    for tile in all_tiles:
        for zone in tile.zones or ["sem_zona"]:
            zone_total[zone] += 1
            zone_findings[zone] += _is_finding(tile)
            if tile.in_scope is False:
                zone_refused[zone] += 1
                refused_scenes.setdefault(zone, Counter())[tile.scene or "?"] += 1

    return RoverFrameResponse(
        model_key=selected.key,
        crop=crop,
        frame_size=list(image.size),
        tile_px=tiles[0].box[2] - tiles[0].box[0],
        tiles_analyzed=len(all_tiles),
        class_counts=dict(
            Counter(tile.predicted_class for tile in all_tiles if tile.in_scope is not False)
        ),
        zone_summary={
            zone: {
                "recortes": zone_total[zone],
                "com_achado": zone_findings[zone],
                "fora_de_escopo": zone_refused[zone],
            }
            for zone in zone_total
        },
        out_of_scope={zone: dict(scenes) for zone, scenes in refused_scenes.items()},
        tiles=findings if only_findings else all_tiles,
        max_forward_batch=settings.max_forward_batch,
        plants=plants,
        field_prevalence=prevalence,
        sigatoka=sigatoka,
        queued_for_review=queued,
    )


def _utc(moment: datetime | None) -> datetime:
    """O instante em UTC; sem fuso, é UTC (é o que o rover manda)."""
    if moment is None:
        return datetime.now(UTC)
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _observation(
    tile: RoverTile, result, prediction_set: list[str] | None = None
) -> dict[str, Any]:
    """O recorte com a distribuição completa — a fusão e a severidade precisam dela.

    ``prediction_set`` é o que a fusão lê; sem ele vale o conjunto do próprio recorte.
    """
    return {
        "box": tile.box,
        "predicted_class": tile.predicted_class,
        "probabilities": {p.class_name: p.probability for p in result.probabilities},
        "confidence": tile.confidence,
        "height_mm": tile.height_mm,
        "zones": tile.zones,
        "prediction_set": tile.prediction_set if prediction_set is None else prediction_set,
        "needs_review": tile.needs_review,
        "in_scope": tile.in_scope,
        "scene": tile.scene,
    }


def _sigatoka_severity(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Severidade da Sigatoka vista neste quadro, na escala de Stover-Gauhl.

    É o que uma câmera vê de uma planta; a planta inteira sai da fusão de
    várias câmeras. A escala ainda não foi validada por agrônomo
    (``VALIDATED = False`` em :mod:`app.sigatoka_severity`), e a resposta diz isso.
    """
    crops = [
        {
            "predicted_class": obs["predicted_class"],
            "probabilities": obs["probabilities"],
            "height_mm": obs["height_mm"],
            "zone": "folhas_baixas"
            if "folhas_baixas" in obs["zones"]
            else next(iter(obs["zones"]), None),
            "in_scope": obs["in_scope"],
        }
        for obs in observations
    ]
    return estimate_plant_severity(crops, n_bootstrap=SIGATOKA_BOOTSTRAP).to_dict()


def _predict_in_chunks(model: Any, names: list[str], crops: list[Any], tta: bool) -> list[Any]:
    """Prevê os recortes em fatias de :data:`TILES_PER_CALL`, na mesma ordem."""
    results: list[Any] = []
    for start in range(0, len(crops), TILES_PER_CALL):
        stop = start + TILES_PER_CALL
        results += model.predict_images(
            names[start:stop], crops[start:stop], tta, species=settings.rover_species_check
        )
    return results


def _review_candidates(tiles: list[RoverTile], limit: int) -> list[int]:
    """Os recortes do quadro que mais ensinam ao próximo treino, no máximo ``limit``.

    Primeiro o que é estranho ao modelo, depois o de conjunto conformal ambíguo
    (cada grupo do menos para o mais confiante), e um só dos que o portão recusou:
    no pseudocaule quase todo recorte é recusado, e eles inundariam a fila.
    """
    novel, ambiguous, refused = [], [], []
    for index, tile in enumerate(tiles):
        if tile.in_scope is False:
            refused.append(index)
        elif tile.needs_review:
            novel.append(index)
        elif tile.prediction_set is not None and len(tile.prediction_set) != 1:
            ambiguous.append(index)

    def by_confidence(index: int) -> float:
        return tiles[index].confidence

    ranked = sorted(novel, key=by_confidence) + sorted(ambiguous, key=by_confidence)
    return (ranked + refused[:1])[:limit]


def _enqueue_tiles(
    queue: Any,
    chosen: list[int],
    crops: list[Any],
    results: list[Any],
    tiles: list[RoverTile],
    context: dict[str, Any],
    frame_name: str,
) -> int:
    """Arquiva os recortes escolhidos, com de onde vieram; devolve quantos entraram."""
    queued = 0
    for index in chosen:
        buffer = io.BytesIO()
        crops[index].convert("RGB").save(buffer, "JPEG", quality=92)
        tile = tiles[index]
        item_id = queue.enqueue(
            buffer.getvalue(),
            results[index].model_copy(update={"filename": f"{frame_name}#recorte_{index}"}),
            "image/jpeg",
            context={
                **context,
                "tile": index,
                "box": tile.box,
                "height_mm": tile.height_mm,
                "zones": tile.zones,
                "scene": tile.scene,
            },
        )
        queued += item_id is not None
    return queued


def _is_finding(tile: RoverTile) -> bool:
    if tile.in_scope is False:
        return False
    ambiguous = tile.prediction_set is not None and len(tile.prediction_set) != 1
    return tile.predicted_class != HEALTHY or ambiguous or tile.needs_review
