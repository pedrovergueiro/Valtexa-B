"""Talhão: o que o rover viu, planta a planta e passada a passada.

O ``/predict/rover-frame`` diagnostica um quadro. Com ``field_id`` e a pose do
rover na requisição, o quadro também alimenta dois estados por talhão, gravados
em ``FIELDS_DIR``:

* **fusão por planta** (:mod:`app.plant_fusion`) — os recortes de várias câmeras
  e quadros viram um diagnóstico por **planta**, com posterior por doença,
  incidência, histórico entre passadas e um mapa GeoJSON do talhão. Com o
  ``<pesos>.fusion.json`` do modelo (``training.fit_fusion``), a evidência, os
  limiares e a posterior são os medidos em plantas rotuladas;
* **incidência do talhão** — a fração de **plantas** doentes, corrigida pela
  sensibilidade e pelo alarme falso por planta medidos na calibração, com
  intervalo de 95% (``GET /fields/{id}/incidence``);
* **prevalência dos recortes** (:mod:`app.inference.prior_shift`) — a proporção de
  cada classe entre os **recortes**, estimada sem rótulo (EM de Saerens). Não é a
  fração de plantas: num talhão simulado de bananeiras todas sadias (02/10/2026)
  ela dava 60% dos recortes como doentes, porque o modelo põe probabilidade de
  doença em pedaço de folha sadia.

Sem ``FIELDS_DIR`` nada disso liga, e o rover continua respondendo por quadro.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import HTTP_422, settings
from app.inference.prior_shift import DIVERGENCE_LIMIT, FieldPriorStore, TrackerConfig
from app.plant_fusion import FusionCalibration, PlantFusionStore, load_fusion_calibration
from app.prescricao import (
    Contagio,
    Economia,
    PrescricaoImpossivelError,
    Produto,
    prescrever,
    resumo_em_texto,
)
from app.registry import ModelSpec, get_spec

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fields", tags=["talhão"])

PRIOR_SUFFIX = ".prior.json"
"""Prevalência das classes no treino, gravada por ``training.fit_prior`` ao lado dos pesos."""


def training_prior_path(spec: ModelSpec) -> Path:
    return spec.weights_path.with_suffix(PRIOR_SUFFIX)


class FieldServices:
    """Os estados de talhão, criados sob demanda dentro de ``FIELDS_DIR``.

    A fusão não é segura entre threads (ver :class:`PlantFusionStore`) e o rover
    pode mandar quadros de várias câmeras ao mesmo tempo: um lock serializa o
    acesso. A prevalência tem lock próprio.
    """

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        self._lock = threading.RLock()
        self._fusion: PlantFusionStore | None = None
        self._priors: dict[str, FieldPriorStore | None] = {}
        self._calibrations: dict[str, FusionCalibration | None] = {}

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def fusion(self) -> PlantFusionStore:
        if self.directory is None:
            raise RuntimeError("FIELDS_DIR não configurado.")
        with self._lock:
            if self._fusion is None:
                self._fusion = PlantFusionStore(
                    self.directory / "plantas", tile_log=settings.fields_tile_log
                )
            return self._fusion

    def calibration(self, spec: ModelSpec, edge: bool = False) -> FusionCalibration | None:
        """A calibração da fusão deste modelo, ou ``None`` (valores de partida).

        ``edge``: a dos recortes classificados no Rover (``<pesos>.edge-fusion.json``), que
        não passam pelo portão de enquadramento. Lida uma vez por processo. Um arquivo
        ilegível não derruba o rover: a fusão segue sem calibração e o erro fica no log.
        """
        chave = f"{spec.key}|edge" if edge else spec.key
        with self._lock:
            if chave not in self._calibrations:
                self._calibrations[chave] = self._load_calibration(
                    spec, spec.edge_fusion_path if edge else spec.fusion_path
                )
            return self._calibrations[chave]

    @staticmethod
    def _load_calibration(spec: ModelSpec, path: Path) -> FusionCalibration | None:
        if not path.exists():
            logger.warning(
                "Sem %s — a fusão por planta de %s usa os valores de partida, sem medida. "
                "Calibre com 'python -m training.%s --model %s'.",
                path.name,
                spec.key,
                "fit_edge fusao" if path == spec.edge_fusion_path else "fit_fusion fit",
                spec.key,
            )
            return None
        try:
            calibration = load_fusion_calibration(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("Calibração da fusão de %s ilegível em %s: %s", spec.key, path, exc)
            return None
        if calibration.model_key != spec.key:
            logger.error(
                "%s foi calibrado para %r, não para %r: ignorado.",
                path,
                calibration.model_key,
                spec.key,
            )
            return None
        logger.info("Fusão por planta de %s calibrada: %s", spec.key, calibration.id)
        return calibration

    def priors(self, spec: ModelSpec) -> FieldPriorStore | None:
        """A prevalência por talhão deste modelo, ou ``None`` sem o prior do treino.

        O EM precisa saber com que prevalência o modelo foi treinado; sem o
        ``.prior.json`` a correção seria um chute, então ela simplesmente não liga.
        """
        if self.directory is None:
            return None
        with self._lock:
            if spec.key not in self._priors:
                self._priors[spec.key] = self._load_priors(spec)
            return self._priors[spec.key]

    def _load_priors(self, spec: ModelSpec) -> FieldPriorStore | None:
        state = self._prior_state_path(spec)
        if state.exists():
            try:
                return FieldPriorStore.load(state)
            except (OSError, ValueError, KeyError) as exc:
                logger.error("Prevalência de %s ilegível em %s: %s", spec.key, state, exc)
        source = training_prior_path(spec)
        if not source.exists():
            logger.info(
                "Sem %s — prevalência por talhão desligada para %s. "
                "Gere com 'python -m training.fit_prior --model %s'.",
                source.name,
                spec.key,
                spec.key,
            )
            return None
        raw = json.loads(source.read_text(encoding="utf-8"))
        prior = [float(raw["prior"][name]) for name in spec.class_names]
        # EM puro: o modo "auto" combina com BBSE e pede a matriz de confusão da
        # validação, que ainda não é gravada no treino.
        return FieldPriorStore(prior, TrackerConfig(method="em"), class_names=spec.class_names)

    def _prior_state_path(self, spec: ModelSpec) -> Path:
        assert self.directory is not None
        return self.directory / "prevalencia" / f"{spec.key}.json"

    def record_frame(
        self,
        spec: ModelSpec,
        observation: dict[str, Any],
        in_scope_probabilities: np.ndarray,
        *,
        edge: bool = False,
    ) -> tuple[list[str], dict[str, float] | None]:
        """Ingere um quadro no talhão: plantas tocadas e prevalência corrente.

        Só os recortes que o portão aceitou entram na prevalência — um pseudocaule
        classificado como doença de folha distorceria a estimativa. ``edge``: recortes
        classificados no Rover, fundidos com a calibração deles.
        """
        with self._lock:
            plants = self.fusion().ingest(
                observation, calibration=self.calibration(spec, edge=edge)
            )
        store = self.priors(spec)
        prevalence = None
        if store is not None:
            field_id = observation["field_id"]
            if len(in_scope_probabilities):
                store.update(field_id, in_scope_probabilities)
                state = self._prior_state_path(spec)
                state.parent.mkdir(parents=True, exist_ok=True)
                store.save(state)
            prevalence = dict(
                zip(spec.class_names, map(float, store.priors(field_id)), strict=True)
            )
        return plants, prevalence

    def reset(self, directory: Path | None) -> None:
        """Troca o diretório e esquece o estado em memória — é por aqui que os testes entram."""
        with self._lock:
            self.directory = directory
            self._fusion = None
            self._priors = {}
            self._calibrations = {}


fields = FieldServices(settings.fields_dir)
"""Estado de talhão único do processo."""


def _require_fields() -> PlantFusionStore:
    if not fields.enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Talhões desligados: configure FIELDS_DIR para guardar a fusão por planta.",
        )
    return fields.fusion()


def _known(store: PlantFusionStore, field_id: str) -> None:
    if field_id not in store.fields():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Talhão {field_id!r} não encontrado.")


MAX_EDGE_TILES = 2000
"""Mesmo teto de recortes por quadro do ``/predict/rover-frame``."""


class EdgeFrameIn(BaseModel):
    """Um quadro que o Rover já classificou: a geometria e as probabilidades de cada recorte.

    É o ``FrameObservation`` de :mod:`app.plant_fusion` (os nomes dos campos são os
    dele). ``frame_id`` é obrigatório: é por ele que o reenvio não conta duas vezes.
    """

    model_config = ConfigDict(extra="forbid")

    row_id: str = Field(min_length=1, max_length=80)
    side: str = Field(pattern="^(left|right)$")
    camera_height_mm: float = Field(gt=0, le=3000)
    distance_mm: float = Field(gt=0, le=5000)
    x_m: float = Field(allow_inf_nan=False)
    pass_id: str = Field(min_length=1, max_length=80)
    timestamp: str
    frame_id: str = Field(min_length=1, max_length=120)
    camera_id: str | None = Field(default=None, max_length=80)
    image_width_px: int = Field(gt=0, le=20000)
    hfov_deg: float = Field(gt=0, lt=180)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    heading_deg: float | None = Field(default=None, allow_inf_nan=False)
    plant_id: str | None = Field(default=None, max_length=80)
    plant_spacing_m: float | None = Field(
        default=None, gt=0, le=20, description="Espaçamento entre plantas do talhão (cadastro)."
    )
    tiles: list[dict[str, Any]] = Field(max_length=MAX_EDGE_TILES)


@router.post("/{field_id}/frames", summary="Funde por planta um quadro classificado no Rover")
def ingest_edge_frame(
    field_id: str,
    frame: EdgeFrameIn,
    model: str = Query(description="Modelo que classificou os recortes no Rover."),
) -> dict[str, Any]:
    """O Rover classifica cada recorte e manda as probabilidades (o quadro normal sobe
    reduzido, pequeno demais para a nuvem rodar o modelo de novo). Aqui elas entram na
    fusão por planta com a calibração medida para elas (``<pesos>.edge-fusion.json``,
    ``training.fit_edge fusao``); sem ela, com os valores de partida — ``calibration``
    nulo na resposta. Sem portão de enquadramento, a prevalência do talhão não muda.
    Reenvio do mesmo ``frame_id`` na mesma passada: ``duplicated``, nada muda.
    """
    store = _require_fields()
    try:
        spec = get_spec(model)
    except KeyError as exc:
        raise HTTPException(HTTP_422, f"Modelo {model!r} desconhecido.") from exc
    observation = {**frame.model_dump(), "field_id": field_id, "crop": spec.crop}
    try:
        with fields._lock:
            if store.has_frame(field_id, frame.pass_id, frame.frame_id):
                return {"duplicated": True, "plants": [], "calibration": None}
            plants, _ = fields.record_frame(spec, observation, np.empty((0, 0)), edge=True)
    except (KeyError, TypeError, ValueError) as exc:
        # Passada fechada, cultura diferente da do talhão, recorte malformado.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    calibration = fields.calibration(spec, edge=True)
    return {
        "duplicated": False,
        "plants": plants,
        "calibration": None if calibration is None else calibration.id,
    }


@router.get("", summary="Talhões com histórico do rover")
def list_fields() -> dict[str, list[str]]:
    return {"fields": _require_fields().fields()}


@router.get("/{field_id}/plants", summary="Diagnóstico fundido de cada planta do talhão")
def list_plants(
    field_id: str,
    pass_id: str | None = Query(default=None, description="Passada; padrão: a mais recente."),
) -> list[dict[str, Any]]:
    """Uma entrada por planta: posterior de cada doença, status, incidência, vistas.

    ``status`` é ``sadia``, ``suspeita``, ``doente`` ou ``revisar``. A posterior
    junta todos os recortes da planta, de todas as câmeras e quadros, com
    desconto para os que se sobrepõem — ver :mod:`app.plant_fusion`.
    """
    store = _require_fields()
    _known(store, field_id)
    with fields._lock:
        return [plant.to_dict() for plant in store.plants(field_id, pass_id)]


@router.get(
    "/{field_id}/plants/{plant_id}/evolution",
    summary="Histórico de uma planta entre passadas, com AUDPC",
)
def plant_evolution(field_id: str, plant_id: str) -> dict[str, Any]:
    store = _require_fields()
    _known(store, field_id)
    try:
        with fields._lock:
            return store.evolution(field_id, plant_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc.args[0])) from exc


@router.post(
    "/{field_id}/passes/{pass_id}/close",
    summary="Fecha uma passada: consolida as plantas e aponta o que mudou",
)
def close_pass(field_id: str, pass_id: str) -> dict[str, Any]:
    """Ao fim da linha: novas infecções, pioras e melhoras desde a passada anterior."""
    store = _require_fields()
    _known(store, field_id)
    try:
        with fields._lock:
            plants = store.close_pass(field_id, pass_id)
            changes = store.changes_in_pass(field_id, pass_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc.args[0])) from exc
    return {"plants": [plant.to_dict() for plant in plants], "changes": changes}


@router.get("/{field_id}/map", summary="Mapa do talhão em GeoJSON, uma planta por ponto")
def field_map(
    field_id: str,
    pass_id: str | None = Query(default=None, description="Passada; padrão: a mais recente."),
) -> JSONResponse:
    """Abre direto num SIG (QGIS) ou num mapa web.

    Com GPS em todos os quadros, as coordenadas são WGS84; sem, são métricas
    locais (metros ao longo da linha), marcadas como tal em cada ponto.
    """
    store = _require_fields()
    _known(store, field_id)
    with fields._lock:
        geojson = store.export_geojson(field_id, pass_id)
    return JSONResponse(geojson, media_type="application/geo+json")


@router.get("/{field_id}/incidence", summary="Fração de plantas doentes no talhão, com intervalo")
def field_incidence(
    field_id: str,
    pass_id: str | None = Query(default=None, description="Passada; padrão: a última."),
) -> dict[str, Any]:
    """Quantas **plantas** estão doentes, não quantos recortes.

    ``apparent`` é a fração de plantas sinalizadas ("suspeita" ou "doente");
    ``incidence`` a corrige pela sensibilidade e pelo alarme falso por planta medidos
    na calibração da fusão (Rogan-Gladen), com ``ci95``. Sem calibração para o
    modelo, só a aparente sai, e ``method`` diz ``aparente``. Plantas "revisar" ficam
    fora da conta (``status_counts`` mostra quantas).
    """
    store = _require_fields()
    _known(store, field_id)
    try:
        with fields._lock:
            return store.incidence(field_id, pass_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc.args[0])) from exc


class _Entrada(BaseModel):
    """Recusa NaN e infinito: um preço NaN passaria por toda comparação como falso."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class ProdutoIn(_Entrada):
    """O defensivo como está na bula e na nota fiscal."""

    nome: str = Field(min_length=1)
    registro_mapa: str = Field(min_length=1, description="Número de registro no MAPA.")
    culturas: list[str] = Field(min_length=1, description="Culturas da bula: banana, citrus.")
    doencas: list[str] = Field(min_length=1, description="Alvos da bula, nas classes do modelo.")
    dose_ml_por_planta: float = Field(gt=0)
    dose_min_ml_por_planta: float = Field(ge=0)
    dose_max_ml_por_planta: float = Field(gt=0)
    preco_por_litro: float = Field(ge=0)
    eficacia: float = Field(gt=0, le=1, description="Fração do dano evitada. Da bula, não nossa.")
    carencia_dias: int = Field(ge=0)


class EconomiaIn(_Entrada):
    """Os números do produtor. Sem padrão, de propósito."""

    perda_por_planta: float = Field(gt=0, description="Quanto vale perder uma planta na safra.")
    custo_operacional_por_planta: float = Field(default=0.0, ge=0)
    orcamento: float | None = Field(default=None, ge=0, description="Teto de gasto; nulo = sem.")


class ContagioIn(_Entrada):
    """Alcance do contágio. Os padrões são escolha do módulo, não medição."""

    alcance_na_rua_m: float = Field(default=3.0, gt=0)
    alcance_entre_ruas_m: float = Field(default=1.5, gt=0)
    ajustado: bool = Field(default=False, description="True só se ajustado neste talhão.")


class PrescricaoIn(_Entrada):
    doenca: str = Field(min_length=1)
    produto: ProdutoIn
    economia: EconomiaIn
    contagio: ContagioIn | None = None
    dias_ate_colheita: int | None = Field(default=None, ge=0)
    pass_id: str | None = None


@router.post("/{field_id}/prescription", summary="Em quais plantas o tratamento se paga")
def field_prescription(
    field_id: str,
    pedido: PrescricaoIn,
    formato: str = Query(default="json", pattern="^(json|geojson)$"),
) -> JSONResponse:
    """Prescrição planta a planta, com o custo, o que ela poupa e o risco que sobra.

    A decisão não é um limiar sobre a posterior: tratar uma planta vale a planta
    **mais o que ela infectaria** se ficasse sem tratamento — ver
    :mod:`app.prescricao`. É daí que sai, sozinha, a borda em volta da reboleira.

    Os preços, a perda por planta e a eficácia são do produtor e entram na
    requisição; o módulo não tem padrão para eles e recusa sem eles. Responde
    **422** quando a bula, a carência ou o registro proíbem a aplicação.

    É recomendação: nada é pulverizado sem o agrônomo aprovar.
    """
    store = _require_fields()
    _known(store, field_id)
    with fields._lock:
        plantas = store.plants(field_id, pedido.pass_id)
    if not plantas:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Talhão {field_id!r} sem plantas nesta passada: rode uma passada antes.",
        )
    crop = plantas[0].crop
    try:
        resultado = prescrever(
            plantas,
            doenca=pedido.doenca,
            produto=Produto(
                **pedido.produto.model_dump(exclude={"culturas", "doencas"}),
                culturas=tuple(pedido.produto.culturas),
                doencas=tuple(pedido.produto.doencas),
            ),
            economia=Economia(**pedido.economia.model_dump()),
            crop=crop,
            field_id=field_id,
            pass_id=pedido.pass_id or plantas[0].pass_id,
            espacamento_entre_ruas_m=store.config.crop(crop).row_spacing_m,
            contagio=Contagio(**pedido.contagio.model_dump()) if pedido.contagio else None,
            dias_ate_colheita=pedido.dias_ate_colheita,
        )
    except PrescricaoImpossivelError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc
    if formato == "geojson":
        return JSONResponse(resultado.to_geojson(), media_type="application/geo+json")
    return JSONResponse({**resultado.to_dict(), "resumo": resumo_em_texto(resultado)})


@router.get("/{field_id}/prevalence", summary="Prevalência das doenças no talhão, sem rótulo")
def field_prevalence(
    field_id: str,
    model: str = Query(description="Modelo, ex.: banana-foliar."),
) -> dict[str, Any]:
    """Quanto de cada doença há no talhão, estimado dos próprios diagnósticos (EM).

    ``training_prior`` é a proporção do dataset de treino; ``field_prior``, a dos
    **recortes** do talhão. Quando elas se afastam, as probabilidades do modelo
    precisam da correção — é o caso típico de doença rara num talhão sadio.

    Para saber quantas **plantas** estão doentes, use ``plant_incidence`` (o mesmo
    de ``GET /fields/{id}/incidence``, na última passada): a proporção de recortes
    não é isso, e erra muito quando o modelo põe probabilidade de doença em pedaço
    de folha sadia.
    """
    fusion = _require_fields()
    try:
        spec = get_spec(model)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc.args[0])) from exc
    store = fields.priors(spec)
    if store is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Sem o prior de treino de {model!r}: rode 'python -m training.fit_prior "
            f"--model {model}'.",
        )
    estimate = store.estimate(field_id)
    shares = store.predicted_shares(field_id)
    divergence = store.divergence(field_id)
    reliable = None if divergence is None else divergence <= DIVERGENCE_LIMIT
    plant_incidence = None
    with fields._lock:
        if field_id in fusion.fields():
            try:
                plant_incidence = fusion.incidence(field_id)
            except KeyError:
                plant_incidence = None
    return {
        "model_key": spec.key,
        "field_id": field_id,
        "training_prior": dict(zip(spec.class_names, map(float, store.source_prior), strict=True)),
        "field_prior": dict(zip(spec.class_names, map(float, estimate.prevalences), strict=True)),
        "predicted_shares": None
        if shares is None
        else dict(zip(spec.class_names, map(float, shares), strict=True)),
        "divergence": divergence,
        "reliable": reliable,
        "note": (
            "Poucos recortes vistos: ainda vale o prior do treino."
            if reliable is None
            else None
            if reliable
            else (
                "Estimativa instável: o EM e a contagem dos palpites discordam, sinal de que o "
                "modelo não está calibrado neste talhão. Use como indicação, não como medida."
            )
        ),
        "method": estimate.method,
        "effective_samples": float(estimate.n_samples),
        "plant_incidence": plant_incidence,
    }
