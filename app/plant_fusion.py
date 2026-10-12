"""Fusão multivista por planta, mapa do talhão e evolução entre passadas (VALTEXA-B).

Apps de mercado classificam uma foto solta. O rover, ao contrário, vê a MESMA planta em
até 6 câmeras e em vários quadros consecutivos. Este módulo junta os recortes classificados
num diagnóstico por planta, com incerteza explícita, e acompanha cada planta ao longo das
passadas. Depende só da stdlib e do numpy para poder ser encaixado no backend FastAPI sem
arrastar dependências.

Visão geral do fluxo
--------------------
1. ``FrameObservation``: um quadro de uma câmera, já fatiado e classificado (``TileResult``).
2. Cada recorte recebe uma posição ao longo da linha (``x`` em metros, mesmo referencial da
   odometria do rover) a partir da coluna do pixel e do GSD.
3. O recorte é associado a uma planta (``plant_id`` explícito ou agrupamento 1D por posição).
4. A evidência do recorte vira log-razão de verossimilhança (LLR) por doença, acumulada por
   quadro; na consulta, quadros e câmeras são combinados com desconto de correlação.
   Com uma ``FusionCalibration`` medida (``training.fit_fusion``), a LLR do recorte, os
   limiares do status e a escala da posterior vêm de plantas rotuladas; sem ela, dos
   valores de partida escritos à mão. Cada passada guarda a calibração com que foi
   ingerida, então trocar a calibração não mistura evidências de modelos diferentes.
5. Ao fechar a passada, plantas provisórias são consolidadas/congeladas e um retrato por
   planta vai para o histórico (evolução temporal e AUDPC).
6. ``PlantFusionStore`` persiste o estado de cada talhão num JSON (escrita atômica),
   exporta o mapa em GeoJSON e estima a incidência do talhão (fração de plantas doentes).

Convenção geométrica (IMPORTANTE)
---------------------------------
* ``x`` cresce no sentido de avanço do rover ao longo da linha (odometria ``x_m``).
* As câmeras laterais olham perpendicularmente à linha, montadas "em pé" e sem espelhamento.
  Para quem olha para a ESQUERDA, a mão direita aponta para frente; logo, na câmera
  esquerda, colunas maiores (``u`` crescente) estão mais à frente (+x). Na câmera direita é
  o contrário: ``u`` crescente aponta para trás (-x). Daí ``SIDE_SIGN``: left=+1, right=-1.
* Deslocamento do recorte = (centro do recorte em px - centro do quadro em px) * GSD * sinal.
  GSD horizontal (mm/px) = 2 * d * tan(FOV_H / 2) / largura_px, com d = distância da câmera
  à planta (profundidade). A 270 mm e 127 graus, o quadro cobre ~1,08 m da linha.
* Se a montagem real for espelhada ou a câmera tiver deslocamento longitudinal em relação ao
  ponto de referência da odometria, ajuste ``SIDE_SIGN``/``camera_forward_offset_m``.

Hipóteses de identidade de linha
--------------------------------
``row_id`` identifica a LINHA DE PLANTAS observada, não o corredor por onde o rover passa.
Se o rover anda entre duas linhas, as câmeras da esquerda e da direita veem linhas
diferentes e o chamador deve informar o ``row_id`` de cada uma. Plantas são indexadas por
(talhão, linha) e agrupadas só dentro da mesma linha.
"""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------------------------
# Constantes do domínio
# ---------------------------------------------------------------------------------------------

HEALTHY = "Healthy"

CROP_CLASSES: dict[str, tuple[str, ...]] = {
    "banana": ("Cordana", "Fusarium_Wilt", "Healthy", "Pestalotiopsis", "Sigatoka"),
    "citrus": (
        "Algal_Leaf_Spot",
        "Anthracnose",
        "Black_Spot",
        "Canker",
        "Curl_Leaf",
        "Greening",
        "Healthy",
        "Leafminer",
        "Melanose",
        "Mites",
        "Scab",
        "Sooty_Mold",
    ),
    "coffee": ("Cercospora_Leaf_Spot", "Healthy", "Leaf_Miner_Damage", "Phoma_Leaf_Spot", "Rust"),
}

# Aceita o nome em português usado no restante do projeto.
CROP_ALIASES: dict[str, str] = {
    "banana": "banana",
    "citrus": "citrus",
    "citros": "citrus",
    "coffee": "coffee",
    "cafe": "coffee",
    "café": "coffee",
}

# Sinal que converte coluna de pixel em deslocamento ao longo da linha (ver docstring do módulo).
SIDE_SIGN: dict[str, float] = {"left": 1.0, "right": -1.0}

FRAME_WIDTH_PX = 4056
FRAME_HEIGHT_PX = 3040
HFOV_DEG = 127.0
VFOV_DEG = 110.0

# Raio médio da Terra (IUGG). Para deslocamentos de poucos metros a projeção equirretangular
# local tem erro desprezível (sub-milimétrico) frente ao erro do GNSS do rover.
EARTH_RADIUS_M = 6_371_008.8

STATUS_HEALTHY = "sadia"
STATUS_SUSPECT = "suspeita"
STATUS_DISEASED = "doente"
STATUS_REVIEW = "revisar"

# Ordem de gravidade usada para classificar mudanças entre passadas ("revisar" fica de fora:
# não é gravidade, é falta de confiança).
_SEVERITY: dict[str, int] = {STATUS_HEALTHY: 0, STATUS_SUSPECT: 1, STATUS_DISEASED: 2}

CHANGE_FIRST = "primeira_observacao"
CHANGE_NEW_INFECTION = "nova_infeccao"
CHANGE_WORSENING = "piora"
CHANGE_IMPROVEMENT = "melhora"
CHANGE_STABLE = "estavel"
CHANGE_DISEASE_SWAP = "troca_de_doenca"
CHANGE_UNDETERMINED = "indeterminado"

UNKNOWN_SCENE = "desconhecida"
SCHEMA_VERSION = 1

# Faixas da probabilidade de uma doença no recorte em que a LLR medida é tabelada. Mais
# finas nas pontas, onde o classificador concentra a massa e onde está a evidência.
LLR_BIN_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0)
FUSION_CALIBRATION_VERSION = 1
# Sorteios do intervalo da incidência: 4.000 dão os percentis de 2,5 e 97,5% com erro de
# Monte Carlo abaixo de meio ponto, em ~1 ms.
INCIDENCE_DRAWS = 4000
INCIDENCE_SEED = 20261002


# ---------------------------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionParams:
    """Sensibilidade/especificidade de UM recorte como "teste" da doença na planta.

    Não são as métricas do classificador em imagens de folha rotuladas: são a chance de um
    recorte qualquer de uma planta doente sair positivo (``sensitivity``) e a chance de um
    recorte de planta sem a doença sair negativo (``specificity``). A sensibilidade por
    recorte é baixa de propósito: numa planta doente a maioria dos recortes mostra tecido
    sadio (lesões localizadas, folhas novas). Isso também impede que muitos recortes
    sadios "apaguem" uma lesão real: a LLR negativa de cada um fica pequena.
    """

    sensitivity: float = 0.30
    specificity: float = 0.97


def _default_detection() -> dict[str, DetectionParams]:
    """Valores de partida por classe, NÃO calibrados; devem vir de dados de campo rotulados.

    Racional dos ajustes em relação ao padrão (0,30 / 0,97):
    * Fusarium_Wilt: doença sistêmica, amarelecimento em muitas folhas -> mais recortes
      positivos por planta doente.
    * Sigatoka: estrias/manchas espalhadas nas folhas mais velhas -> sensibilidade um pouco
      maior.
    * Greening: o mosqueado se confunde com deficiência nutricional -> especificidade menor.
    * Sooty_Mold: fumagina cobre áreas grandes, mas sujeira/sombra gera falsos positivos.
    * Leafminer/Mites: dano pequeno e localizado -> poucos recortes positivos.
    * Café: ferrugem espalhada nas folhas do terço médio e inferior -> sensibilidade um
      pouco maior; bicho-mineiro, lesão por folha -> poucos recortes positivos.
    """
    return {
        "Fusarium_Wilt": DetectionParams(0.45, 0.97),
        "Sigatoka": DetectionParams(0.35, 0.97),
        "Greening": DetectionParams(0.35, 0.93),
        "Sooty_Mold": DetectionParams(0.40, 0.95),
        "Leafminer": DetectionParams(0.20, 0.97),
        "Mites": DetectionParams(0.20, 0.96),
        "Rust": DetectionParams(0.35, 0.96),
        "Leaf_Miner_Damage": DetectionParams(0.20, 0.97),
    }


@dataclass
class CropConfig:
    """Parâmetros agronômicos e do modelo de evidência por cultura."""

    classes: tuple[str, ...]
    plant_spacing_m: float
    row_spacing_m: float
    default_prior: float = 0.05
    priors: dict[str, float] = field(default_factory=dict)
    default_detection: DetectionParams = field(default_factory=DetectionParams)
    detection: dict[str, DetectionParams] = field(default_factory=_default_detection)

    @property
    def diseases(self) -> tuple[str, ...]:
        """Classes de doença (todas menos a sadia), na ordem usada nos vetores de LLR."""
        return tuple(c for c in self.classes if c != HEALTHY)

    def prior(self, disease: str) -> float:
        return self.priors.get(disease, self.default_prior)

    def params(self, disease: str) -> DetectionParams:
        return self.detection.get(disease, self.default_detection)


def default_crops() -> dict[str, CropConfig]:
    """Espaçamentos típicos: banana ~2,5 m entre plantas, citros ~4 m (ruas de 3 e 6 m),
    café ~0,7 m na linha e 3,5 m entre ruas (adensado do Sul de Minas).

    No cafezal os pés se tocam: a 0,7 m, a "planta" da fusão é na prática um trecho de
    linha do tamanho de um pé. É a unidade que o agrônomo amostra e a que a prescrição
    trata; o espaçamento do talhão no SaaS (``plant_spacing_m``) substitui o padrão.
    """
    return {
        "banana": CropConfig(CROP_CLASSES["banana"], plant_spacing_m=2.5, row_spacing_m=3.0),
        "citrus": CropConfig(CROP_CLASSES["citrus"], plant_spacing_m=4.0, row_spacing_m=6.0),
        "coffee": CropConfig(CROP_CLASSES["coffee"], plant_spacing_m=0.7, row_spacing_m=3.5),
    }


@dataclass
class FusionConfig:
    """Configuração da fusão.

    A LLR por recorte (que depende de sensibilidade/especificidade e dos pesos de revisão)
    é calculada NA INGESTÃO e persistida somada; priors, correlações, limiares e carryover
    são aplicados NA CONSULTA e podem ser trocados sem reprocessar.
    """

    crops: dict[str, CropConfig] = field(default_factory=default_crops)
    # Tolerância de associação = fração do espaçamento. Abaixo de 0,5 garante que um recorte
    # nunca fica "no meio" de duas plantas conhecidas e casa com as duas.
    association_tolerance_frac: float = 0.45
    # Correlação entre recortes do mesmo quadro (sobreposição de 25%, mesma luz e foco).
    tile_correlation: float = 0.5
    # Correlação entre quadros consecutivos da mesma câmera na mesma passada (veem as
    # mesmas folhas de ângulos quase iguais).
    frame_correlation: float = 0.6
    # Limite do |LLR| de um recorte: um único recorte nunca decide a planta sozinho.
    llr_clip: float = 3.0
    # Peso da evidência de recortes marcados como novidade (needs_review).
    review_weight: float = 0.5
    suspect_threshold: float = 0.5
    diseased_threshold: float = 0.9
    # Fração de recortes needs_review acima da qual uma planta sem achado vira "revisar".
    review_fraction_threshold: float = 0.5
    # Fração do log-odds (acima do prior) herdada da passada anterior. 0 = cada passada é
    # julgada só pela própria evidência (o padrão, para não mascarar melhora/piora).
    temporal_carryover: float = 0.0
    # Variação mínima de incidência para "piora"/"melhora" quando o status não muda.
    incidence_change_delta: float = 0.15
    # Distância lateral extra do eixo do rover até o sensor (somada à profundidade) no mapa.
    camera_lateral_offset_m: float = 0.0
    max_review_refs: int = 50

    def crop(self, crop: str) -> CropConfig:
        try:
            return self.crops[crop]
        except KeyError as exc:
            raise ValueError(f"cultura sem configuração: {crop!r}") from exc


@dataclass(frozen=True)
class FusionCalibration:
    """Modelo de evidência MEDIDO para um modelo de diagnóstico (``training.fit_fusion``).

    Troca os chutes de ``DetectionParams`` por números tirados de plantas rotuladas:

    * ``llr_tables``: a LLR de um recorte por faixa (``bin_edges``) da probabilidade da
      doença, P(faixa | recorte de planta COM a doença) / P(faixa | recorte de planta
      SEM ela). Pega o que a média das probabilidades perde: quando o modelo põe muita
      probabilidade de uma doença em folha sadia, só a cauda perto de 1 separa as plantas.
    * ``tile_correlation`` e ``llr_clip``: os da maior detecção em validação cruzada.
    * Platt, ``evidence_offset + evidence_scale * L``: converte a soma fundida L numa
      log-razão de verossimilhança por planta, para a posterior ser probabilidade de
      verdade (a prescrição pesa custo contra ela).
    * ``suspect_evidence``/``diseased_evidence``: limiares na evidência calibrada fixados
      pelo alarme falso em plantas SADIAS; não dependem da prevalência do talhão.
    * ``plant_sensitivity``/``plant_false_alarm``: as taxas por planta medidas fora da
      amostra (``n_diseased`` doentes, ``n_healthy`` sadias), que corrigem a incidência.

    Vale para um modelo (``model_key``) e para recortes sem TTA, como o rover manda.
    """

    id: str
    model_key: str
    crop: str
    bin_edges: tuple[float, ...]
    llr_tables: Mapping[str, tuple[float, ...]]
    llr_clip: float
    tile_correlation: float
    evidence_offset: float
    evidence_scale: float
    suspect_evidence: float
    diseased_evidence: float
    plant_sensitivity: float
    plant_false_alarm: float
    n_diseased: int
    n_healthy: int
    # Tamanho efetivo das amostras que mediram as duas taxas: plantas da mesma fonte se
    # parecem, e a taxa varia entre amostras mais que a binomial de n plantas. None = n.
    sensitivity_n_eff: float | None = None
    false_alarm_n_eff: float | None = None

    def __post_init__(self) -> None:
        edges = self.bin_edges
        if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError("bin_edges tem de ir de 0 a 1")
        if any(b <= a for a, b in itertools.pairwise(edges)):
            raise ValueError("bin_edges tem de ser estritamente crescente")
        for disease, table in self.llr_tables.items():
            if len(table) != len(edges) - 1:
                raise ValueError(
                    f"tabela de {disease!r} com {len(table)} faixas, esperava {len(edges) - 1}"
                )
            if not all(math.isfinite(v) for v in table):
                raise ValueError(f"tabela de {disease!r} com valor não finito")
        if not self.llr_clip > 0:
            raise ValueError("llr_clip tem de ser positivo")
        if not 0.0 <= self.tile_correlation <= 1.0:
            raise ValueError("tile_correlation fora de [0, 1]")
        if not self.evidence_scale > 0:
            raise ValueError(
                "evidence_scale tem de ser positiva (evidência a favor não "
                "pode baixar a posterior)"
            )
        if not self.suspect_evidence <= self.diseased_evidence:
            raise ValueError("o limiar de 'doente' não pode ficar abaixo do de 'suspeita'")
        if self.suspect_evidence < -1e-9:
            # Evidência calibrada < 0 = razão de verossimilhança < 1: a planta pende para
            # sadia. Um limiar assim marcaria "suspeita" planta com 0,1% de posterior.
            raise ValueError("o limiar de 'suspeita' não pode aceitar evidência contra a doença")
        if not 0.0 <= self.plant_false_alarm < self.plant_sensitivity <= 1.0:
            raise ValueError("precisa 0 <= alarme falso < sensibilidade <= 1 por planta")
        if self.n_diseased < 1 or self.n_healthy < 1:
            raise ValueError("a calibração precisa de plantas doentes e sadias")
        for n_eff, n in (
            (self.sensitivity_n_eff, self.n_diseased),
            (self.false_alarm_n_eff, self.n_healthy),
        ):
            if n_eff is not None and not 0 < n_eff <= n:
                raise ValueError("tamanho efetivo fora de (0, n]")

    def check_crop(self, crop: str, diseases: Iterable[str]) -> None:
        """Falha cedo se a calibração não serve para esta cultura e estas doenças."""
        if normalize_crop(self.crop) != normalize_crop(crop):
            raise ValueError(f"calibração {self.id!r} é de {self.crop}, não de {crop}")
        missing = sorted(set(diseases) - set(self.llr_tables))
        if missing:
            raise ValueError(f"calibração {self.id!r} sem tabela para: {', '.join(missing)}")

    def tile_llr(self, disease: str, p: float) -> float:
        """LLR de um recorte com probabilidade ``p`` da doença: a da faixa, limitada."""
        p = min(max(p, 0.0), 1.0)
        i = min(max(bisect.bisect_right(self.bin_edges, p) - 1, 0), len(self.bin_edges) - 2)
        value = self.llr_tables[disease][i]
        return min(max(value, -self.llr_clip), self.llr_clip)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bin_edges"] = list(self.bin_edges)
        data["llr_tables"] = {d: list(t) for d, t in sorted(self.llr_tables.items())}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FusionCalibration:
        return cls(
            id=str(data["id"]),
            model_key=str(data["model_key"]),
            crop=str(data["crop"]),
            bin_edges=tuple(float(v) for v in data["bin_edges"]),
            llr_tables={str(d): tuple(float(v) for v in t) for d, t in data["llr_tables"].items()},
            llr_clip=float(data["llr_clip"]),
            tile_correlation=float(data["tile_correlation"]),
            evidence_offset=float(data["evidence_offset"]),
            evidence_scale=float(data["evidence_scale"]),
            suspect_evidence=float(data["suspect_evidence"]),
            diseased_evidence=float(data["diseased_evidence"]),
            plant_sensitivity=float(data["plant_sensitivity"]),
            plant_false_alarm=float(data["plant_false_alarm"]),
            n_diseased=int(data["n_diseased"]),
            n_healthy=int(data["n_healthy"]),
            sensitivity_n_eff=_optional_float(data.get("sensitivity_n_eff")),
            false_alarm_n_eff=_optional_float(data.get("false_alarm_n_eff")),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_fusion_calibration(path: str | os.PathLike[str]) -> FusionCalibration:
    """Lê o ``<pesos>.fusion.json`` gravado por ``training.fit_fusion``.

    O arquivo traz também o relatório da validação cruzada; aqui só interessa o bloco
    ``calibration``.
    """
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    version = data.get("version")
    if version != FUSION_CALIBRATION_VERSION:
        raise ValueError(f"versão de calibração da fusão não suportada: {version!r}")
    return FusionCalibration.from_dict(data["calibration"])


# ---------------------------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------------------------


def normalize_crop(crop: str) -> str:
    key = crop.strip().lower()
    if key not in CROP_ALIASES:
        raise ValueError(f"cultura desconhecida: {crop!r}")
    return CROP_ALIASES[key]


def parse_timestamp(value: str) -> datetime:
    """ISO 8601 -> datetime com fuso. Sem fuso, assume UTC (o rover grava em UTC)."""
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


@dataclass
class TileResult:
    """Um recorte classificado, no formato já produzido pelo pipeline do projeto real."""

    box: tuple[float, float, float, float]
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float = 0.0
    height_mm: float | None = None
    zones: list[str] = field(default_factory=list)
    prediction_set: list[str] | None = None
    needs_review: bool = False
    in_scope: bool | None = None
    scene: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TileResult:
        box = data["box"]
        pset = data.get("prediction_set")
        return cls(
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            predicted_class=str(data["predicted_class"]),
            probabilities={str(k): float(v) for k, v in data.get("probabilities", {}).items()},
            confidence=float(data.get("confidence", 0.0)),
            height_mm=None if data.get("height_mm") is None else float(data["height_mm"]),
            zones=[str(z) for z in data.get("zones") or []],
            prediction_set=None if pset is None else [str(c) for c in pset],
            needs_review=bool(data.get("needs_review", False)),
            in_scope=data.get("in_scope"),
            scene=data.get("scene"),
        )

    @property
    def center_x_px(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def counts_for_diagnosis(self) -> bool:
        """``in_scope`` None = portão não rodou; só False descarta o recorte."""
        return self.in_scope is not False


@dataclass
class FrameObservation:
    """Um quadro de uma câmera numa passada, com seus recortes já classificados."""

    field_id: str
    row_id: str
    side: str
    camera_height_mm: float
    distance_mm: float
    x_m: float
    pass_id: str
    timestamp: str
    crop: str
    tiles: list[TileResult] = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    plant_id: str | None = None
    frame_id: str | None = None
    camera_id: str | None = None
    image_width_px: int = FRAME_WIDTH_PX
    hfov_deg: float = HFOV_DEG
    # Posição longitudinal da câmera em relação ao ponto de referência da odometria/GNSS.
    camera_forward_offset_m: float = 0.0
    # Espaçamento entre plantas do talhão (cadastro do SaaS). No café varia de 0,5 a 1 m
    # entre fazendas; sem ele, vale o padrão da cultura (``default_crops``).
    plant_spacing_m: float | None = None

    def __post_init__(self) -> None:
        self.side = self.side.strip().lower()
        if self.side not in SIDE_SIGN:
            raise ValueError(f"lado inválido: {self.side!r} (use 'left' ou 'right')")
        self.crop = normalize_crop(self.crop)
        if not self.distance_mm > 0:
            raise ValueError("distance_mm deve ser positiva")
        self.row_id = str(self.row_id)
        parse_timestamp(self.timestamp)  # falha cedo com timestamp inválido

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameObservation:
        def opt_float(key: str) -> float | None:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            field_id=str(data["field_id"]),
            row_id=str(data["row_id"]),
            side=str(data["side"]),
            camera_height_mm=float(data["camera_height_mm"]),
            distance_mm=float(data["distance_mm"]),
            x_m=float(data["x_m"]),
            pass_id=str(data["pass_id"]),
            timestamp=str(data["timestamp"]),
            crop=str(data["crop"]),
            tiles=[
                t if isinstance(t, TileResult) else TileResult.from_dict(t)
                for t in data.get("tiles", [])
            ],
            lat=opt_float("lat"),
            lon=opt_float("lon"),
            heading_deg=opt_float("heading_deg"),
            plant_id=data.get("plant_id"),
            frame_id=data.get("frame_id"),
            camera_id=data.get("camera_id"),
            image_width_px=int(data.get("image_width_px", FRAME_WIDTH_PX)),
            hfov_deg=float(data.get("hfov_deg", HFOV_DEG)),
            camera_forward_offset_m=float(data.get("camera_forward_offset_m", 0.0)),
            plant_spacing_m=opt_float("plant_spacing_m"),
        )

    @property
    def camera_key(self) -> str:
        """Câmera física: o id explícito ou lado + altura (são 6 combinações únicas)."""
        if self.camera_id:
            return str(self.camera_id)
        return f"{self.side}-{round(self.camera_height_mm)}mm"

    @property
    def frame_key(self) -> str:
        if self.frame_id:
            return str(self.frame_id)
        return f"{self.camera_key}|{self.timestamp}|{self.x_m:.3f}"

    @property
    def time(self) -> datetime:
        return parse_timestamp(self.timestamp)

    @property
    def gsd_mm_per_px(self) -> float:
        return ground_sample_distance_mm(self.distance_mm, self.hfov_deg, self.image_width_px)

    def tile_along_row_m(self, tile: TileResult) -> float:
        """Posição absoluta do centro do recorte ao longo da linha (m)."""
        offset = along_row_offset_m(
            tile.center_x_px, self.image_width_px, self.gsd_mm_per_px, self.side
        )
        return self.x_m + self.camera_forward_offset_m + offset

    @property
    def has_geo(self) -> bool:
        return self.lat is not None and self.lon is not None


# ---------------------------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------------------------


def ground_sample_distance_mm(distance_mm: float, fov_deg: float, pixels: int) -> float:
    """mm por pixel no plano da planta: largura coberta (2 d tan(fov/2)) / pixels."""
    return 2.0 * distance_mm * math.tan(math.radians(fov_deg) / 2.0) / pixels


def along_row_offset_m(center_x_px: float, image_width_px: int, gsd_mm: float, side: str) -> float:
    """Deslocamento (m) do centro do recorte em relação ao eixo óptico, no sentido +x."""
    return (center_x_px - image_width_px / 2.0) * gsd_mm * SIDE_SIGN[side] / 1000.0


def project_local_offset(
    lat: float, lon: float, heading_deg: float, forward_m: float, right_m: float
) -> tuple[float, float]:
    """Soma um deslocamento local (frente/direita do rover) a uma posição lat/lon.

    Projeção equirretangular local: norte = frente*cos(h) - direita*sin(h),
    leste = frente*sin(h) + direita*cos(h), com ``heading`` em graus a partir do norte, sentido
    horário. dlat = norte / R e dlon = leste / (R cos(lat)). Para dezenas de metros o erro
    é irrelevante frente ao GNSS; perto dos polos não vale (não é o caso de pomares).
    """
    h = math.radians(heading_deg)
    north = forward_m * math.cos(h) - right_m * math.sin(h)
    east = forward_m * math.sin(h) + right_m * math.cos(h)
    dlat = math.degrees(north / EARTH_RADIUS_M)
    dlon = math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


# ---------------------------------------------------------------------------------------------
# Modelo de evidência
# ---------------------------------------------------------------------------------------------


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    x = min(max(x, -40.0), 40.0)
    return 1.0 / (1.0 + math.exp(-x))


def tile_log_likelihood_ratio(p: float, params: DetectionParams, clip: float) -> float:
    """LLR de um recorte para "a planta tem d", tratando a probabilidade como evidência suave.

    Com s = sensibilidade e f = 1 - especificidade, o recorte é um teste binário cujo
    resultado é incerto: "positivo" com probabilidade p (a saída do classificador). Pela
    regra de evidência virtual:
        LR = (s p + (1 - s)(1 - p)) / (f p + (1 - f)(1 - p))
    p = 1 -> s/f (positivo firme); p = 0 -> (1-s)/(1-f) (negativo firme, fraco de propósito);
    p = 0,5 -> 1 (neutro). É limitada e monotônica, então um softmax exagerado não explode.
    """
    p = min(max(p, 0.0), 1.0)
    s = params.sensitivity
    f = 1.0 - params.specificity
    num = s * p + (1.0 - s) * (1.0 - p)
    den = f * p + (1.0 - f) * (1.0 - p)
    llr = math.log(max(num, 1e-12) / max(den, 1e-12))
    return min(max(llr, -clip), clip)


def tile_weight(tile: TileResult, review_weight: float) -> float:
    """Peso da evidência do recorte: novidade vale menos; conjunto conformal grande também.

    Um conjunto conformal com k classes diz que o modelo não distingue entre elas no nível
    de cobertura escolhido; dividir por k atenua a evidência desse recorte ambíguo.
    """
    weight = review_weight if tile.needs_review else 1.0
    if tile.prediction_set:
        weight /= max(len(tile.prediction_set), 1)
    return weight


def effective_sample_size(n: int, rho: float) -> float:
    """Tamanho efetivo de n amostras igualmente correlacionadas: n / (1 + (n - 1) rho).

    rho = 0 -> n (independentes); rho = 1 -> 1 (cópias). É o desconto de correlação usado
    para recortes do mesmo quadro e para quadros da mesma câmera na mesma passada.
    """
    if n <= 0:
        return 0.0
    return n / (1.0 + (n - 1) * rho)


def classify_status(
    max_posterior: float,
    n_in_scope: int,
    review_fraction: float,
    cfg: FusionConfig,
    *,
    evidence: float | None = None,
    calibration: FusionCalibration | None = None,
) -> str:
    """Achado firme vence; sem achado, muita novidade ou nenhuma evidência -> revisar.

    Com calibração, o achado sai da ``evidence`` (log-razão calibrada da doença mais
    forte) contra limiares fixados pelo alarme falso em plantas sadias, e não depende do
    prior. Sem, sai da posterior contra ``suspect_threshold``/``diseased_threshold``.
    """
    if n_in_scope == 0:
        return STATUS_REVIEW
    if calibration is not None and evidence is not None:
        diseased = evidence >= calibration.diseased_evidence
        suspect = evidence >= calibration.suspect_evidence
    else:
        diseased = max_posterior >= cfg.diseased_threshold
        suspect = max_posterior >= cfg.suspect_threshold
    if diseased:
        return STATUS_DISEASED
    if suspect:
        return STATUS_SUSPECT
    if review_fraction >= cfg.review_fraction_threshold:
        return STATUS_REVIEW
    return STATUS_HEALTHY


def estimate_incidence(
    n_flagged: int, n_judged: int, calibration: FusionCalibration | None
) -> dict[str, Any]:
    """Fração de plantas doentes do talhão, corrigida pelos erros medidos da fusão.

    A fração aparente (plantas "suspeita" ou "doente") mistura dois erros: planta doente
    que passa (sensibilidade < 1) e planta sadia sinalizada (alarme falso > 0). Com as
    duas taxas medidas na calibração, Rogan-Gladen desfaz a mistura:

        incidência = (aparente - alarme falso) / (sensibilidade - alarme falso)

    O intervalo de 95% soma a incerteza das três proporções: a do talhão e as duas taxas,
    que vêm de amostras finitas de plantas rotuladas — com o tamanho efetivo, que desconta
    a heterogeneidade entre amostras (ver ``FusionCalibration.sensitivity_n_eff``). Cada
    uma vira uma Beta com prior de Jeffreys e o intervalo sai de sorteios (semente fixa: a
    mesma resposta a cada consulta). O método delta encolhia o intervalo quando o alarme
    falso medido era 0 ou 1 em poucas dezenas de sadias.
    """
    out: dict[str, Any] = {
        "apparent": None,
        "incidence": None,
        "ci95": None,
        "method": "aparente",
        "plant_sensitivity": None,
        "plant_false_alarm": None,
    }
    if n_judged <= 0:
        return out
    apparent = n_flagged / n_judged
    out["apparent"] = apparent
    if calibration is None:
        return out
    t, f = calibration.plant_sensitivity, calibration.plant_false_alarm
    theta = min(max((apparent - f) / (t - f), 0.0), 1.0)
    rng = np.random.default_rng(INCIDENCE_SEED)

    def draws(rate: float, n: float) -> np.ndarray:
        hits = rate * n
        return rng.beta(hits + 0.5, n - hits + 0.5, INCIDENCE_DRAWS)

    a = draws(apparent, n_judged)
    t_s = draws(t, calibration.sensitivity_n_eff or calibration.n_diseased)
    f_s = draws(f, calibration.false_alarm_n_eff or calibration.n_healthy)
    span = np.maximum(t_s - f_s, 1e-6)
    sample = np.clip((a - f_s) / span, 0.0, 1.0)
    low, high = np.quantile(sample, [0.025, 0.975])
    out.update(
        incidence=theta,
        ci95=[float(min(low, theta)), float(max(high, theta))],
        method="rogan_gladen",
        plant_sensitivity=t,
        plant_false_alarm=f,
    )
    return out


def classify_change(prev: Mapping[str, Any], curr: Mapping[str, Any], cfg: FusionConfig) -> str:
    """Compara dois retratos consecutivos de uma planta."""
    if prev["status"] == STATUS_REVIEW or curr["status"] == STATUS_REVIEW:
        return CHANGE_UNDETERMINED
    a, b = _SEVERITY[prev["status"]], _SEVERITY[curr["status"]]
    if a == 0 and b > 0:
        return CHANGE_NEW_INFECTION
    if b > a:
        return CHANGE_WORSENING
    if b < a:
        return CHANGE_IMPROVEMENT
    if a > 0:
        if prev["primary_disease"] != curr["primary_disease"]:
            return CHANGE_DISEASE_SWAP
        delta = curr["incidence"] - prev["incidence"]
        if delta >= cfg.incidence_change_delta:
            return CHANGE_WORSENING
        if delta <= -cfg.incidence_change_delta:
            return CHANGE_IMPROVEMENT
    return CHANGE_STABLE


def audpc(times_days: Iterable[float], values: Iterable[float]) -> float:
    """Área sob a curva de progresso (regra trapezoidal): soma (y_i + y_i+1)/2 * dt."""
    t = np.asarray(list(times_days), dtype=float)
    y = np.asarray(list(values), dtype=float)
    if t.size < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(t)))


# ---------------------------------------------------------------------------------------------
# Estado interno (serializável)
# ---------------------------------------------------------------------------------------------


@dataclass
class _FrameEvidence:
    """Evidência de UM quadro para UMA planta: soma ponderada das LLR dos recortes."""

    camera: str
    n_tiles: int
    llr_sum: list[float]


@dataclass
class _PassEvidence:
    """Tudo o que uma passada disse sobre uma planta (contagens brutas + LLR por quadro)."""

    frames: dict[str, _FrameEvidence] = field(default_factory=dict)
    n_in_scope: int = 0
    n_out_of_scope: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    zones_seen: list[str] = field(default_factory=list)
    zones_by_class: dict[str, list[str]] = field(default_factory=dict)
    out_of_scope_by_scene: dict[str, int] = field(default_factory=dict)
    n_needs_review: int = 0
    review_refs: list[dict[str, Any]] = field(default_factory=list)
    first_time: str | None = None
    last_time: str | None = None

    def touch_time(self, ts: datetime) -> None:
        iso = ts.isoformat()
        if self.first_time is None or ts < parse_timestamp(self.first_time):
            self.first_time = iso
        if self.last_time is None or ts > parse_timestamp(self.last_time):
            self.last_time = iso

    def merge(self, other: _PassEvidence, max_refs: int) -> None:
        for key, fr in other.frames.items():
            mine = self.frames.get(key)
            if mine is None:
                self.frames[key] = _FrameEvidence(fr.camera, fr.n_tiles, list(fr.llr_sum))
            else:
                mine.n_tiles += fr.n_tiles
                mine.llr_sum = [a + b for a, b in zip(mine.llr_sum, fr.llr_sum, strict=True)]
        self.n_in_scope += other.n_in_scope
        self.n_out_of_scope += other.n_out_of_scope
        for k, v in other.class_counts.items():
            self.class_counts[k] = self.class_counts.get(k, 0) + v
        self.zones_seen = sorted(set(self.zones_seen) | set(other.zones_seen))
        for k, zs in other.zones_by_class.items():
            self.zones_by_class[k] = sorted(set(self.zones_by_class.get(k, [])) | set(zs))
        for k, v in other.out_of_scope_by_scene.items():
            self.out_of_scope_by_scene[k] = self.out_of_scope_by_scene.get(k, 0) + v
        self.n_needs_review += other.n_needs_review
        self.review_refs = (self.review_refs + other.review_refs)[:max_refs]
        for t in (other.first_time, other.last_time):
            if t is not None:
                self.touch_time(parse_timestamp(t))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> _PassEvidence:
        payload = dict(data)
        payload["frames"] = {k: _FrameEvidence(**v) for k, v in data.get("frames", {}).items()}
        return cls(**payload)


@dataclass
class _PlantState:
    plant_id: str
    row_id: str
    center_m: float
    n_center: int = 0
    frozen: bool = False
    external: bool = False
    passes: dict[str, _PassEvidence] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    geo_lat_sum: float = 0.0
    geo_lon_sum: float = 0.0
    geo_n: int = 0

    def update_center(self, x: float) -> None:
        """Média corrente enquanto a planta é provisória; congelada, não se move mais."""
        if self.frozen:
            return
        self.n_center += 1
        self.center_m += (x - self.center_m) / self.n_center

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passes"] = {k: v.to_dict() for k, v in self.passes.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> _PlantState:
        payload = dict(data)
        payload["passes"] = {
            k: _PassEvidence.from_dict(v) for k, v in data.get("passes", {}).items()
        }
        return cls(**payload)


@dataclass
class _PassInfo:
    pass_id: str
    first_time: str
    last_time: str
    closed: bool = False
    n_frames: int = 0
    # A calibração (FusionCalibration.to_dict) com que a passada foi ingerida; None = os
    # valores de partida. A LLR persistida só faz sentido com o modelo que a gerou, então a
    # passada inteira usa este, mesmo que o servidor troque de calibração no meio.
    evidence: dict[str, Any] | None = None


@dataclass
class _FieldState:
    field_id: str
    crop: str
    plants: dict[str, _PlantState] = field(default_factory=dict)
    passes: dict[str, _PassInfo] = field(default_factory=dict)
    pass_order: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    next_seq: int = 1
    unassigned_out_of_scope: dict[str, int] = field(default_factory=dict)
    plant_spacing_m: float | None = None
    """O do talhão, se o SaaS mandou; senão o padrão da cultura."""

    def resolve(self, plant_id: str) -> str:
        seen: set[str] = set()
        while plant_id in self.aliases and plant_id not in seen:
            seen.add(plant_id)
            plant_id = self.aliases[plant_id]
        return plant_id

    def row_plants(self, row_id: str) -> list[_PlantState]:
        return [p for p in self.plants.values() if p.row_id == row_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "field_id": self.field_id,
            "crop": self.crop,
            "plants": {k: v.to_dict() for k, v in self.plants.items()},
            "passes": {k: asdict(v) for k, v in self.passes.items()},
            "pass_order": list(self.pass_order),
            "aliases": dict(self.aliases),
            "next_seq": self.next_seq,
            "unassigned_out_of_scope": dict(self.unassigned_out_of_scope),
            "plant_spacing_m": self.plant_spacing_m,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> _FieldState:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"versão de esquema não suportada: {data.get('schema_version')}")
        return cls(
            field_id=data["field_id"],
            crop=data["crop"],
            plants={k: _PlantState.from_dict(v) for k, v in data["plants"].items()},
            passes={k: _PassInfo(**v) for k, v in data["passes"].items()},
            pass_order=list(data["pass_order"]),
            aliases=dict(data["aliases"]),
            next_seq=int(data["next_seq"]),
            unassigned_out_of_scope=dict(data.get("unassigned_out_of_scope", {})),
            plant_spacing_m=data.get("plant_spacing_m"),
        )


# ---------------------------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------------------------


@dataclass
class PlantSummary:
    """Diagnóstico fundido de uma planta numa passada."""

    plant_id: str
    field_id: str
    row_id: str
    crop: str
    pass_id: str
    time: str | None
    x_along_m: float
    status: str
    primary_disease: str | None
    primary_posterior: float
    posteriors: dict[str, float]
    incidence: float
    incidence_by_class: dict[str, float]
    n_tiles: int
    n_tiles_out_of_scope: int
    n_frames: int
    n_cameras: int
    cameras: list[str]
    effective_views: float
    zones_seen: list[str]
    zones_by_disease: dict[str, list[str]]
    n_needs_review: int
    needs_review_refs: list[dict[str, Any]]
    out_of_scope_by_scene: dict[str, int]
    lat: float | None = None
    lon: float | None = None
    log_odds: dict[str, float] = field(default_factory=dict)
    # Calibração da passada (FusionCalibration.id); None = valores de partida, sem medida.
    calibration_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------------------------


def _safe_filename(field_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", field_id)
    if safe != field_id or not safe:
        # Sufixo de hash evita que "a/b" e "a_b" caiam no mesmo arquivo.
        safe = f"{safe}-{hashlib.sha1(field_id.encode()).hexdigest()[:8]}"
    return f"{safe}.json"


def _row_sort_key(row_id: str) -> tuple[int, float, str]:
    try:
        return (0, float(row_id), row_id)
    except ValueError:
        return (1, 0.0, row_id)


class PlantFusionStore:
    """Estado de fusão por talhão, persistido em ``<root>/<talhão>.json``.

    Uso típico no backend: ``ingest`` a cada quadro classificado, ``close_pass`` ao fim da
    passada, ``plants``/``evolution``/``export_geojson`` nos endpoints de consulta. Não é
    thread-safe: num servidor com vários workers, serialize o acesso por talhão.
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        config: FusionConfig | None = None,
        *,
        autosave: bool = True,
        tile_log: bool = False,
    ) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or FusionConfig()
        # autosave=True grava a cada quadro (seguro contra queda, mais I/O). Em produção com
        # muitos quadros por segundo, use False e chame save()/close_pass().
        self.autosave = autosave
        # tile_log=True guarda as probabilidades de cada recorte com a planta a que ele foi
        # associado, em <root>/registro/<talhão>.jsonl: é o que a validação de campo precisa
        # para recalibrar (training.fit_fusion) com o rótulo do agrônomo por planta.
        self.tile_log = tile_log
        self._fields: dict[str, _FieldState] = {}
        self._models: dict[str, FusionCalibration] = {}

    # ------------------------------------------------------------------ persistência

    def _path(self, field_id: str) -> Path:
        return self.root / _safe_filename(field_id)

    def _load(self, field_id: str) -> _FieldState | None:
        if field_id in self._fields:
            return self._fields[field_id]
        path = self._path(field_id)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            state = _FieldState.from_dict(json.load(fh))
        self._fields[field_id] = state
        return state

    def _require(self, field_id: str) -> _FieldState:
        state = self._load(field_id)
        if state is None:
            raise KeyError(f"talhão desconhecido: {field_id!r}")
        return state

    def save(self, field_id: str | None = None) -> None:
        """Grava atomicamente: arquivo temporário no MESMO diretório + os.replace.

        Assim um leitor (ou uma queda de energia no rover/servidor) nunca vê um JSON pela
        metade: ou fica o arquivo antigo, ou o novo inteiro.
        """
        ids = [field_id] if field_id is not None else list(self._fields)
        for fid in ids:
            state = self._require(fid)
            target = self._path(fid)
            fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(state.to_dict(), fh, ensure_ascii=False, separators=(",", ":"))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise

    def fields(self) -> list[str]:
        on_disk: set[str] = set()
        for path in self.root.glob("*.json"):
            if path.name.startswith(".tmp-"):
                continue
            try:
                with path.open(encoding="utf-8") as fh:
                    on_disk.add(json.load(fh)["field_id"])
            except (OSError, ValueError, KeyError):
                continue
        return sorted(on_disk | set(self._fields))

    # ------------------------------------------------------------------ ingestão

    def ingest(
        self,
        observation: FrameObservation | Mapping[str, Any],
        calibration: FusionCalibration | None = None,
    ) -> list[str]:
        """Incorpora um quadro. Retorna os ids das plantas que receberam recortes.

        ``calibration`` vale para a passada que este quadro abre; numa passada já aberta,
        segue a calibração com que ela começou (ver ``_PassInfo.evidence``).
        """
        obs = (
            observation
            if isinstance(observation, FrameObservation)
            else FrameObservation.from_dict(observation)
        )
        cfg = self.config
        crop_cfg = cfg.crop(obs.crop)
        diseases = crop_cfg.diseases
        if calibration is not None:
            calibration.check_crop(obs.crop, diseases)
        state = self._load(obs.field_id)
        if state is None:
            state = _FieldState(field_id=obs.field_id, crop=obs.crop)
            self._fields[obs.field_id] = state
        if state.crop != obs.crop:
            raise ValueError(
                f"talhão {obs.field_id!r} é de {state.crop}, observação veio como {obs.crop}"
            )
        ts = obs.time

        info = state.passes.get(obs.pass_id)
        if info is None:
            info = _PassInfo(
                obs.pass_id,
                ts.isoformat(),
                ts.isoformat(),
                evidence=None if calibration is None else calibration.to_dict(),
            )
            state.passes[obs.pass_id] = info
            state.pass_order.append(obs.pass_id)
        if info.closed:
            raise ValueError(f"passada {obs.pass_id!r} já foi fechada")
        info.first_time = min(parse_timestamp(info.first_time), ts).isoformat()
        info.last_time = max(parse_timestamp(info.last_time), ts).isoformat()
        info.n_frames += 1

        if obs.plant_spacing_m is not None and obs.plant_spacing_m > 0:
            state.plant_spacing_m = float(obs.plant_spacing_m)
        tolerance = cfg.association_tolerance_frac * self._spacing(state, crop_cfg)
        # LLR depende só da probabilidade do recorte; os parâmetros por doença são fixos.
        params = [crop_cfg.params(d) for d in diseases]
        model = self._pass_model(info)
        touched: list[str] = []
        logged: dict[str, list[TileResult]] = {}

        for tile in obs.tiles:
            x = obs.tile_along_row_m(tile)
            in_scope = tile.counts_for_diagnosis
            plant = self._associate(state, obs, x, tolerance, create=in_scope)
            if plant is None:
                # Recorte fora do escopo longe de qualquer planta: pode ser chão, céu,
                # poste... Não cria planta, mas fica contado no talhão.
                scene = tile.scene or UNKNOWN_SCENE
                state.unassigned_out_of_scope[scene] = (
                    state.unassigned_out_of_scope.get(scene, 0) + 1
                )
                continue
            if in_scope:
                plant.update_center(x)
            ev = plant.passes.setdefault(obs.pass_id, _PassEvidence())
            ev.touch_time(ts)
            if not in_scope:
                scene = tile.scene or UNKNOWN_SCENE
                ev.n_out_of_scope += 1
                ev.out_of_scope_by_scene[scene] = ev.out_of_scope_by_scene.get(scene, 0) + 1
            else:
                self._record_tile(ev, obs, tile, diseases, params, model)
                if self.tile_log:
                    logged.setdefault(plant.plant_id, []).append(tile)
            if obs.has_geo:
                self._record_geo(plant, obs, x)
            if plant.plant_id not in touched:
                touched.append(plant.plant_id)

        if logged:
            self._log_tiles(obs, logged, model)
        if self.autosave:
            self.save(obs.field_id)
        return touched

    @staticmethod
    def _spacing(state: _FieldState, crop_cfg: CropConfig) -> float:
        return state.plant_spacing_m or crop_cfg.plant_spacing_m

    def has_frame(self, field_id: str, pass_id: str, frame_key: str) -> bool:
        """O quadro já entrou nesta passada? Ingerir duas vezes somaria a evidência duas vezes."""
        state = self._load(field_id)
        if state is None:
            return False
        return any(
            pass_id in plant.passes and frame_key in plant.passes[pass_id].frames
            for plant in state.plants.values()
        )

    def _pass_model(self, info: _PassInfo) -> FusionCalibration | None:
        """A calibração da passada, lida uma vez por id."""
        if info.evidence is None:
            return None
        key = str(info.evidence.get("id"))
        model = self._models.get(key)
        if model is None:
            model = FusionCalibration.from_dict(info.evidence)
            self._models[key] = model
        return model

    def tile_log_path(self, field_id: str) -> Path:
        return self.root / "registro" / (_safe_filename(field_id)[: -len(".json")] + ".jsonl")

    def _log_tiles(
        self,
        obs: FrameObservation,
        by_plant: Mapping[str, list[TileResult]],
        model: FusionCalibration | None,
    ) -> None:
        """Uma linha por (quadro, planta), no formato que ``training.fit_fusion`` lê.

        O id é o da planta NA INGESTÃO; se ``close_pass`` fundir a planta em outra, o
        ``fit_fusion`` resolve pelo ``aliases`` do estado do talhão.
        """
        path = self.tile_log_path(obs.field_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for plant_id, tiles in by_plant.items():
                record = {
                    "field_id": obs.field_id,
                    "pass_id": obs.pass_id,
                    "plant": plant_id,
                    "crop": obs.crop,
                    "frame": obs.frame_key,
                    "camera": obs.camera_key,
                    "camera_height_mm": obs.camera_height_mm,
                    "timestamp": obs.timestamp,
                    "calibration_id": None if model is None else model.id,
                    "tiles": [
                        {
                            "box": list(t.box),
                            "predicted_class": t.predicted_class,
                            "probabilities": t.probabilities,
                            "prediction_set": t.prediction_set,
                            "needs_review": t.needs_review,
                            "in_scope": t.in_scope,
                        }
                        for t in tiles
                    ],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _associate(
        self,
        state: _FieldState,
        obs: FrameObservation,
        x: float,
        tolerance: float,
        *,
        create: bool,
    ) -> _PlantState | None:
        """Associa um recorte a uma planta da mesma linha.

        Regra (determinística para a mesma ordem de entrada):
        1. ``plant_id`` explícito vence (planta externa, nunca fundida automaticamente).
        2. Senão, a planta de centro mais próximo, se |x - centro| <= tolerância; empate
           decidido pelo id.
        3. Senão, cria planta PROVISÓRIA (só para recortes no escopo). O centro dela é a média
           corrente dos recortes até ``close_pass``, quando é consolidada e congelada: nas
           passadas seguintes os centros não se movem, o que torna os ids estáveis.
        """
        if obs.plant_id:
            pid = state.resolve(str(obs.plant_id))
            plant = state.plants.get(pid)
            if plant is None:
                plant = _PlantState(plant_id=pid, row_id=obs.row_id, center_m=x, external=True)
                state.plants[pid] = plant
            return plant

        best: _PlantState | None = None
        best_key: tuple[float, str] | None = None
        for plant in state.row_plants(obs.row_id):
            key = (abs(plant.center_m - x), plant.plant_id)
            if key[0] <= tolerance and (best_key is None or key < best_key):
                best, best_key = plant, key
        if best is not None or not create:
            return best

        pid = f"{obs.row_id}-P{state.next_seq:04d}"
        state.next_seq += 1
        plant = _PlantState(plant_id=pid, row_id=obs.row_id, center_m=x)
        state.plants[pid] = plant
        return plant

    def _record_tile(
        self,
        ev: _PassEvidence,
        obs: FrameObservation,
        tile: TileResult,
        diseases: tuple[str, ...],
        params: list[DetectionParams],
        model: FusionCalibration | None = None,
    ) -> None:
        cfg = self.config
        weight = tile_weight(tile, cfg.review_weight)
        if model is None:
            llrs = [
                weight
                * tile_log_likelihood_ratio(tile.probabilities.get(d, 0.0), prm, cfg.llr_clip)
                for d, prm in zip(diseases, params, strict=True)
            ]
        else:
            llrs = [weight * model.tile_llr(d, tile.probabilities.get(d, 0.0)) for d in diseases]
        frame = ev.frames.get(obs.frame_key)
        if frame is None:
            frame = _FrameEvidence(obs.camera_key, 0, [0.0] * len(diseases))
            ev.frames[obs.frame_key] = frame
        frame.n_tiles += 1
        frame.llr_sum = [a + b for a, b in zip(frame.llr_sum, llrs, strict=True)]

        ev.n_in_scope += 1
        cls = tile.predicted_class
        ev.class_counts[cls] = ev.class_counts.get(cls, 0) + 1
        if tile.zones:
            ev.zones_seen = sorted(set(ev.zones_seen) | set(tile.zones))
            ev.zones_by_class[cls] = sorted(set(ev.zones_by_class.get(cls, [])) | set(tile.zones))
        if tile.needs_review:
            ev.n_needs_review += 1
            if len(ev.review_refs) < cfg.max_review_refs:
                ev.review_refs.append(
                    {"frame": obs.frame_key, "camera": obs.camera_key, "box": list(tile.box)}
                )

    def _record_geo(self, plant: _PlantState, obs: FrameObservation, x: float) -> None:
        """Estimativa geográfica da planta vista por este recorte, acumulada em média.

        À frente: x do recorte - x do rover (a lat/lon é do ponto de referência do rover).
        Ao lado: profundidade + deslocamento lateral do sensor; esquerda = direita negativa.
        Sem rumo não dá para orientar o deslocamento: usa a própria posição do rover (erro
        de ~1 m, aceitável para mapa de talhão, e documentado).
        """
        assert obs.lat is not None and obs.lon is not None
        if obs.heading_deg is None:
            lat, lon = obs.lat, obs.lon
        else:
            lateral = obs.distance_mm / 1000.0 + self.config.camera_lateral_offset_m
            right = lateral if obs.side == "right" else -lateral
            lat, lon = project_local_offset(obs.lat, obs.lon, obs.heading_deg, x - obs.x_m, right)
        plant.geo_lat_sum += lat
        plant.geo_lon_sum += lon
        plant.geo_n += 1

    # ------------------------------------------------------------------ passadas

    def close_pass(self, field_id: str, pass_id: str) -> list[PlantSummary]:
        """Fecha a passada: consolida plantas provisórias, congela centros e grava histórico.

        Consolidação: por linha, as provisórias (ordenadas por centro e id) são fundidas na
        planta já mantida mais próxima se ficaram a menos da tolerância dela (a média corrente
        pode ter aproximado dois grupos criados cedo demais, p.ex. pela borda da folhagem). O
        id absorvido vira alias do sobrevivente.
        """
        state = self._require(field_id)
        info = state.passes.get(pass_id)
        if info is None:
            raise KeyError(f"passada desconhecida: {pass_id!r}")
        if info.closed:
            raise ValueError(f"passada {pass_id!r} já foi fechada")
        crop_cfg = self.config.crop(state.crop)
        tolerance = self.config.association_tolerance_frac * self._spacing(state, crop_cfg)
        for row_id in sorted({p.row_id for p in state.plants.values()}):
            self._consolidate_row(state, row_id, tolerance)

        summaries: list[PlantSummary] = []
        for plant in sorted(state.plants.values(), key=lambda p: p.plant_id):
            if pass_id not in plant.passes:
                continue
            summary = self._summarize(state, plant, pass_id)
            plant.history = [h for h in plant.history if h["pass_id"] != pass_id]
            plant.history.append(
                {
                    "pass_id": pass_id,
                    "time": summary.time,
                    "status": summary.status,
                    "primary_disease": summary.primary_disease,
                    "primary_posterior": summary.primary_posterior,
                    "incidence": summary.incidence,
                    "n_tiles": summary.n_tiles,
                    "log_odds": summary.log_odds,
                }
            )
            plant.history.sort(key=lambda h: state.pass_order.index(h["pass_id"]))
            summaries.append(summary)
        info.closed = True
        self.save(field_id)
        return summaries

    def _consolidate_row(self, state: _FieldState, row_id: str, tolerance: float) -> None:
        plants = state.row_plants(row_id)
        kept = sorted((p for p in plants if p.frozen or p.external), key=lambda p: p.plant_id)
        provisional = sorted(
            (p for p in plants if not p.frozen and not p.external),
            key=lambda p: (p.center_m, p.plant_id),
        )
        for plant in provisional:
            target: _PlantState | None = None
            target_key: tuple[float, str] | None = None
            for other in kept:
                key = (abs(other.center_m - plant.center_m), other.plant_id)
                if key[0] <= tolerance and (target_key is None or key < target_key):
                    target, target_key = other, key
            if target is None:
                kept.append(plant)
                continue
            self._merge_plants(state, plant, target, self.config.max_review_refs)
        for plant in kept:
            plant.frozen = True

    @staticmethod
    def _merge_plants(
        state: _FieldState, src: _PlantState, dst: _PlantState, max_refs: int
    ) -> None:
        if not dst.frozen and src.n_center:
            total = dst.n_center + src.n_center
            dst.center_m = (dst.center_m * dst.n_center + src.center_m * src.n_center) / total
            dst.n_center = total
        for pid, ev in src.passes.items():
            if pid in dst.passes:
                dst.passes[pid].merge(ev, max_refs)
            else:
                dst.passes[pid] = ev
        dst.geo_lat_sum += src.geo_lat_sum
        dst.geo_lon_sum += src.geo_lon_sum
        dst.geo_n += src.geo_n
        del state.plants[src.plant_id]
        state.aliases[src.plant_id] = dst.plant_id
        for old, new in list(state.aliases.items()):
            if new == src.plant_id:
                state.aliases[old] = dst.plant_id

    # ------------------------------------------------------------------ fusão / consulta

    def _pass_log_likelihood(
        self, ev: _PassEvidence, n: int, tile_correlation: float | None = None
    ) -> tuple[np.ndarray, float]:
        """Soma das LLR com desconto de correlação hierárquico.

        quadro:  c_q = ESS(n_recortes, rho_recorte) * média das LLR dos recortes
        câmera:  c_c = ESS(n_quadros, rho_quadro) * média dos c_q dessa câmera na passada
        planta:  L   = soma dos c_c (câmeras diferentes veem folhas/alturas/lados distintos
                       e são tratadas como independentes)
        Com rho = 0 volta à soma ingênua (naive Bayes); com rho = 1, N cópias valem uma.
        Retorna também o número efetivo de "vistas" (soma dos ESS de quadros por câmera).
        ``tile_correlation`` é o rho_recorte da calibração da passada (padrão: o global).
        """
        cfg = self.config
        rho = cfg.tile_correlation if tile_correlation is None else tile_correlation
        by_camera: dict[str, list[np.ndarray]] = {}
        for key in sorted(ev.frames):
            fr = ev.frames[key]
            if fr.n_tiles == 0:
                continue
            mean = np.asarray(fr.llr_sum, dtype=float) / fr.n_tiles
            by_camera.setdefault(fr.camera, []).append(
                mean * effective_sample_size(fr.n_tiles, rho)
            )
        total = np.zeros(n, dtype=float)
        views = 0.0
        for cam in sorted(by_camera):
            contribs = np.vstack(by_camera[cam])
            ess = effective_sample_size(contribs.shape[0], cfg.frame_correlation)
            total += contribs.mean(axis=0) * ess
            views += ess
        return total, views

    def _previous_snapshot(
        self, state: _FieldState, plant: _PlantState, pass_id: str
    ) -> dict[str, Any] | None:
        order = state.pass_order.index(pass_id)
        prev = [h for h in plant.history if state.pass_order.index(h["pass_id"]) < order]
        return prev[-1] if prev else None

    def _summarize(self, state: _FieldState, plant: _PlantState, pass_id: str) -> PlantSummary:
        cfg = self.config
        crop_cfg = cfg.crop(state.crop)
        diseases = crop_cfg.diseases
        ev = plant.passes[pass_id]
        info = state.passes.get(pass_id)
        model = None if info is None else self._pass_model(info)

        llr, views = self._pass_log_likelihood(
            ev, len(diseases), None if model is None else model.tile_correlation
        )
        if model is not None:
            # Platt: a soma fundida vira log-razão de verossimilhança calibrada por planta.
            llr = model.evidence_offset + model.evidence_scale * llr
        base_lo = np.array([logit(crop_cfg.prior(d)) for d in diseases])
        prior_lo = base_lo
        prev = self._previous_snapshot(state, plant, pass_id)
        if prev is not None and cfg.temporal_carryover > 0:
            prev_lo = np.array(
                [prev["log_odds"].get(d, lo) for d, lo in zip(diseases, prior_lo, strict=True)]
            )
            prior_lo = prior_lo + cfg.temporal_carryover * (prev_lo - prior_lo)
        log_odds = prior_lo + llr
        posteriors = {d: sigmoid(float(lo)) for d, lo in zip(diseases, log_odds, strict=True)}

        if diseases:
            primary = max(diseases, key=lambda d: (posteriors[d], d))
            primary_post = posteriors[primary]
        else:
            primary, primary_post = None, 0.0

        n = ev.n_in_scope
        review_frac = ev.n_needs_review / n if n else 0.0
        status = classify_status(
            primary_post,
            n,
            review_frac,
            cfg,
            evidence=float(np.max(log_odds - base_lo)) if diseases else None,
            calibration=model,
        )
        n_diseased = sum(v for k, v in ev.class_counts.items() if k != HEALTHY)
        cameras = sorted({fr.camera for fr in ev.frames.values()})

        lat = lon = None
        if plant.geo_n:
            lat = plant.geo_lat_sum / plant.geo_n
            lon = plant.geo_lon_sum / plant.geo_n

        return PlantSummary(
            plant_id=plant.plant_id,
            field_id=state.field_id,
            row_id=plant.row_id,
            crop=state.crop,
            pass_id=pass_id,
            time=ev.first_time,
            x_along_m=plant.center_m,
            status=status,
            primary_disease=primary,
            primary_posterior=primary_post,
            posteriors=posteriors,
            incidence=n_diseased / n if n else 0.0,
            incidence_by_class={d: (ev.class_counts.get(d, 0) / n if n else 0.0) for d in diseases},
            n_tiles=n,
            n_tiles_out_of_scope=ev.n_out_of_scope,
            n_frames=len(ev.frames),
            n_cameras=len(cameras),
            cameras=cameras,
            effective_views=views,
            zones_seen=list(ev.zones_seen),
            zones_by_disease={
                d: list(zs) for d, zs in sorted(ev.zones_by_class.items()) if d != HEALTHY
            },
            n_needs_review=ev.n_needs_review,
            needs_review_refs=list(ev.review_refs),
            out_of_scope_by_scene=dict(sorted(ev.out_of_scope_by_scene.items())),
            lat=lat,
            lon=lon,
            log_odds={d: float(lo) for d, lo in zip(diseases, log_odds, strict=True)},
            calibration_id=None if model is None else model.id,
        )

    def _latest_pass(self, state: _FieldState, plant: _PlantState) -> str | None:
        for pid in reversed(state.pass_order):
            if pid in plant.passes:
                return pid
        return None

    def plants(self, field_id: str, pass_id: str | None = None) -> list[PlantSummary]:
        """Diagnóstico de cada planta: na passada pedida, ou na última em que foi vista.

        Funciona também com a passada aberta (diagnóstico ao vivo, antes da consolidação).
        """
        state = self._require(field_id)
        out: list[PlantSummary] = []
        for plant in sorted(state.plants.values(), key=lambda p: (p.row_id, p.center_m)):
            pid = pass_id if pass_id is not None else self._latest_pass(state, plant)
            if pid is None or pid not in plant.passes:
                continue
            out.append(self._summarize(state, plant, pid))
        return out

    def plant(self, field_id: str, plant_id: str, pass_id: str | None = None) -> PlantSummary:
        state = self._require(field_id)
        plant = state.plants.get(state.resolve(plant_id))
        if plant is None:
            raise KeyError(f"planta desconhecida: {plant_id!r}")
        pid = pass_id if pass_id is not None else self._latest_pass(state, plant)
        if pid is None or pid not in plant.passes:
            raise KeyError(f"planta {plant_id!r} sem observações na passada {pass_id!r}")
        return self._summarize(state, plant, pid)

    def plant_ids(self, field_id: str) -> list[str]:
        return sorted(self._require(field_id).plants)

    def resolve_plant_id(self, field_id: str, plant_id: str) -> str:
        return self._require(field_id).resolve(plant_id)

    def unassigned_out_of_scope(self, field_id: str) -> dict[str, int]:
        return dict(self._require(field_id).unassigned_out_of_scope)

    def incidence(self, field_id: str, pass_id: str | None = None) -> dict[str, Any]:
        """Incidência do talhão numa passada: a fração de PLANTAS doentes, com intervalo.

        A unidade é a planta, a do agrônomo, e não o recorte: uma planta com uma lesão
        numa folha dá dezenas de recortes sadios. Padrão: a última passada (aberta ou não;
        com o rover ainda andando, vale para as plantas já vistas). Plantas "revisar" ficam
        fora da conta, contadas à parte. Ver :func:`estimate_incidence`.
        """
        state = self._require(field_id)
        if not state.pass_order:
            raise KeyError(f"talhão {field_id!r} sem passadas")
        pid = state.pass_order[-1] if pass_id is None else pass_id
        info = state.passes.get(pid)
        if info is None:
            raise KeyError(f"passada desconhecida: {pid!r}")
        summaries = [
            self._summarize(state, plant, pid)
            for plant in sorted(state.plants.values(), key=lambda p: p.plant_id)
            if pid in plant.passes
        ]
        counts = Counter(s.status for s in summaries)
        flagged = [s for s in summaries if s.status in (STATUS_SUSPECT, STATUS_DISEASED)]
        judged = len(summaries) - counts[STATUS_REVIEW]
        model = self._pass_model(info)
        result = estimate_incidence(len(flagged), judged, model)
        if judged == 0:
            note = "Nenhuma planta julgada nesta passada ainda."
        elif model is None:
            note = (
                "Fusão sem calibração para este modelo: a fração é a aparente (plantas "
                "sinalizadas), sem descontar alarme falso nem corrigir a sensibilidade. "
                "Calibre com 'python -m training.fit_fusion'."
            )
        else:
            note = (
                f"Corrigida pela sensibilidade ({model.plant_sensitivity:.0%}) e pelo alarme "
                f"falso ({model.plant_false_alarm:.0%}) por planta medidos fora do talhão "
                f"({model.n_diseased} plantas doentes e {model.n_healthy} sadias rotuladas). "
                "Vale enquanto o talhão se parecer com essas plantas: a validação de campo "
                "deve remedir as duas taxas com o rótulo do agrônomo."
            )
        return {
            "field_id": field_id,
            "crop": state.crop,
            "pass_id": pid,
            "pass_closed": info.closed,
            "n_plants": len(summaries),
            "n_judged": judged,
            "status_counts": {
                s: counts.get(s, 0)
                for s in (STATUS_HEALTHY, STATUS_SUSPECT, STATUS_DISEASED, STATUS_REVIEW)
            },
            **result,
            "flagged_by_disease": dict(
                sorted(Counter(s.primary_disease or "?" for s in flagged).items())
            ),
            "calibration_id": None if model is None else model.id,
            "note": note,
        }

    # ------------------------------------------------------------------ evolução

    def evolution(self, field_id: str, plant_id: str) -> dict[str, Any]:
        """Histórico de passadas FECHADAS de uma planta, mudanças e AUDPC da incidência.

        O AUDPC usa só as passadas em que a planta foi vista (uma passada que não passou
        por ela não é "incidência zero"), com tempo em dias desde a primeira observação e
        o instante em que o rover passou pela planta. Unidade: incidência x dia.
        """
        state = self._require(field_id)
        plant = state.plants.get(state.resolve(plant_id))
        if plant is None:
            raise KeyError(f"planta desconhecida: {plant_id!r}")
        history = [dict(h) for h in plant.history]
        for h in history:
            h.pop("log_odds", None)
        changes: list[dict[str, Any]] = []
        for i, curr in enumerate(history):
            if i == 0:
                change = CHANGE_FIRST
                prev_pass = prev_status = None
            else:
                prev = history[i - 1]
                change = classify_change(prev, curr, self.config)
                prev_pass, prev_status = prev["pass_id"], prev["status"]
            changes.append(
                {
                    "from_pass": prev_pass,
                    "to_pass": curr["pass_id"],
                    "from_status": prev_status,
                    "to_status": curr["status"],
                    "change": change,
                }
            )
        times = [parse_timestamp(h["time"]) for h in history if h["time"] is not None]
        if len(times) == len(history) and times:
            t0 = times[0]
            days = [(t - t0).total_seconds() / 86400.0 for t in times]
            area = audpc(days, [h["incidence"] for h in history])
        else:
            area = 0.0
        return {
            "plant_id": plant.plant_id,
            "history": history,
            "changes": changes,
            "audpc": area,
            "audpc_unit": "incidencia*dia",
        }

    def changes_in_pass(self, field_id: str, pass_id: str) -> list[dict[str, Any]]:
        """Mudanças que chegaram nesta passada (útil para alertas de nova infecção)."""
        out: list[dict[str, Any]] = []
        for pid in self.plant_ids(field_id):
            evo = self.evolution(field_id, pid)
            for change in evo["changes"]:
                if change["to_pass"] == pass_id:
                    out.append({"plant_id": pid, **change})
        return out

    # ------------------------------------------------------------------ mapa

    def export_geojson(self, field_id: str, pass_id: str | None = None) -> dict[str, Any]:
        """FeatureCollection (RFC 7946) com um Point por planta.

        Se TODAS as plantas exportadas têm posição geográfica, as coordenadas são WGS84
        [lon, lat]. Caso contrário, nenhuma é (misturar sistemas no mesmo arquivo engana
        quem abrir num SIG): usamos coordenadas métricas LOCAIS [x ao longo da linha,
        ordem da linha * espaçamento entre linhas], marcadas em cada feição
        (``coordinate_system = "local_row_metric"``) e no membro ``valtexa`` da coleção.
        Atenção: coordenadas locais NÃO são lon/lat; um SIG que assuma WGS84 vai
        posicioná-las errado.
        """
        state = self._require(field_id)
        crop_cfg = self.config.crop(state.crop)
        summaries = self.plants(field_id, pass_id)
        use_geo = bool(summaries) and all(s.lat is not None for s in summaries)
        system = "WGS84" if use_geo else "local_row_metric"
        rows = sorted({p.row_id for p in state.plants.values()}, key=_row_sort_key)
        row_index = {r: i for i, r in enumerate(rows)}

        features: list[dict[str, Any]] = []
        for s in summaries:
            if use_geo:
                assert s.lat is not None and s.lon is not None
                coords = [round(s.lon, 8), round(s.lat, 8)]
            else:
                coords = [
                    round(s.x_along_m, 3),
                    round(row_index[s.row_id] * crop_cfg.row_spacing_m, 3),
                ]
            features.append(
                {
                    "type": "Feature",
                    "id": s.plant_id,
                    "geometry": {"type": "Point", "coordinates": coords},
                    "properties": {
                        "plant_id": s.plant_id,
                        "row_id": s.row_id,
                        "status": s.status,
                        "primary_disease": s.primary_disease,
                        "posterior": round(s.primary_posterior, 4),
                        "incidence": round(s.incidence, 4),
                        "n_views": s.n_frames,
                        "n_cameras": s.n_cameras,
                        "n_tiles": s.n_tiles,
                        "last_pass": s.pass_id,
                        "last_time": s.time,
                        "x_along_m": round(s.x_along_m, 3),
                        "coordinate_system": system,
                    },
                }
            )
        return {
            "type": "FeatureCollection",
            "features": features,
            "valtexa": {
                "field_id": field_id,
                "crop": state.crop,
                "pass_id": pass_id,
                "coordinate_system": system,
                "note": (
                    "coordenadas WGS84 [lon, lat]"
                    if use_geo
                    else "coordenadas LOCAIS em metros [x ao longo da linha, y entre linhas]; "
                    "não são lon/lat"
                ),
            },
        }
