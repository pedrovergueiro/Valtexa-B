"""Calibra a fusão por planta de um modelo: ``<pesos>.fusion.json``.

    # 1) Os recortes do rover, planta a planta (GPU)
    python -m training.fit_fusion collect --model banana-foliar \\
        --data-dir data/banana-foliar-campo \\
        --plantseg ~/valtexa-raw/plantseg/plantseg/images/test \\
        --out medicoes/fusao/banana-foliar.jsonl
    # No citros, só as sadias do acervo: as doentes dele são folha sobre papel (ver abaixo)
    python -m training.fit_fusion collect --model citrus-foliar \\
        --data-dir data/citrus-foliar-campo --data-dir-classes Healthy \\
        --plantseg ~/valtexa-raw/plantseg/plantseg/images/test \\
        --out medicoes/fusao/citrus-foliar.jsonl
    # 2) A calibração (CPU, menos de um minuto), gravada ao lado dos pesos
    python -m training.fit_fusion fit --model banana-foliar \\
        --frames medicoes/fusao/banana-foliar.jsonl
    # Validação de campo: o registro do rover (FIELDS_TILE_LOG=true) e o rótulo por planta.
    # Primeiro mede a calibração instalada, sem reajuste; depois recalibra com o campo.
    python -m training.fit_fusion evaluate --model banana-foliar \\
        --frames fields/plantas/registro/T01.jsonl --labels rotulos.csv --fields-dir fields
    python -m training.fit_fusion fit --model banana-foliar \\
        --frames fields/plantas/registro/T01.jsonl --labels rotulos.csv --fields-dir fields

Por que existe
--------------
A fusão (:mod:`app.plant_fusion`) soma a evidência dos recortes de uma planta. Até
02/10/2026 ela usava sensibilidade e especificidade por recorte escritas à mão e
limiares de posterior (0,5 e 0,9) sem medida, e no talhão de ponta a ponta achou 9 de
18 bananeiras doentes. O modelo põe muita probabilidade de doença em pedaço de folha
sadia (em média 45% na banana), então a média das probabilidades quase não separa as
plantas: é a cauda perto de 1 que separa. Este script mede, em plantas rotuladas, o que
a fusão precisa — ver :class:`app.plant_fusion.FusionCalibration`.

Como (fixado antes de olhar os números)
---------------------------------------
* Uma planta é um grupo de fotos, uma por altura de câmera do rover, cada foto fatiada
  e classificada como no ``/predict/rover-frame`` (sem TTA, com o conjunto da fusão).
  Fontes: o teste isolado do dataset (doentes e sadias da mesma fonte, para a calibração
  não aprender a diferença entre acervos) e, se houver, o teste do PlantSeg. Exceção
  medida em 02/10/2026: as doentes do acervo do citros são folha destacada sobre papel;
  com elas, a detecção nas fotos de campo do PlantSeg caiu de 33 para 25 de 33, e as
  tabelas ajustadas sem elas acham mais doença até no papel (106 contra 98 de 315).
  No citros entram só as sadias do acervo (``--data-dir-classes Healthy``).
* LLR por recorte: histograma da probabilidade da doença em recortes de plantas com e
  sem ela, com encolhimento (doença sem planta rotulada herda a distribuição combinada).
* Validação cruzada aninhada, estratificada por rótulo e fonte: em cada parte de treino,
  uma validação cruzada interna escolhe ``tile_correlation`` e ``llr_clip`` (maior
  detecção com o alarme falso na meta), os limiares e o Platt; a parte de fora só mede.
* Limiares pelo alarme falso nas plantas sadias: "suspeita ou doente" <= 5%, "doente"
  <= 2%. Não dependem da prevalência do talhão.
* A sensibilidade e o alarme falso por planta gravados (que corrigem a incidência do
  talhão) são os da validação cruzada externa — fora da amostra.

Limite: as plantas daqui são fotos de folha agrupadas, não o rover no pomar. Os números
valem como ponto de partida; a validação de campo deve recalibrar com o registro do
rover e o rótulo do agrônomo (subcomando ``fit`` com ``--labels``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.plant_fusion import (
    CROP_CLASSES,
    FUSION_CALIBRATION_VERSION,
    HEALTHY,
    LLR_BIN_EDGES,
    FusionCalibration,
    FusionConfig,
    TileResult,
    effective_sample_size,
    logit,
    tile_log_likelihood_ratio,
    tile_weight,
)
from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED

CAMERA_HEIGHTS_MM = (450.0, 1000.0, 1540.0)
DISTANCE_MM = 270.0
RHO_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
CLIP_GRID = (1.0, 2.0, 3.0)
MIN_HEALTHY = 200
MIN_DISEASED = 50
"""Abaixo disso o relatório avisa: com 83 sadias e sem o piso nos limiares, o alarme falso
real das metades sorteadas do citros foi de 0 a 17% para uma meta de 5%."""
SHRINK_TILES = 50.0
"""Pseudo-recortes do encolhimento: uma doença com poucos recortes rotulados fica
perto da distribuição combinada das outras em vez de seguir o ruído."""

PLANTSEG_PREFIXES = {
    "banana": {
        "banana_black_leaf_streak": "Sigatoka",
        "banana_cordana_leaf_spot": "Cordana",
        "banana_panama_disease": "Fusarium_Wilt",
    },
    "citrus": {"citrus_canker": "Canker", "citrus_greening_disease": "Greening"},
    # Fotos de campo do PlantSeg: ferrugem e olho-pardo (= cercosporiose). As de fruto
    # (berry blotch, black rot) ficam de fora: não são doenças de folha do catálogo.
    "coffee": {"coffee_leaf_rust": "Rust", "coffee_brown_eye_spot": "Cercospora_Leaf_Spot"},
}
"""Prefixos do PlantSeg com classe no catálogo. ``banana_bunchy_top`` e
``banana_anthracnose`` ficam de fora: não são classes do modelo."""


# ---------------------------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------------------------


FOLD_GROUP_PREFIX = "dobra:"
"""Planta com fonte ``dobra:N`` vai sempre para a parte ``N`` da validação cruzada.

É o que o mosaico do café (``training.fit_edge mosaico``) usa: as fotos são repartidas em
grupos sem repetição e cada planta sintética sai de um grupo só — assim "fora da amostra"
quer dizer "fotos que o ajuste não viu", e não só "planta sintética que ele não viu"."""


def fixed_folds(groups: list[str], k: int) -> np.ndarray | None:
    """As partes pelas fontes ``dobra:N``, se todas forem assim; senão ``None``."""
    if not groups or not all(g.startswith(FOLD_GROUP_PREFIX) for g in groups):
        return None
    return np.array([int(g.removeprefix(FOLD_GROUP_PREFIX)) % k for g in groups])


@dataclass
class Frame:
    """Os recortes de uma planta num quadro: faixa e probabilidade de cada doença."""

    camera: str
    probs: np.ndarray  # (n, D)
    bins: np.ndarray  # (n, D), índice da faixa em ``edges``
    weights: np.ndarray  # (n,), peso do recorte na fusão (novidade, conjunto conformal)
    n_review: int


@dataclass
class Plant:
    plant_id: str
    label: str
    group: str
    frames: list[Frame] = field(default_factory=list)

    @property
    def diseased(self) -> bool:
        return self.label != HEALTHY

    @property
    def n_in_scope(self) -> int:
        return sum(len(f.weights) for f in self.frames)

    @property
    def review_fraction(self) -> float:
        n = self.n_in_scope
        return sum(f.n_review for f in self.frames) / n if n else 0.0


def bin_index(probs: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Faixa de cada probabilidade, igual a ``FusionCalibration.tile_llr``."""
    idx = np.searchsorted(np.asarray(edges), np.clip(probs, 0.0, 1.0), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def _resolve(aliases: dict[str, str], plant_id: str) -> str:
    seen: set[str] = set()
    while plant_id in aliases and plant_id not in seen:
        seen.add(plant_id)
        plant_id = aliases[plant_id]
    return plant_id


def read_labels(path: Path) -> dict[tuple[str, str, str | None], str]:
    """CSV ``field_id,plant_id,label[,pass_id]`` — o rótulo do agrônomo por planta."""
    out: dict[tuple[str, str, str | None], str] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (
                row["field_id"].strip(),
                row["plant_id"].strip(),
                (row.get("pass_id") or "").strip() or None,
            )
            out[key] = row["label"].strip()
    return out


def load_plants(
    paths: list[Path],
    diseases: tuple[str, ...],
    *,
    edges: tuple[float, ...] = LLR_BIN_EDGES,
    labels: dict[tuple[str, str, str | None], str] | None = None,
    fields_dir: Path | None = None,
    review_weight: float = 0.5,
) -> list[Plant]:
    """Lê os JSONL do ``collect`` (com rótulo) ou do registro do rover (rótulo à parte).

    No registro, uma planta é (talhão, passada, planta): cada passada é uma observação
    dela. O id gravado é o da ingestão; ``--fields-dir`` resolve as fusões feitas no
    fechamento da passada pelo ``aliases`` do estado do talhão.
    """
    aliases: dict[str, dict[str, str]] = {}
    plants: dict[str, Plant] = {}
    unlabeled: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                field_id = rec.get("field_id")
                plant_id = str(rec["plant"])
                pass_id = rec.get("pass_id")
                if field_id is not None:
                    if field_id not in aliases:
                        aliases[field_id] = _field_aliases(fields_dir, field_id)
                    plant_id = _resolve(aliases[field_id], plant_id)
                label = rec.get("label")
                if label is None and labels is not None and field_id is not None:
                    label = labels.get((field_id, plant_id, pass_id)) or labels.get(
                        (field_id, plant_id, None)
                    )
                key = plant_id if field_id is None else f"{field_id}|{pass_id}|{plant_id}"
                if label is None:
                    unlabeled.add(key)
                    continue
                plant = plants.get(key)
                if plant is None:
                    group = rec.get("group") or f"talhao:{field_id}"
                    plant = plants[key] = Plant(key, str(label), str(group))
                tiles = [TileResult.from_dict(t) for t in rec.get("tiles") or []]
                tiles = [t for t in tiles if t.counts_for_diagnosis]
                if not tiles:
                    continue
                probs = np.array([[t.probabilities.get(d, 0.0) for d in diseases] for t in tiles])
                camera = rec.get("camera") or f"h{round(float(rec.get('camera_height_mm', 0)))}"
                plant.frames.append(
                    Frame(
                        camera=str(camera),
                        probs=probs,
                        bins=bin_index(probs, edges),
                        weights=np.array([tile_weight(t, review_weight) for t in tiles]),
                        n_review=sum(t.needs_review for t in tiles),
                    )
                )
    if unlabeled:
        print(f"  {len(unlabeled)} plantas sem rótulo ficaram de fora.")
    return sorted(plants.values(), key=lambda p: p.plant_id)


def _field_aliases(fields_dir: Path | None, field_id: str) -> dict[str, str]:
    if fields_dir is None:
        return {}
    from app.plant_fusion import _safe_filename

    state = fields_dir / "plantas" / _safe_filename(field_id)
    if not state.exists():
        return {}
    return dict(json.loads(state.read_text(encoding="utf-8")).get("aliases", {}))


# ---------------------------------------------------------------------------------------------
# Evidência (a mesma conta do PlantFusionStore)
# ---------------------------------------------------------------------------------------------


def fit_tables(
    plants: list[Plant], diseases: tuple[str, ...], n_bins: int, shrink: float = SHRINK_TILES
) -> np.ndarray:
    """LLR por faixa: log P(faixa | planta com d) - log P(faixa | planta sem d).

    Cada distribuição leva +1 por faixa (Laplace). O numerador ainda é puxado para a
    distribuição combinada da probabilidade da própria classe nas plantas doentes, com
    peso de ``shrink`` recortes: doença sem nenhum recorte rotulado fica com ela inteira.
    """
    d = len(diseases)
    pos = np.zeros((d, n_bins))
    neg = np.zeros((d, n_bins))
    for plant in plants:
        for frame in plant.frames:
            for j, disease in enumerate(diseases):
                counts = np.bincount(frame.bins[:, j], minlength=n_bins)
                if disease == plant.label:
                    pos[j] += counts
                else:
                    neg[j] += counts
    pooled = pos.sum(axis=0) + 1.0
    pooled /= pooled.sum()
    tables = np.zeros((d, n_bins))
    for j in range(d):
        num = (pos[j] + shrink * pooled) / (pos[j].sum() + shrink)
        den = (neg[j] + 1.0) / (neg[j].sum() + n_bins)
        tables[j] = np.log(num / den)
    return tables


def plant_evidence(
    plant: Plant, llr_of: Any, rho: float, frame_correlation: float, n_diseases: int
) -> np.ndarray:
    """L_d da planta, como ``PlantFusionStore._pass_log_likelihood``.

    ``llr_of(frame)`` devolve a LLR já limitada de cada recorte, (n, D).
    """
    by_camera: dict[str, list[np.ndarray]] = defaultdict(list)
    for frame in plant.frames:
        llr = frame.weights[:, None] * llr_of(frame)
        n = len(frame.weights)
        by_camera[frame.camera].append(llr.mean(axis=0) * effective_sample_size(n, rho))
    total = np.zeros(n_diseases)
    for camera in sorted(by_camera):
        contribs = np.vstack(by_camera[camera])
        total += contribs.mean(axis=0) * effective_sample_size(len(contribs), frame_correlation)
    return total


def table_llr(tables: np.ndarray, clip: float):
    cols = np.arange(tables.shape[0])[None, :]
    clipped = np.clip(tables, -clip, clip)
    return lambda frame: clipped[cols, frame.bins]


def evidence_matrix(plants, tables, clip, rho, frame_correlation) -> np.ndarray:
    llr_of = table_llr(tables, clip)
    return np.array(
        [plant_evidence(p, llr_of, rho, frame_correlation, tables.shape[0]) for p in plants]
    ).reshape(len(plants), tables.shape[0])


# ---------------------------------------------------------------------------------------------
# Status, limiares, Platt
# ---------------------------------------------------------------------------------------------


def statuses(
    evidence: np.ndarray,
    plants: list[Plant],
    t_suspect: float,
    t_diseased: float,
    review_threshold: float,
) -> list[str]:
    """Como ``classify_status``: achado vence; sem achado, novidade demais -> revisar."""
    out = []
    for row, plant in zip(evidence, plants, strict=True):
        best = float(row.max()) if row.size else -math.inf
        if plant.n_in_scope == 0:
            out.append("revisar")
        elif best >= t_diseased:
            out.append("doente")
        elif best >= t_suspect:
            out.append("suspeita")
        elif plant.review_fraction >= review_threshold:
            out.append("revisar")
        else:
            out.append("sadia")
    return out


def fa_thresholds(healthy_best: np.ndarray, fa_suspect: float, fa_diseased: float):
    """Menores limiares com alarme falso <= meta nas sadias (k = piso(meta * n) passam)."""
    values = np.sort(np.asarray(healthy_best, dtype=float))
    n = len(values)
    if n == 0:
        raise SystemExit("A calibração precisa de plantas sadias.")

    def at(rate: float) -> float:
        k = int(math.floor(rate * n))
        return float(np.nextafter(values[n - k - 1], math.inf))

    t_s = at(fa_suspect)
    return t_s, max(at(fa_diseased), t_s)


def floored_thresholds(
    thresholds: tuple[float, float], offset: float, scale: float, cfg: Settings
) -> tuple[float, float]:
    """Limiares do alarme falso, com piso no que o status quer dizer.

    "suspeita" exige evidência a favor da doença (razão de verossimilhança calibrada >= 1)
    e "doente", posterior de pelo menos 50% no prior da cultura. Sem os pisos, a meta de 5%
    num conjunto de sadias muito negativo punha o limiar abaixo de zero: em 02/10/2026, o
    citros marcava "suspeita" planta com 0,1% de posterior na demonstração do SaaS.
    """
    t_s, t_d = thresholds
    t_s = max(t_s, (0.0 - offset) / scale)
    t_d = max(t_d, (cfg.diseased_floor - offset) / scale, t_s)
    return t_s, t_d


def platt(x: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> tuple[float, float]:
    """Regressão logística y ~ a + b x (Newton), devolvida como LLR: a - logit(ȳ) + b x.

    Tirar o logit da prevalência da amostra deixa só a razão de verossimilhança: a
    amostra de calibração tem a proporção que tem (poucas doentes de cada doença), e o
    prior certo entra depois, o do talhão.
    """
    design = np.c_[np.ones_like(x), x]
    w = np.zeros(2)
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(design @ w)))
        grad = design.T @ (p - y) + l2 * w
        hess = design.T @ (design * (p * (1 - p))[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    rate = float(np.clip(y.mean(), 1e-9, 1 - 1e-9))
    return float(w[0] - logit(rate)), float(w[1])


def stratified_folds(plants: list[Plant], k: int, seed: int) -> np.ndarray:
    """Parte de cada planta, estratificada por (rótulo, fonte) e sorteada com semente.

    Fontes ``dobra:N`` (o mosaico do café) fixam a parte — ver :data:`FOLD_GROUP_PREFIX`.
    """
    fixas = fixed_folds([p.group for p in plants], k)
    if fixas is not None:
        return fixas
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, p in enumerate(plants):
        strata[(p.label, p.group)].append(i)
    fold = np.zeros(len(plants), dtype=int)
    offset = 0
    for key in sorted(strata):
        idx = strata[key]
        for position, i in enumerate(rng.permutation(idx)):
            fold[i] = (position + offset) % k
        offset += len(idx)
    return fold


@dataclass
class Procedure:
    """O que a calibração escolhe a partir de um conjunto de plantas rotuladas."""

    tables: np.ndarray
    rho: float
    clip: float
    t_suspect: float
    t_diseased: float
    offset: float
    scale: float


@dataclass
class Settings:
    diseases: tuple[str, ...]
    n_bins: int
    fa_suspect: float = 0.05
    fa_diseased: float = 0.02
    inner_folds: int = 4
    seed: int = 0
    frame_correlation: float = FusionConfig().frame_correlation
    review_threshold: float = FusionConfig().review_fraction_threshold
    prior: float = 0.05  # o default_prior da cultura; fit() passa o da configuração

    @property
    def diseased_floor(self) -> float:
        """Evidência calibrada mínima de "doente": posterior >= 50% no prior da cultura."""
        return logit(0.5) - logit(self.prior)


def detection(status: list[str], plants: list[Plant]) -> tuple[int, int, int]:
    det = sum(
        s in ("suspeita", "doente") for s, p in zip(status, plants, strict=True) if p.diseased
    )
    det_d = sum(s == "doente" for s, p in zip(status, plants, strict=True) if p.diseased)
    return det, det_d, sum(p.diseased for p in plants)


def fit_procedure(plants: list[Plant], cfg: Settings) -> Procedure:
    """Validação cruzada interna -> (rho, clip), limiares, Platt; tabelas com tudo."""
    k = max(2, min(cfg.inner_folds, min(Counter(p.diseased for p in plants).values())))
    fold = stratified_folds(plants, k, cfg.seed)
    tables = [
        fit_tables(
            [p for p, f in zip(plants, fold, strict=True) if f != i], cfg.diseases, cfg.n_bins
        )
        for i in range(k)
    ]
    healthy = np.array([not p.diseased for p in plants])
    # Platt nas evidências fora da amostra: uma amostra por (planta, doença).
    labels = np.array([[float(p.label == d) for d in cfg.diseases] for p in plants]).ravel()
    best = None
    for clip in CLIP_GRID:
        for rho in RHO_GRID:
            evidence = np.zeros((len(plants), len(cfg.diseases)))
            for i in range(k):
                rows = np.flatnonzero(fold == i)
                evidence[rows] = evidence_matrix(
                    [plants[r] for r in rows], tables[i], clip, rho, cfg.frame_correlation
                )
            offset, scale = platt(evidence.ravel(), labels)
            if not scale > 0:
                continue
            t_s, t_d = floored_thresholds(
                fa_thresholds(evidence[healthy].max(axis=1), cfg.fa_suspect, cfg.fa_diseased),
                offset,
                scale,
                cfg,
            )
            status = statuses(evidence, plants, t_s, t_d, cfg.review_threshold)
            det, det_d, _ = detection(status, plants)
            right = sum(
                s in ("suspeita", "doente") and cfg.diseases[int(np.argmax(e))] == p.label
                for s, e, p in zip(status, evidence, plants, strict=True)
                if p.diseased
            )
            key = (det, det_d, right, rho)
            if best is None or key > best[0]:
                best = (key, rho, clip, t_s, t_d, offset, scale)
    if best is None:
        raise SystemExit("Platt com escala <= 0 na grade toda: a evidência não separa as plantas.")
    _, rho, clip, t_s, t_d, offset, scale = best
    return Procedure(
        tables=fit_tables(plants, cfg.diseases, cfg.n_bins),
        rho=rho,
        clip=clip,
        t_suspect=t_s,
        t_diseased=t_d,
        offset=offset,
        scale=scale,
    )


def apply(proc: Procedure, plants: list[Plant], cfg: Settings) -> tuple[np.ndarray, list[str]]:
    evidence = evidence_matrix(plants, proc.tables, proc.clip, proc.rho, cfg.frame_correlation)
    return evidence, statuses(
        evidence, plants, proc.t_suspect, proc.t_diseased, cfg.review_threshold
    )


# ---------------------------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------------------------


def summarize(
    plants: list[Plant], evidence: np.ndarray, status: list[str], diseases: tuple[str, ...]
) -> dict[str, Any]:
    """Detecção, alarme falso e acerto da doença, no total, por classe e por fonte."""
    flagged = [s in ("suspeita", "doente") for s in status]
    out: dict[str, Any] = {}
    sick = [i for i, p in enumerate(plants) if p.diseased]
    well = [i for i, p in enumerate(plants) if not p.diseased]
    out["doentes"] = len(sick)
    out["sadias"] = len(well)
    out["detectadas"] = sum(flagged[i] for i in sick)
    out["detectadas_como_doente"] = sum(status[i] == "doente" for i in sick)
    out["doenca_certa"] = sum(
        flagged[i] and diseases[int(np.argmax(evidence[i]))] == plants[i].label for i in sick
    )
    out["alarme_falso"] = sum(flagged[i] for i in well)
    out["alarme_falso_doente"] = sum(status[i] == "doente" for i in well)
    out["revisar"] = sum(s == "revisar" for s in status)
    by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, p in enumerate(plants):
        target = by_class[p.label]
        target[0] += flagged[i]
        target[1] += 1
        group = by_group[f"{'doentes' if p.diseased else 'sadias'}:{p.group}"]
        group[0] += flagged[i]
        group[1] += 1
    out["sinalizadas_por_classe"] = {k: f"{a}/{b}" for k, (a, b) in sorted(by_class.items())}
    out["sinalizadas_por_fonte"] = {k: f"{a}/{b}" for k, (a, b) in sorted(by_group.items())}
    return out


def baseline(plants: list[Plant], crop: str, cfg: Settings) -> dict[str, Any]:
    """A fusão sem calibração (valores de partida do app), nas mesmas plantas."""
    fusion = FusionConfig()
    crop_cfg = fusion.crop(crop)
    params = [crop_cfg.params(d) for d in cfg.diseases]

    def llr_of(frame: Frame) -> np.ndarray:
        return np.array(
            [
                [
                    tile_log_likelihood_ratio(float(p), prm, fusion.llr_clip)
                    for p, prm in zip(row, params, strict=True)
                ]
                for row in frame.probs
            ]
        ).reshape(len(frame.probs), len(params))

    evidence = np.array(
        [
            plant_evidence(
                p, llr_of, fusion.tile_correlation, fusion.frame_correlation, len(params)
            )
            for p in plants
        ]
    ).reshape(len(plants), len(params))
    prior = logit(crop_cfg.default_prior)
    # Limiar de posterior vira limiar de evidência: posterior >= t <=> L >= logit(t) - logit(prior).
    t_s = logit(fusion.suspect_threshold) - prior
    t_d = logit(fusion.diseased_threshold) - prior
    return summarize(
        plants, evidence, statuses(evidence, plants, t_s, t_d, cfg.review_threshold), cfg.diseases
    )


def effective_size(flags_by_fold: list[list[bool]]) -> tuple[float, float]:
    """(taxa, n efetivo) de uma taxa medida nas K partes da validação cruzada.

    Se as plantas fossem independentes, a taxa de cada parte variaria como binomial. Varia
    mais: plantas da mesma fonte se parecem (mesmo acervo, câmera, estágio da doença), e num
    talhão novo a taxa muda do mesmo jeito. Medido em 02/10/2026 na banana: duas metades de
    62 doentes deram 85% e 97% de detecção, longe do que a binomial explica. A dispersão
    entre as partes (qui-quadrado de Pearson sobre os graus de liberdade) é o efeito de
    desenho; o intervalo da incidência usa n / efeito em vez de n.
    """
    n_k = np.array([len(f) for f in flags_by_fold if f], dtype=float)
    x_k = np.array([sum(f) for f in flags_by_fold if f], dtype=float)
    n = float(n_k.sum())
    rate = float(x_k.sum() / n) if n else 0.0
    if len(n_k) < 2 or rate in (0.0, 1.0):
        return rate, n
    chi2 = float(np.sum((x_k - n_k * rate) ** 2 / (n_k * rate * (1.0 - rate))))
    return rate, n / max(1.0, chi2 / (len(n_k) - 1))


def calibrate(
    plants: list[Plant], crop: str, model_key: str, cfg: Settings, outer_folds: int
) -> tuple[FusionCalibration, dict[str, Any]]:
    """Validação cruzada externa (os números) + procedimento em tudo (o arquivo)."""
    counts = Counter(p.diseased for p in plants)
    if counts[True] < 2 or counts[False] < 2:
        raise SystemExit(
            f"Poucas plantas rotuladas: {counts[True]} doentes, {counts[False]} sadias."
        )
    k = max(2, min(outer_folds, counts[True], counts[False]))
    fold = stratified_folds(plants, k, cfg.seed + 1)
    evidence = np.zeros((len(plants), len(cfg.diseases)))
    status: list[str] = [""] * len(plants)
    chosen = []
    for i in range(k):
        train = [p for p, f in zip(plants, fold, strict=True) if f != i]
        rows = np.flatnonzero(fold == i)
        proc = fit_procedure(train, cfg)
        chosen.append({"tile_correlation": proc.rho, "llr_clip": proc.clip})
        ev, st = apply(proc, [plants[r] for r in rows], cfg)
        evidence[rows] = ev
        for r, s in zip(rows, st, strict=True):
            status[r] = s
    cv = summarize(plants, evidence, status, cfg.diseases)

    final = fit_procedure(plants, cfg)
    flagged = [s in ("suspeita", "doente") for s in status]
    sick_by_fold = [
        [flagged[r] for r in np.flatnonzero(fold == i) if plants[r].diseased] for i in range(k)
    ]
    well_by_fold = [
        [flagged[r] for r in np.flatnonzero(fold == i) if not plants[r].diseased] for i in range(k)
    ]
    sensitivity, sensitivity_n_eff = effective_size(sick_by_fold)
    false_alarm, false_alarm_n_eff = effective_size(well_by_fold)
    if not false_alarm < sensitivity:
        raise SystemExit(
            f"Fora da amostra, a fusão sinaliza {false_alarm:.0%} das sadias e "
            f"{sensitivity:.0%} das doentes: não separa as plantas."
        )
    edges = tuple(float(e) for e in LLR_BIN_EDGES)
    body = {
        "model_key": model_key,
        "crop": crop,
        "bin_edges": list(edges),
        "llr_tables": {
            d: [round(float(v), 5) for v in final.tables[j]] for j, d in enumerate(cfg.diseases)
        },
        "llr_clip": final.clip,
        "tile_correlation": final.rho,
        "evidence_offset": final.offset,
        "evidence_scale": final.scale,
        # max(0, ...): com o piso ativo, a volta para a escala calibrada dá -1e-16.
        "suspect_evidence": max(0.0, final.offset + final.scale * final.t_suspect),
        "diseased_evidence": max(0.0, final.offset + final.scale * final.t_diseased),
        "plant_sensitivity": sensitivity,
        "plant_false_alarm": false_alarm,
        "n_diseased": cv["doentes"],
        "n_healthy": cv["sadias"],
        "sensitivity_n_eff": round(sensitivity_n_eff, 2),
        "false_alarm_n_eff": round(false_alarm_n_eff, 2),
    }
    digest = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:8]
    calibration = FusionCalibration.from_dict(
        {**body, "id": f"{model_key}@{datetime.now(UTC):%Y%m%d}-{digest}"}
    )
    warnings = []
    if counts[False] < MIN_HEALTHY:
        warnings.append(
            f"Só {counts[False]} plantas sadias: o limiar de alarme falso e a taxa medida ficam "
            "instáveis (no citros, com 83 e sem o piso nos limiares, o alarme falso real variou "
            "de 0 a 17% entre metades sorteadas). A incidência corrigida é indicação até juntar "
            "mais sadias."
        )
    if counts[True] < MIN_DISEASED:
        warnings.append(
            f"Só {counts[True]} plantas doentes: a sensibilidade por planta e as tabelas das "
            "doenças sem planta rotulada (que herdam a distribuição das outras) são frágeis."
        )
    report = {
        "avisos": warnings,
        "validacao_cruzada": {"partes": k, **cv, "escolhas_por_parte": chosen},
        "sem_calibracao": baseline(plants, crop, cfg),
        "metas_de_alarme_falso": {"suspeita_ou_doente": cfg.fa_suspect, "doente": cfg.fa_diseased},
        "limiares_na_evidencia_crua": {"suspeita": final.t_suspect, "doente": final.t_diseased},
        "plantas_por_classe": dict(sorted(Counter(p.label for p in plants).items())),
        "plantas_por_fonte": dict(sorted(Counter(p.group for p in plants).items())),
    }
    return calibration, report


# ---------------------------------------------------------------------------------------------
# collect: as plantas pela API do rover
# ---------------------------------------------------------------------------------------------


def _as_frame(path: Path, min_width: int = 1024) -> bytes:
    """Foto estreita demais para o recorte de 150 mm é ampliada, como no e2e do talhão."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width >= min_width:
            return path.read_bytes()
        scale = min_width / image.width
        buffer = io.BytesIO()
        image.resize((min_width, round(image.height * scale)), Image.Resampling.LANCZOS).save(
            buffer, "JPEG", quality=95
        )
        return buffer.getvalue()


def build_plants(
    args: argparse.Namespace, class_names: tuple[str, ...], crop: str
) -> list[tuple[str, str, str, list[Path]]]:
    """(id, rótulo, fonte, fotos): trios de fotos do teste isolado e do PlantSeg."""
    from PIL import Image

    from training.dataset import index_by_class, split_indices
    from training.prepare_dataset import looks_like_label_mask

    per_plant = len(args.camera_heights)
    plants: list[tuple[str, str, str, list[Path]]] = []
    paths, labels, names = index_by_class(args.data_dir)
    if tuple(names) != tuple(class_names):
        raise SystemExit(f"Classes de {args.data_dir} ({', '.join(names)}) diferem das do modelo.")
    _, _, test_idx = split_indices(len(paths), args.seed, args.validation_split, args.test_split)
    by_class: dict[str, list[Path]] = defaultdict(list)
    for i in test_idx:
        path = Path(paths[i])
        if not looks_like_label_mask(path):
            by_class[names[labels[i]]].append(path)
    wanted = set(args.data_dir_classes or names)
    for name in sorted(by_class):
        if name not in wanted:
            continue
        photos = by_class[name]
        cap = args.max_healthy if name == HEALTHY else args.max_plants
        for k in range(min(cap, len(photos) // per_plant)):
            plants.append(
                (
                    f"campo-{name}-{k:03d}",
                    name,
                    "campo",
                    photos[k * per_plant : (k + 1) * per_plant],
                )
            )
    if args.plantseg is not None:
        for prefix, label in sorted(PLANTSEG_PREFIXES.get(crop, {}).items()):
            photos = []
            for path in sorted(args.plantseg.glob(f"{prefix}*")):
                with Image.open(path) as image:
                    if image.size[0] / image.size[1] < 2.0:  # panorama não é quadro de rover
                        photos.append(path)
            for k in range(len(photos) // per_plant):
                plants.append(
                    (
                        f"plantseg-{label}-{k:03d}",
                        label,
                        "plantseg",
                        photos[k * per_plant : (k + 1) * per_plant],
                    )
                )
    return plants


def collect(args: argparse.Namespace) -> None:
    from app.framing import framing_model
    from app.inference.preprocess import load_image_from_bytes
    from app.model import CornDiseaseModel
    from app.rover import plan_tiles
    from app.rover_api import _predict_in_chunks

    spec = get_spec(args.model)
    crop = spec.crop
    if crop not in CROP_CLASSES:
        raise SystemExit(f"{spec.key} não é de uma cultura do rover ({', '.join(CROP_CLASSES)}).")
    args.data_dir = args.data_dir or spec.data_dir
    model = CornDiseaseModel(spec)
    model.load()
    if not model.is_loaded:
        raise SystemExit(f"Não foi possível carregar os pesos de {spec.weights_path}.")
    if spec.head_path.exists() or spec.species_path.exists():
        framing_model.load()
    plants = build_plants(args, tuple(model.class_names), crop)
    print(
        f"{spec.key}: {len(plants)} plantas — "
        + ", ".join(
            f"{k} {v}" for k, v in sorted(Counter(f"{g}:{lab}" for _, lab, g, _ in plants).items())
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".parcial")
    started = time.perf_counter()
    with partial.open("w", encoding="utf-8") as fh:
        for n, (plant_id, label, group, photos) in enumerate(plants):
            for photo, height in zip(photos, args.camera_heights, strict=True):
                image = load_image_from_bytes(_as_frame(photo), photo.name)
                base = {
                    "crop": crop,
                    "model_key": spec.key,
                    "plant": plant_id,
                    "label": label,
                    "group": group,
                    "photo": photo.name,
                    "camera": f"h{round(height)}",
                    "camera_height_mm": height,
                    "image_width_px": image.size[0],
                }
                try:
                    tiles = plan_tiles(image.size, args.distance_mm, height, crop=crop)
                except ValueError as exc:  # a API responde 422 a esse quadro
                    fh.write(
                        json.dumps({**base, "refused": str(exc), "tiles": []}, ensure_ascii=False)
                        + "\n"
                    )
                    continue
                crops = [image.crop(t.box) for t in tiles]
                results = _predict_in_chunks(
                    model, [f"r{i}" for i in range(len(tiles))], crops, False
                )
                fusion_sets = model.fusion_prediction_sets(results, False)
                fh.write(
                    json.dumps(
                        {
                            **base,
                            "tiles": [
                                {
                                    "box": list(t.box),
                                    "predicted_class": r.predicted_class,
                                    "probabilities": {
                                        p.class_name: p.probability for p in r.probabilities
                                    },
                                    "confidence": r.confidence,
                                    "height_mm": t.center_height_mm,
                                    "zones": list(t.zones),
                                    "prediction_set": fset,
                                    "needs_review": r.needs_review,
                                    "in_scope": None if r.framing is None else r.framing.in_scope,
                                    "scene": None if r.framing is None else r.framing.scene,
                                }
                                for t, r, fset in zip(tiles, results, fusion_sets, strict=True)
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if n % 25 == 0:
                print(
                    f"  {n + 1}/{len(plants)} plantas em {time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    partial.replace(args.out)
    print(f"Gravado em {args.out}")


# ---------------------------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------------------------


def fit(args: argparse.Namespace) -> None:
    spec = get_spec(args.model)
    crop = spec.crop
    if crop not in CROP_CLASSES:
        raise SystemExit(f"{spec.key} não é de uma cultura do rover ({', '.join(CROP_CLASSES)}).")
    diseases = FusionConfig().crop(crop).diseases
    labels = read_labels(args.labels) if args.labels else None
    plants = load_plants(args.frames, diseases, labels=labels, fields_dir=args.fields_dir)
    cfg = Settings(
        diseases=diseases,
        n_bins=len(LLR_BIN_EDGES) - 1,
        fa_suspect=args.fa_suspect,
        fa_diseased=args.fa_diseased,
        seed=args.seed,
        prior=FusionConfig().crop(crop).default_prior,
    )
    print(
        f"{spec.key}: {sum(p.diseased for p in plants)} plantas doentes, "
        f"{sum(not p.diseased for p in plants)} sadias"
    )
    calibration, report = calibrate(plants, crop, spec.key, cfg, args.folds)
    report["dados"] = [str(p) for p in args.frames]
    out = args.out or spec.fusion_path
    out.write_text(
        json.dumps(
            {
                "version": FUSION_CALIBRATION_VERSION,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "calibration": calibration.to_dict(),
                "report": report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cv, base = report["validacao_cruzada"], report["sem_calibracao"]
    print(f"\nFora da amostra ({cv['partes']} partes), contra a fusão sem calibração:")
    for name, row in (("calibrada", cv), ("sem calibração", base)):
        print(
            f"  {name:>15}: detecta {row['detectadas']}/{row['doentes']} doentes "
            f"({row['detectadas_como_doente']} como 'doente', "
            f"{row['doenca_certa']} com a doença certa); "
            f"alarme falso {row['alarme_falso']}/{row['sadias']} sadias "
            f"({row['alarme_falso_doente']} como 'doente')"
        )
    print(f"  por classe: {cv['sinalizadas_por_classe']}")
    print(f"  por fonte:  {cv['sinalizadas_por_fonte']}")
    print(
        f"Calibração {calibration.id}: rho {calibration.tile_correlation}, clip "
        f"{calibration.llr_clip}, Platt {calibration.evidence_offset:+.2f} + "
        f"{calibration.evidence_scale:.2f} L"
    )
    for warning in report["avisos"]:
        print(f"AVISO: {warning}")
    print(f"Gravado em {out}")


def procedure_of(calibration: FusionCalibration, diseases: tuple[str, ...]) -> Procedure:
    """A calibração gravada de volta na forma que ``apply`` usa (limiares na L crua)."""
    return Procedure(
        tables=np.array([calibration.llr_tables[d] for d in diseases]),
        rho=calibration.tile_correlation,
        clip=calibration.llr_clip,
        t_suspect=(calibration.suspect_evidence - calibration.evidence_offset)
        / calibration.evidence_scale,
        t_diseased=(calibration.diseased_evidence - calibration.evidence_offset)
        / calibration.evidence_scale,
        offset=calibration.evidence_offset,
        scale=calibration.evidence_scale,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """A calibração instalada, SEM reajuste, em plantas rotuladas — a validação externa.

    É a pergunta da validação de campo: o que foi calibrado em fotos agrupadas vale
    para o rover no pomar? Mede detecção, alarme falso e a incidência estimada contra
    a real das plantas rotuladas (fileiras inteiras, para não enviesar).
    """
    from app.plant_fusion import estimate_incidence, load_fusion_calibration

    spec = get_spec(args.model)
    diseases = FusionConfig().crop(spec.crop).diseases
    calibration = load_fusion_calibration(args.calibration or spec.fusion_path)
    calibration.check_crop(spec.crop, diseases)
    labels = read_labels(args.labels) if args.labels else None
    plants = load_plants(args.frames, diseases, labels=labels, fields_dir=args.fields_dir)
    cfg = Settings(diseases=diseases, n_bins=len(calibration.bin_edges) - 1)
    evidence, status = apply(procedure_of(calibration, diseases), plants, cfg)
    result = summarize(plants, evidence, status, diseases)
    judged = [s for s in status if s != "revisar"]
    flagged = sum(s in ("suspeita", "doente") for s in judged)
    truth = [p.diseased for p, s in zip(plants, status, strict=True) if s != "revisar"]
    result["incidencia"] = {
        "real": sum(truth) / len(truth) if truth else None,
        **estimate_incidence(flagged, len(judged), calibration),
    }
    result["calibracao"] = calibration.id
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibração da fusão por planta")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="Recortes do rover planta a planta (GPU).")
    c.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    c.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    c.add_argument("--plantseg", type=Path, default=None, help="Imagens de teste do PlantSeg.")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument(
        "--data-dir-classes",
        nargs="+",
        default=None,
        help="Só estas classes do --data-dir (padrão: todas). No citros, só Healthy: as "
        "doentes do acervo são folha destacada sobre papel, que o rover não vê.",
    )
    c.add_argument("--max-plants", type=int, default=40, help="Plantas doentes por classe.")
    c.add_argument("--max-healthy", type=int, default=300, help="Plantas sadias.")
    c.add_argument("--camera-heights", type=float, nargs="+", default=list(CAMERA_HEIGHTS_MM))
    c.add_argument("--distance-mm", type=float, default=DISTANCE_MM)
    c.add_argument("--seed", type=int, default=DEFAULT_SEED)
    c.add_argument("--validation-split", type=float, default=0.15)
    c.add_argument("--test-split", type=float, default=0.15)

    f = sub.add_parser("fit", help="Calibra e grava <pesos>.fusion.json (CPU).")
    f.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    f.add_argument("--frames", type=Path, nargs="+", required=True)
    f.add_argument(
        "--labels", type=Path, default=None, help="CSV field_id,plant_id,label[,pass_id]."
    )
    f.add_argument("--fields-dir", type=Path, default=None, help="FIELDS_DIR, para os aliases.")
    f.add_argument("--out", type=Path, default=None, help="Padrão: ao lado dos pesos.")
    f.add_argument("--fa-suspect", type=float, default=0.05)
    f.add_argument("--fa-diseased", type=float, default=0.02)
    f.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Partes da validação cruzada externa. Mais partes, taxas menos pessimistas: "
        "cada parte de treino fica mais perto do conjunto inteiro que a calibração final usa.",
    )
    f.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("evaluate", help="Mede a calibração instalada, sem reajuste.")
    e.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    e.add_argument("--frames", type=Path, nargs="+", required=True)
    e.add_argument(
        "--labels", type=Path, default=None, help="CSV field_id,plant_id,label[,pass_id]."
    )
    e.add_argument("--fields-dir", type=Path, default=None, help="FIELDS_DIR, para os aliases.")
    e.add_argument("--calibration", type=Path, default=None, help="Padrão: a dos pesos.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "collect":
        collect(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        fit(args)


if __name__ == "__main__":
    main(sys.argv[1:])
