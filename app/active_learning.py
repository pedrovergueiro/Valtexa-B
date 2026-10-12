"""Aprendizado ativo para a fila de revisão do VALTEXA-B.

Problema
--------
O rover e a API geram milhares de recortes por dia, mas o agrônomo só consegue rotular algumas
dezenas. Ordenar a fila por data desperdiça esse tempo com folhas que o modelo já acerta com
folga. Este módulo escolhe, sob um orçamento, os itens cujo rótulo mais ensina ao próximo treino.

Ideias usadas (com referências)
-------------------------------
* Incerteza: entropia normalizada e margem entre as duas maiores probabilidades
  (Settles, "Active Learning Literature Survey", 2009; Scheffer et al., 2001 para a margem) e o
  tamanho do conjunto de predição conformal (Angelopoulos & Bates, "A Gentle Introduction to
  Conformal Prediction", 2021): conjunto vazio = nenhuma classe plausível, conjunto grande =
  ambíguo.
* Novidade: ``novelty_score`` (distância de Mahalanobis normalizada, Lee et al., NeurIPS 2018);
  acima de 1 o item está fora da distribuição de treino.
* Raridade: inverso da frequência da classe prevista entre os já rotulados do mesmo modelo, com
  suavização de Laplace, para que classes raras (ex.: Cordana) não fiquem eternamente sub-
  representadas.
* Lote diverso: BADGE (Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient
  Lower Bounds", ICLR 2020) e core-set guloso k-center (Sener & Savarese, "Active Learning for
  Convolutional Neural Networks: A Core-Set Approach", ICLR 2018).
* Itens com ``in_scope`` False (o portão de enquadramento diz que não é folha) vão para uma
  trilha separada (:func:`select_out_of_scope`) e nunca competem com as folhas no lote principal.

Só depende de stdlib + numpy, para poder ser encaixado no backend FastAPI sem trazer sklearn.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

Strategy = Literal["badge", "coreset", "uncertainty", "random"]
STRATEGIES: tuple[str, ...] = ("badge", "coreset", "uncertainty", "random")

TRACK_MAIN = "principal"
TRACK_OUT_OF_SCOPE = "fora_de_escopo"

_EPS = 1e-12


# ---------------------------------------------------------------------------------------------
# Modelos de dados
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewItem:
    """Um item da fila de revisão (pendente) ou do conjunto já rotulado.

    Espelha o registro do backend. ``label`` só existe nos itens rotulados. O ``embedding`` é o
    vetor da penúltima camada (ex.: 1280 dimensões no MobileNetV2/EfficientNet) e é opcional:
    sem ele, BADGE e core-set caem para o ranking por pontuação.
    """

    id: str
    model_key: str
    crop: str = ""
    predicted_class: str = ""
    confidence: float = 0.0
    probabilities: Mapping[str, float] = field(default_factory=dict)
    novelty_score: float | None = None
    prediction_set: Sequence[str] | None = None
    needs_review: bool = False
    in_scope: bool | None = None
    created_at: str = ""
    embedding: np.ndarray | None = None
    field_id: str | None = None
    plant_id: str | None = None
    camera: str | None = None
    label: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReviewItem:
        """Converte o dicionário vindo do banco/JSON.

        Aceita os metadados tanto no nível raiz quanto aninhados em ``metadata``/``metadados``,
        porque a API e o rover gravam em formatos ligeiramente diferentes.
        """
        meta: Mapping[str, Any] = data.get("metadata") or data.get("metadados") or {}

        def pick(key: str) -> Any:
            return data.get(key, meta.get(key))

        emb = data.get("embedding")
        return cls(
            id=str(data["id"]),
            model_key=str(data.get("model_key", "")),
            crop=str(data.get("crop", "")),
            predicted_class=str(data.get("predicted_class", "")),
            confidence=float(data.get("confidence") or 0.0),
            probabilities=dict(data.get("probabilities") or {}),
            novelty_score=_opt_float(data.get("novelty_score")),
            prediction_set=(
                None if data.get("prediction_set") is None else list(data["prediction_set"])
            ),
            needs_review=bool(data.get("needs_review", False)),
            in_scope=data.get("in_scope"),
            created_at=str(data.get("created_at") or ""),
            embedding=None if emb is None else np.asarray(emb, dtype=np.float64).ravel(),
            field_id=_opt_str(pick("field_id")),
            plant_id=_opt_str(pick("plant_id")),
            camera=_opt_str(pick("camera")),
            label=_opt_str(data.get("label")),
        )


ItemLike = ReviewItem | Mapping[str, Any]


@dataclass(frozen=True)
class ScoreWeights:
    """Pesos da pontuação combinada.

    Incerteza domina porque é o sinal mais estável; novidade vem em seguida porque, em campo,
    o que é estranho ao treino (nova doença, nova câmera, praga fora de época) é exatamente o
    que o modelo mais precisa ver; raridade corrige o desbalanceamento do dataset.
    """

    uncertainty: float = 0.5
    novelty: float = 0.3
    rarity: float = 0.2
    # Pequeno bônus quando o próprio pipeline de inferência já marcou o item para revisão.
    needs_review_bonus: float = 0.05
    # Composição da incerteza: entropia, margem e conjunto conformal.
    entropy_share: float = 0.4
    margin_share: float = 0.4
    set_share: float = 0.2
    # Suavização de Laplace (alfa) da frequência de classe na raridade.
    rarity_smoothing: float = 1.0


@dataclass(frozen=True)
class ItemScores:
    """Pontuações normalizadas em [0, 1] de um item, com os valores brutos para explicação."""

    entropy: float
    margin: float
    set_size: int | None
    uncertainty: float
    novelty_raw: float | None
    novelty: float
    rarity: float
    rarity_count: int
    rarity_total: int
    total: float
    reasons: tuple[str, ...]

    def components(self) -> dict[str, float]:
        """Componentes normalizados, para exibir na UI de revisão."""
        return {
            "incerteza": round(self.uncertainty, 4),
            "novidade": round(self.novelty, 4),
            "raridade": round(self.rarity, 4),
            "total": round(self.total, 4),
        }


@dataclass(frozen=True)
class Selection:
    """Um item escolhido para o agrônomo, com posição (1 = rotular primeiro) e motivos."""

    id: str
    position: int
    score: float
    reasons: tuple[str, ...]
    strategy: str
    track: str = TRACK_MAIN
    model_key: str = ""
    predicted_class: str = ""
    components: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Pontuações
# ---------------------------------------------------------------------------------------------


def uncertainty_arrays(
    probs: np.ndarray, n_classes: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Entropia normalizada e margem (p1 - p2) por linha, vetorizado.

    ``probs`` pode vir com colunas de preenchimento (zeros) quando os modelos têm números de
    classes diferentes; por isso a entropia é dividida por log(K) do próprio item, e não da
    matriz. 0*log(0) é tratado como 0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if n_classes is None:
        n_classes = np.full(probs.shape[0], probs.shape[1])
    safe = np.where(probs > 0, probs, 1.0)
    ent = -(probs * np.log(safe)).sum(axis=1)
    log_k = np.log(np.maximum(n_classes, 2))
    ent_n = np.clip(ent / log_k, 0.0, 1.0)
    if probs.shape[1] >= 2:
        top2 = -np.partition(-probs, 1, axis=1)[:, :2]
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = np.ones(probs.shape[0])
    return ent_n, np.clip(margin, 0.0, 1.0)


def rarity_from_counts(
    counts: Mapping[str, int], classes: Iterable[str], smoothing: float = 1.0
) -> dict[str, float]:
    """Raridade por classe = 1 - f_c / f_max, com f_c suavizada por Laplace.

    Equivale a normalizar o inverso da frequência (w_c = 1/f_c) pela menor dessas razões:
    1 - w_min / w_c. A classe mais comum fica com 0, uma classe sem nenhum rótulo num dataset
    grande tende a 1, e sem rótulos nenhum a raridade é 0 para todas (não há o que corrigir).
    A suavização evita divisão por zero e o salto infinito de classes nunca rotuladas.
    """
    classes = sorted(set(classes) | set(counts))
    if not classes:
        return {}
    k = len(classes)
    total = sum(counts.values())
    freq = {c: (counts.get(c, 0) + smoothing) / (total + smoothing * k) for c in classes}
    f_max = max(freq.values())
    return {c: 1.0 - freq[c] / f_max for c in classes}


def score_items(
    pending: Sequence[ItemLike],
    labeled: Sequence[ItemLike] = (),
    weights: ScoreWeights | None = None,
) -> dict[str, ItemScores]:
    """Pontua cada item pendente (id -> :class:`ItemScores`).

    A pontuação total é uma média ponderada de componentes em [0, 1], mais o bônus de
    ``needs_review``. Ela é explicável: cada componente relevante gera um motivo em português.
    """
    weights = weights or ScoreWeights()
    items = [_coerce(x) for x in pending]
    lab = [_coerce(x) for x in labeled]
    if not items:
        return {}
    universe = _class_universe(items, lab)
    counts: dict[str, Counter[str]] = {}
    for it in lab:
        if it.label is not None:
            counts.setdefault(it.model_key, Counter())[it.label] += 1
    rarity_tables = {
        m: rarity_from_counts(counts.get(m, Counter()), cls, weights.rarity_smoothing)
        for m, cls in universe.items()
    }

    probs, n_cls, _ = _prob_matrix(items, universe)
    ent, margin = uncertainty_arrays(probs, n_cls)

    out: dict[str, ItemScores] = {}
    for i, it in enumerate(items):
        k = int(n_cls[i])
        set_size = None if it.prediction_set is None else len(it.prediction_set)
        parts = [(weights.entropy_share, ent[i]), (weights.margin_share, 1.0 - margin[i])]
        if set_size is not None:
            # Conjunto vazio: o calibrador não aceita nenhuma classe -> tão informativo quanto
            # o conjunto com todas as classes.
            set_score = 1.0 if set_size == 0 else (set_size - 1) / max(k - 1, 1)
            parts.append((weights.set_share, min(set_score, 1.0)))
        w_sum = sum(w for w, _ in parts) or 1.0
        unc = float(sum(w * v for w, v in parts) / w_sum)

        nov_raw = it.novelty_score
        # n / (1 + n): 0 em casa, 0,5 no limiar (n = 1) e satura suavemente para itens muito
        # estranhos, para que um único outlier extremo não esmague todo o resto.
        nov = 0.0 if nov_raw is None or nov_raw <= 0 else nov_raw / (1.0 + nov_raw)

        cls = _predicted(it, universe[it.model_key])
        rar = rarity_tables[it.model_key].get(cls, 0.0)
        c_counts = counts.get(it.model_key, Counter())

        w_total = weights.uncertainty + weights.novelty + weights.rarity or 1.0
        total = (weights.uncertainty * unc + weights.novelty * nov + weights.rarity * rar) / (
            w_total
        )
        if it.needs_review:
            total += weights.needs_review_bonus

        reasons: list[str] = []
        if margin[i] < 0.2:
            reasons.append(f"incerteza alta: margem {_br(margin[i])}")
        elif ent[i] >= 0.6:
            reasons.append(f"incerteza alta: entropia {_br(ent[i])}")
        if set_size == 0:
            reasons.append("conjunto conformal vazio: nenhuma classe plausível")
        elif set_size is not None and set_size > 1:
            reasons.append(f"conjunto conformal ambíguo: {set_size} classes")
        if nov_raw is not None and nov_raw >= 1.0:
            reasons.append(f"novidade {_br(nov_raw, 1)}")
        if rar >= 0.5:
            reasons.append(
                f"classe rara: {cls} ({c_counts.get(cls, 0)} de {sum(c_counts.values())} rotulados)"
            )
        if it.needs_review:
            reasons.append("marcado para revisão pela inferência")

        out[it.id] = ItemScores(
            entropy=float(ent[i]),
            margin=float(margin[i]),
            set_size=set_size,
            uncertainty=unc,
            novelty_raw=nov_raw,
            novelty=float(nov),
            rarity=float(rar),
            rarity_count=int(c_counts.get(cls, 0)),
            rarity_total=int(sum(c_counts.values())),
            total=float(total),
            reasons=tuple(reasons),
        )
    return out


# ---------------------------------------------------------------------------------------------
# Restrições do lote: cotas e quase-duplicatas
# ---------------------------------------------------------------------------------------------


class _Constraints:
    """Mantém a máscara de elegíveis enquanto o lote é montado.

    Todas as estratégias (inclusive a aleatória) passam pelas mesmas restrições, para que a
    comparação entre elas seja justa e para que o agrônomo nunca receba 20 fotos da mesma planta
    ou 40 folhas da mesma classe/talhão.
    """

    def __init__(
        self,
        n: int,
        class_codes: np.ndarray | None = None,
        field_codes: np.ndarray | None = None,
        max_per_class: int | None = None,
        max_per_field: int | None = None,
        unit_emb: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        dup_similarity: float | None = None,
    ) -> None:
        self.eligible = np.ones(n, dtype=bool)
        self.class_codes = class_codes
        self.field_codes = field_codes
        self.max_per_class = max_per_class
        self.max_per_field = max_per_field
        self.unit_emb = unit_emb
        self.groups = groups
        self.dup_similarity = dup_similarity
        self._class_count: Counter[int] = Counter()
        self._field_count: Counter[int] = Counter()

    def commit(self, i: int) -> None:
        self.eligible[i] = False
        if self.max_per_class is not None and self.class_codes is not None:
            c = int(self.class_codes[i])
            self._class_count[c] += 1
            if self._class_count[c] >= self.max_per_class:
                self.eligible[self.class_codes == c] = False
        if self.max_per_field is not None and self.field_codes is not None:
            f = int(self.field_codes[i])
            if f >= 0:  # talhão desconhecido (-1) não tem cota
                self._field_count[f] += 1
                if self._field_count[f] >= self.max_per_field:
                    self.eligible[self.field_codes == f] = False
        if self.dup_similarity is not None and self.unit_emb is not None:
            sims = self.unit_emb @ self.unit_emb[i]
            dup = sims >= self.dup_similarity
            if self.groups is not None:
                dup &= self.groups == self.groups[i]
            self.eligible[dup] = False


# ---------------------------------------------------------------------------------------------
# Núcleos de seleção (operam em arrays; reaproveitados pelo simulador)
# ---------------------------------------------------------------------------------------------


def badge_select(
    grad_a: np.ndarray,
    emb: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    constraints: _Constraints | None = None,
    scale: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """BADGE: semeadura k-means++ no embedding de gradiente, sem materializá-lo.

    O embedding de gradiente da última camada (entropia cruzada, rótulo = classe prevista) é
    g = (p - onehot(y_hat)) (x) h, de dimensão C*D (ex.: 8*1280). Em vez de construí-lo, usamos a
    identidade do produto de Kronecker <a_i (x) h_i, a_j (x) h_j> = <a_i, a_j> * <h_i, h_j>, com
    a = p - onehot(y_hat). Assim, cada nova semente custa O(n*(C + D)) e a memória fica O(n*D).

    Como em Ash et al. (2020), a primeira semente é o maior ||g|| e as seguintes são sorteadas
    com probabilidade proporcional a D^2 (distância ao centro mais próximo), o que favorece
    ao mesmo tempo magnitude (incerteza) e diversidade. ``scale`` multiplica g_i (extensão nossa
    para dar peso a novidade/raridade). Devolve (índice, distância ao escolhido mais próximo).
    """
    n = grad_a.shape[0]
    cons = constraints or _Constraints(n)
    s = np.ones(n) if scale is None else np.asarray(scale, dtype=np.float64)
    diag = (grad_a * grad_a).sum(1) * (emb * emb).sum(1) * s * s
    min_d2 = np.full(n, np.inf)
    chosen: list[tuple[int, float]] = []
    while len(chosen) < budget and cons.eligible.any():
        idx = np.flatnonzero(cons.eligible)
        if not chosen:
            pick = int(idx[np.argmax(diag[idx])])
            dist = math.inf
        else:
            w = min_d2[idx]
            total = float(w.sum())
            if not np.isfinite(total) or total <= _EPS:
                pick = int(rng.choice(idx))  # todos idênticos ao já escolhido: sorteio uniforme
            else:
                pick = int(rng.choice(idx, p=w / total))
            dist = math.sqrt(max(float(min_d2[pick]), 0.0))
        chosen.append((pick, dist))
        col = (grad_a @ grad_a[pick]) * (emb @ emb[pick]) * s * s[pick]
        d2 = np.maximum(diag + diag[pick] - 2.0 * col, 0.0)
        np.minimum(min_d2, d2, out=min_d2)
        cons.commit(pick)
    return chosen


def kcenter_select(
    emb: np.ndarray,
    budget: int,
    centers: np.ndarray | None = None,
    priority: np.ndarray | None = None,
    constraints: _Constraints | None = None,
    groups: np.ndarray | None = None,
    center_groups: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Core-set guloso k-center (Sener & Savarese, 2018).

    A cada passo escolhe o ponto mais distante de todos os centros (já rotulados + já escolhidos),
    o que minimiza o raio de cobertura: garantia 2-aproximada do problema k-center. Distâncias só
    valem dentro do mesmo grupo (modelo): embeddings de modelos diferentes não são comparáveis, e
    um grupo sem nenhum centro fica a distância infinita, sendo coberto primeiro. Empates (ex.:
    vários infinitos) são desempatados por ``priority`` (a pontuação do item).
    """
    n = emb.shape[0]
    cons = constraints or _Constraints(n)
    g = np.zeros(n, dtype=np.int64) if groups is None else groups
    prio = np.zeros(n) if priority is None else priority
    min_d = np.full(n, np.inf)
    if centers is not None and len(centers):
        cg = np.zeros(len(centers), dtype=np.int64) if center_groups is None else center_groups
        for grp in np.unique(g):
            mine = np.flatnonzero(g == grp)
            ctr = centers[cg == grp]
            if len(ctr):
                min_d[mine] = _min_euclidean(emb[mine], ctr)
    chosen: list[tuple[int, float]] = []
    while len(chosen) < budget and cons.eligible.any():
        idx = np.flatnonzero(cons.eligible)
        md = min_d[idx]
        best = md.max()
        ties = idx[md == best]
        pick = int(ties[np.argmax(prio[ties])])
        chosen.append((pick, float(best)))
        same = g == g[pick]
        d = np.linalg.norm(emb[same] - emb[pick], axis=1)
        min_d[same] = np.minimum(min_d[same], d)
        cons.commit(pick)
    return chosen


def ranked_select(
    order: Sequence[int], budget: int, constraints: _Constraints | None = None
) -> list[int]:
    """Percorre uma ordem de prioridade aceitando só itens ainda elegíveis."""
    cons = constraints or _Constraints(len(order))
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= budget:
            break
        if cons.eligible[i]:
            chosen.append(int(i))
            cons.commit(int(i))
    return chosen


# ---------------------------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------------------------


def select_for_review(
    pending: Sequence[ItemLike],
    labeled: Sequence[ItemLike] = (),
    budget: int = 50,
    strategy: Strategy = "badge",
    seed: int | None = 0,
    *,
    max_per_class: int | float | None = None,
    max_per_field: int | float | None = None,
    near_duplicate_distance: float | None = 0.03,
    weights: ScoreWeights | None = None,
    badge_boost: float = 1.0,
) -> list[Selection]:
    """Escolhe até ``budget`` itens pendentes para o agrônomo rotular, em ordem de prioridade.

    Parâmetros
    ----------
    strategy:
        ``"badge"`` (padrão; precisa de embeddings), ``"coreset"`` (precisa de embeddings),
        ``"uncertainty"`` (ranking pela pontuação combinada) ou ``"random"`` (linha de base).
        Sem embedding em algum candidato, BADGE/core-set caem para ``"uncertainty"`` e isso fica
        registrado em ``Selection.strategy``.
    max_per_class / max_per_field:
        Cotas por (modelo, classe prevista) e por talhão. Inteiro = número absoluto; float em
        (0, 1] = fração do orçamento (arredondada para cima). São restrições rígidas: se não
        houver itens elegíveis suficientes, o lote sai menor que o orçamento.
    near_duplicate_distance:
        Distância cosseno máxima no embedding para dois recortes serem considerados a mesma
        planta/quadro vizinho. Ao escolher um item, seus quase-duplicados (pendentes e do mesmo
        modelo) saem da disputa; pendentes quase-duplicados de itens já rotulados nem entram.
        ``None`` desliga.
    badge_boost:
        Multiplica o gradiente de cada item por 1 + boost*(novidade/raridade ponderadas), para que
        BADGE também valorize o que é estranho ao treino e as classes raras. 0 = BADGE puro.

    Itens com ``in_scope`` False nunca entram aqui (use :func:`select_out_of_scope`), assim como
    pendentes cujo id já aparece entre os rotulados.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy!r}; use uma de {STRATEGIES}")
    if budget <= 0:
        return []
    weights = weights or ScoreWeights()
    lab = [_coerce(x) for x in labeled]
    lab_ids = {x.id for x in lab}
    items = [
        it
        for it in (_coerce(x) for x in pending)
        if it.in_scope is not False and it.id not in lab_ids
    ]
    if not items:
        return []

    scores = score_items(items, lab, weights)
    total = np.array([scores[it.id].total for it in items])
    universe = _class_universe(items, lab)

    has_emb = all(it.embedding is not None for it in items)
    used = strategy
    if strategy in ("badge", "coreset") and not has_emb:
        logger.warning("%s sem embeddings em todos os itens; usando 'uncertainty'", strategy)
        used = "uncertainty"

    emb, groups = _embedding_matrix(items) if has_emb else (None, None)
    cons = _build_constraints(
        items,
        budget,
        max_per_class,
        max_per_field,
        emb,
        groups,
        near_duplicate_distance,
    )
    if emb is not None and near_duplicate_distance is not None:
        _drop_duplicates_of_labeled(items, lab, emb, groups, near_duplicate_distance, cons)

    rng = np.random.default_rng(seed)
    picks: list[tuple[int, float | None]]
    if used == "badge":
        assert emb is not None
        _, _, grad_a = _prob_matrix(items, universe, with_grad=True)
        w_extra = weights.novelty + weights.rarity
        extra = np.array(
            [
                (weights.novelty * scores[it.id].novelty + weights.rarity * scores[it.id].rarity)
                / (w_extra or 1.0)
                for it in items
            ]
        )
        picks = list(badge_select(grad_a, emb, budget, rng, cons, scale=1.0 + badge_boost * extra))
    elif used == "coreset":
        assert emb is not None and groups is not None
        centers, center_groups = _labeled_centers(items, lab)
        picks = list(kcenter_select(emb, budget, centers, total, cons, groups, center_groups))
    elif used == "uncertainty":
        order = sorted(
            range(len(items)), key=lambda i: (-total[i], items[i].created_at, items[i].id)
        )
        picks = [(i, None) for i in ranked_select(order, budget, cons)]
    else:
        order = [int(i) for i in rng.permutation(len(items))]
        picks = [(i, None) for i in ranked_select(order, budget, cons)]

    label = used if used == strategy else f"{used} (fallback de {strategy})"
    finite = [d for _, d in picks if d is not None and math.isfinite(d)]
    d_ref = max(finite) if finite else 1.0
    out: list[Selection] = []
    for pos, (i, dist) in enumerate(picks, start=1):
        it = items[i]
        sc = scores[it.id]
        reasons = list(sc.reasons)
        if used == "random":
            reasons = ["amostra aleatória (linha de base)"]
        elif used == "badge" and pos == 1:
            reasons.append("maior gradiente esperado do lote")
        elif used == "coreset" and dist is not None and not math.isfinite(dist):
            reasons.append("diversidade: cobre região do modelo ainda sem rótulos")
        if dist is not None and math.isfinite(dist) and used != "random":
            rel = dist / d_ref if d_ref > _EPS else 0.0
            reasons.append(f"diversidade: longe dos já escolhidos (distância relativa {_br(rel)})")
        if not reasons:
            reasons.append(f"pontuação combinada {_br(sc.total)}")
        out.append(
            Selection(
                id=it.id,
                position=pos,
                score=round(sc.total, 6),
                reasons=tuple(reasons),
                strategy=label,
                track=TRACK_MAIN,
                model_key=it.model_key,
                predicted_class=it.predicted_class,
                components=sc.components(),
            )
        )
    return out


def select_out_of_scope(
    pending: Sequence[ItemLike],
    budget: int = 20,
    near_duplicate_distance: float | None = 0.03,
) -> list[Selection]:
    """Trilha separada para itens que o portão de enquadramento rejeitou (``in_scope`` False).

    Não faz sentido comparar essas imagens com folhas: elas não ensinam o classificador de
    doenças, mas ensinam o portão. O item mais valioso aqui é o possível falso negativo do portão
    (o classificador vê uma doença com confiança alta), seguido do mais estranho (novidade).
    """
    items = [it for it in (_coerce(x) for x in pending) if it.in_scope is False]
    if not items or budget <= 0:
        return []
    key = [
        (it.confidence + 0.25 * min((it.novelty_score or 0.0) / 2.0, 1.0), it.id) for it in items
    ]
    order = sorted(range(len(items)), key=lambda i: (-key[i][0], key[i][1]))
    has_emb = all(it.embedding is not None for it in items)
    emb, groups = _embedding_matrix(items) if has_emb else (None, None)
    cons = _build_constraints(items, budget, None, None, emb, groups, near_duplicate_distance)
    out: list[Selection] = []
    for pos, i in enumerate(ranked_select(order, budget, cons), start=1):
        it = items[i]
        reasons = ["fora de escopo pelo portão de enquadramento"]
        if it.confidence >= 0.8:
            reasons.append(
                f"possível falso negativo do portão: modelo confiante {_br(it.confidence)} "
                f"em {it.predicted_class}"
            )
        if it.novelty_score is not None and it.novelty_score >= 1.0:
            reasons.append(f"novidade {_br(it.novelty_score, 1)}")
        out.append(
            Selection(
                id=it.id,
                position=pos,
                score=round(key[i][0], 6),
                reasons=tuple(reasons),
                strategy="ranking fora de escopo",
                track=TRACK_OUT_OF_SCOPE,
                model_key=it.model_key,
                predicted_class=it.predicted_class,
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# Simulador de ganho
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CurvePoint:
    """Um ponto da curva: nº de rótulos x acurácia (média e desvio sobre as seeds)."""

    n_labels: int
    accuracy_mean: float
    accuracy_std: float
    balanced_accuracy_mean: float


@dataclass(frozen=True)
class SimulationResult:
    """Curvas por estratégia e matriz bruta (seeds x rodadas) de acurácia."""

    curves: dict[str, list[CurvePoint]]
    per_seed_accuracy: dict[str, np.ndarray]

    def mean_accuracy(self, strategy: str) -> float:
        """Acurácia média ao longo da curva (área sob a curva normalizada)."""
        return float(self.per_seed_accuracy[strategy].mean())

    def table(self) -> str:
        """Tabela em texto: linhas = nº de rótulos, colunas = estratégias."""
        names = list(self.curves)
        head = "rótulos | " + " | ".join(f"{n:>11}" for n in names)
        lines = [head, "-" * len(head)]
        for j, pt in enumerate(self.curves[names[0]]):
            cells = [
                f"{self.curves[n][j].accuracy_mean:.3f}±{self.curves[n][j].accuracy_std:.3f}"
                for n in names
            ]
            lines.append(f"{pt.n_labels:>7} | " + " | ".join(f"{c:>11}" for c in cells))
        return "\n".join(lines)


def fit_softmax_regression(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    lr: float = 0.5,
    epochs: int = 300,
    l2: float = 1e-3,
) -> np.ndarray:
    """Regressão logística multinomial por gradiente descendente em lote (numpy puro).

    Devolve W de forma (D+1, C), com o viés na última linha. A perda é convexa, então lote
    completo com passo fixo converge bem para os tamanhos do simulador (dezenas a centenas de
    rótulos) sem precisar de um otimizador externo.
    """
    xb = _with_bias(x)
    w = np.zeros((xb.shape[1], n_classes))
    onehot = np.eye(n_classes)[y]
    n = max(len(y), 1)
    for _ in range(epochs):
        p = _softmax(xb @ w)
        grad = xb.T @ (p - onehot) / n
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return w


def predict_proba(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Probabilidades softmax do modelo linear."""
    return _softmax(_with_bias(x) @ w)


def simulate_labeling(
    features: np.ndarray,
    labels: np.ndarray,
    strategies: Sequence[str] = ("random", "uncertainty", "badge", "coreset"),
    *,
    initial_size: int = 10,
    batch_size: int = 10,
    rounds: int = 10,
    seeds: Sequence[int] = (0, 1, 2),
    test_fraction: float = 0.3,
    epochs: int = 300,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> SimulationResult:
    """Simula rodadas de rotulagem e mede acurácia x nº de rótulos por estratégia.

    Para cada seed: separa um conjunto de teste estratificado, sorteia o mesmo lote inicial para
    todas as estratégias (comparação pareada), treina uma regressão logística multinomial e, a
    cada rodada, cada estratégia escolhe ``batch_size`` itens do pool cujo rótulo é "revelado".
    O embedding h usado por BADGE/core-set é a própria feature padronizada (com viés), que é
    exatamente a entrada da última camada do modelo linear.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"estratégia desconhecida: {s!r}")

    acc = {s: np.zeros((len(seeds), rounds + 1)) for s in strategies}
    bacc = {s: np.zeros((len(seeds), rounds + 1)) for s in strategies}
    n_labels = [initial_size + r * batch_size for r in range(rounds + 1)]
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        test_idx, pool_idx = _stratified_split(y_idx, test_fraction, rng)
        mu = x[pool_idx].mean(0)
        sd = x[pool_idx].std(0) + _EPS
        xs = (x - mu) / sd  # padronização só com o pool (não olha rótulos)
        init = rng.choice(len(pool_idx), size=min(initial_size, len(pool_idx)), replace=False)
        for s in strategies:
            srng = np.random.default_rng([seed, STRATEGIES.index(s)])
            chosen = np.zeros(len(pool_idx), dtype=bool)
            chosen[init] = True
            for r in range(rounds + 1):
                tr = pool_idx[chosen]
                w = fit_softmax_regression(
                    xs[tr], y_idx[tr], n_classes, lr=lr, epochs=epochs, l2=l2
                )
                pred = predict_proba(w, xs[test_idx]).argmax(1)
                acc[s][si, r] = float((pred == y_idx[test_idx]).mean())
                bacc[s][si, r] = _balanced_accuracy(y_idx[test_idx], pred, n_classes)
                if r == rounds:
                    break
                new = _simulated_batch(
                    s, w, xs, pool_idx, chosen, y_idx, n_classes, batch_size, srng
                )
                chosen[new] = True
    curves = {
        s: [
            CurvePoint(
                n_labels=n_labels[r],
                accuracy_mean=float(acc[s][:, r].mean()),
                accuracy_std=float(acc[s][:, r].std()),
                balanced_accuracy_mean=float(bacc[s][:, r].mean()),
            )
            for r in range(rounds + 1)
        ]
        for s in strategies
    }
    return SimulationResult(curves=curves, per_seed_accuracy=acc)


def make_synthetic_crop_data(
    n: int = 1200,
    n_features: int = 16,
    class_weights: Sequence[float] = (0.45, 0.3, 0.15, 0.07, 0.03),
    separation: float = 2.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussianas desbalanceadas, com uma classe rara (a última), para o simulador e os testes.

    Imita o cenário de campo: poucas classes dominam (folha sadia, Sigatoka) e uma doença rara
    (ex.: Cordana) aparece em ~3% dos recortes.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(class_weights, dtype=np.float64)
    p = p / p.sum()
    k = len(p)
    means = rng.normal(size=(k, n_features))
    means *= separation / np.linalg.norm(means, axis=1, keepdims=True)
    y = rng.choice(k, size=n, p=p)
    # garante ao menos alguns exemplos de cada classe mesmo com n pequeno
    for c in range(k):
        if (y == c).sum() < 6:
            y[rng.choice(n, 6, replace=False)] = c
    x = means[y] + rng.normal(size=(n, n_features))
    return x, y


# ---------------------------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------------------------


def _simulated_batch(
    strategy: str,
    w: np.ndarray,
    xs: np.ndarray,
    pool_idx: np.ndarray,
    chosen: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Um passo de seleção no simulador; devolve índices relativos ao pool."""
    cand = np.flatnonzero(~chosen)
    if len(cand) == 0:
        return cand
    budget = min(batch, len(cand))
    xc = xs[pool_idx[cand]]
    if strategy == "random":
        return rng.choice(cand, size=budget, replace=False)
    probs = predict_proba(w, xc)
    if strategy == "uncertainty":
        ent, margin = uncertainty_arrays(probs)
        counts = Counter(y_idx[pool_idx[chosen]].tolist())
        rar_tab = rarity_from_counts(
            {str(k): v for k, v in counts.items()}, [str(c) for c in range(n_classes)]
        )
        rar = np.array([rar_tab[str(c)] for c in probs.argmax(1)])
        wt = ScoreWeights()
        unc = (wt.entropy_share * ent + wt.margin_share * (1 - margin)) / (
            wt.entropy_share + wt.margin_share
        )
        score = (wt.uncertainty * unc + wt.rarity * rar) / (wt.uncertainty + wt.rarity)
        order = np.argsort(-score, kind="stable")
        return cand[order[:budget]]
    h = _with_bias(xc)
    if strategy == "badge":
        grad_a = probs - np.eye(n_classes)[probs.argmax(1)]
        picks = badge_select(grad_a, h, budget, rng)
    else:
        centers = _with_bias(xs[pool_idx[chosen]])
        picks = kcenter_select(h, budget, centers)
    return cand[[i for i, _ in picks]]


def _coerce(x: ItemLike) -> ReviewItem:
    return x if isinstance(x, ReviewItem) else ReviewItem.from_mapping(x)


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)


def _br(v: float, digits: int = 2) -> str:
    """Número no formato brasileiro (vírgula decimal), como o agrônomo lê."""
    return f"{v:.{digits}f}".replace(".", ",")


def _class_universe(items: Sequence[ReviewItem], lab: Sequence[ReviewItem]) -> dict[str, list[str]]:
    """Classes conhecidas por modelo: chaves das probabilidades + previstas + rótulos."""
    uni: dict[str, set[str]] = {}
    for it in items:
        s = uni.setdefault(it.model_key, set())
        s.update(it.probabilities)
        if it.predicted_class:
            s.add(it.predicted_class)
    for it in lab:
        if it.model_key in uni and it.label is not None:
            uni[it.model_key].add(it.label)
    return {m: sorted(s) for m, s in uni.items()}


def _predicted(it: ReviewItem, classes: Sequence[str]) -> str:
    if it.predicted_class:
        return it.predicted_class
    if it.probabilities:
        return max(it.probabilities.items(), key=lambda kv: kv[1])[0]
    return classes[0] if classes else ""


def _prob_matrix(
    items: Sequence[ReviewItem], universe: Mapping[str, Sequence[str]], with_grad: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilidades alinhadas por classe e, opcionalmente, a = p - onehot(y_hat).

    Para ``a`` as colunas são pares (modelo, classe): modelos diferentes têm cabeças (últimas
    camadas) diferentes, logo seus gradientes vivem em subespaços ortogonais e <a_i, a_j> = 0
    entre modelos, o que o BADGE herda automaticamente pela identidade de Kronecker.
    """
    c_max = max(len(c) for c in universe.values())
    probs = np.zeros((len(items), c_max))
    n_cls = np.zeros(len(items), dtype=np.int64)
    offsets: dict[str, int] = {}
    acc = 0
    for m in sorted(universe):
        offsets[m] = acc
        acc += len(universe[m])
    grad_a = np.zeros((len(items), acc)) if with_grad else np.zeros((0, 0))
    for i, it in enumerate(items):
        cls = universe[it.model_key]
        col = {c: j for j, c in enumerate(cls)}
        row = np.zeros(len(cls))
        for c, p in it.probabilities.items():
            row[col[c]] = max(float(p), 0.0)
        if row.sum() <= 0:
            # sem vetor de probabilidades: reconstrói a partir da confiança
            row[col[_predicted(it, cls)]] = it.confidence
            rest = (1.0 - it.confidence) / max(len(cls) - 1, 1)
            row[row == 0] = rest
        row /= row.sum()
        probs[i, : len(cls)] = row
        n_cls[i] = len(cls)
        if with_grad:
            a = row.copy()
            a[int(np.argmax(row))] -= 1.0  # rótulo hipotético = classe prevista (BADGE)
            grad_a[i, offsets[it.model_key] : offsets[it.model_key] + len(cls)] = a
    return probs, n_cls, grad_a


def _embedding_matrix(items: Sequence[ReviewItem]) -> tuple[np.ndarray, np.ndarray]:
    """Empilha embeddings (com preenchimento de zeros se as dimensões diferem) e códigos de grupo.

    O preenchimento é seguro: distâncias e similaridades só são usadas dentro do mesmo modelo, e
    no BADGE o termo <a_i, a_j> zera o produto entre modelos.
    """
    dims = [len(it.embedding) for it in items if it.embedding is not None]
    d = max(dims)
    emb = np.zeros((len(items), d))
    for i, it in enumerate(items):
        assert it.embedding is not None
        emb[i, : len(it.embedding)] = it.embedding
    models = sorted({it.model_key for it in items})
    code = {m: j for j, m in enumerate(models)}
    groups = np.array([code[it.model_key] for it in items], dtype=np.int64)
    return emb, groups


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + _EPS)


def _resolve_quota(q: int | float | None, budget: int) -> int | None:
    if q is None:
        return None
    if isinstance(q, float) and 0 < q <= 1:
        return max(1, math.ceil(q * budget))
    if int(q) < 1:
        raise ValueError("cota deve ser >= 1 ou uma fração em (0, 1]")
    return int(q)


def _build_constraints(
    items: Sequence[ReviewItem],
    budget: int,
    max_per_class: int | float | None,
    max_per_field: int | float | None,
    emb: np.ndarray | None,
    groups: np.ndarray | None,
    dup_distance: float | None,
) -> _Constraints:
    class_keys = [(it.model_key, it.predicted_class) for it in items]
    cmap = {k: j for j, k in enumerate(sorted(set(class_keys)))}
    fmap = {k: j for j, k in enumerate(sorted({it.field_id for it in items if it.field_id}))}
    return _Constraints(
        len(items),
        class_codes=np.array([cmap[k] for k in class_keys], dtype=np.int64),
        field_codes=np.array([fmap.get(it.field_id or "", -1) for it in items], dtype=np.int64),
        max_per_class=_resolve_quota(max_per_class, budget),
        max_per_field=_resolve_quota(max_per_field, budget),
        unit_emb=None if emb is None else _unit(emb),
        groups=groups,
        dup_similarity=None if dup_distance is None or emb is None else 1.0 - dup_distance,
    )


def _labeled_matrix(
    lab: Sequence[ReviewItem], items: Sequence[ReviewItem], d: int
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings dos rotulados cujo modelo aparece entre os pendentes, com código de grupo."""
    models = sorted({it.model_key for it in items})
    code = {m: j for j, m in enumerate(models)}
    rows = [it for it in lab if it.embedding is not None and it.model_key in code]
    mat = np.zeros((len(rows), d))
    for i, it in enumerate(rows):
        assert it.embedding is not None
        e = it.embedding[:d]
        mat[i, : len(e)] = e
    return mat, np.array([code[it.model_key] for it in rows], dtype=np.int64)


def _labeled_centers(
    items: Sequence[ReviewItem], lab: Sequence[ReviewItem]
) -> tuple[np.ndarray, np.ndarray]:
    d = max(len(it.embedding) for it in items if it.embedding is not None)
    return _labeled_matrix(lab, items, d)


def _drop_duplicates_of_labeled(
    items: Sequence[ReviewItem],
    lab: Sequence[ReviewItem],
    emb: np.ndarray,
    groups: np.ndarray | None,
    dup_distance: float,
    cons: _Constraints,
    chunk: int = 2048,
) -> None:
    """Tira da disputa pendentes que são quase cópia de algo já rotulado (em blocos de memória)."""
    lab_mat, lab_groups = _labeled_matrix(lab, items, emb.shape[1])
    if not len(lab_mat):
        return
    unit_p, unit_l = _unit(emb), _unit(lab_mat)
    g = np.zeros(len(items), dtype=np.int64) if groups is None else groups
    for grp in np.unique(g):
        mine = np.flatnonzero(g == grp)
        ref = unit_l[lab_groups == grp]
        if not len(ref):
            continue
        for start in range(0, len(mine), chunk):
            block = mine[start : start + chunk]
            best = (unit_p[block] @ ref.T).max(axis=1)
            cons.eligible[block[best >= 1.0 - dup_distance]] = False


def _min_euclidean(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """min_j ||a_i - b_j|| em blocos, via ||a||^2 + ||b||^2 - 2<a, b>."""
    out = np.empty(len(a))
    b2 = (b * b).sum(1)
    for start in range(0, len(a), chunk):
        blk = a[start : start + chunk]
        d2 = (blk * blk).sum(1)[:, None] + b2[None, :] - 2.0 * blk @ b.T
        out[start : start + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _with_bias(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _stratified_split(
    y: np.ndarray, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    test: list[int] = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = max(1, round(test_fraction * len(idx)))
        test.extend(idx[:k].tolist())
    test_arr = np.array(sorted(test))
    pool = np.setdiff1d(np.arange(len(y)), test_arr)
    return test_arr, pool


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    recalls = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0
