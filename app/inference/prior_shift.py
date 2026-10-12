"""Adaptação à prevalência do talhão (label shift) e monitor de deriva.

Motivação
---------
Os classificadores do VALTEXA-B foram treinados em datasets públicos com prevalência artificial
(ex.: banana com 35% Sigatoka e 30% sadia). Numa fazenda a prevalência real pode ser 3% de doença
e muda por talhão e estação. Pela regra de Bayes, o posterior aprendido embute o prior do treino:

    p_s(y|x) ∝ p(x|y) · π_s(y)

Se a aparência de cada classe não muda (hipótese de *label shift*: p_t(x|y) = p_s(x|y)), o
posterior correto no talhão é

    p_t(y|x) ∝ p_s(y|x) · π_t(y) / π_s(y)

(Saerens, Latinne & Decaestecker, 2002, eq. 2.4). Falta só conhecer π_t, que estimamos SEM
rótulos a partir das previsões do próprio talhão.

Quando usar cada estimador
--------------------------
* **EM / MLLS** (Saerens et al., 2002): máxima verossimilhança de π_t usando as probabilidades
  completas. É o estimador mais eficiente quando o modelo está *calibrado* — Alexandari, Kundaje
  & Shrikumar (2020, "Maximum likelihood with bias-corrected calibration is hard-to-beat at
  label shift adaptation") e Garg et al. (2020, "A unified view of label shift estimation")
  mostram que ele bate o BBSE nesse regime. Mas, se o modelo for descalibrado (ex.: redes
  super-confiantes, ou suavizadas por label smoothing), o EM converge para o ponto errado,
  porque trata p(y|x) como verdade.
* **BBSE** (Lipton, Wang & Smola, 2018): só exige que a matriz de confusão C[i, j] =
  P(ŷ=i, y=j), medida numa validação rotulada com o prior do treino, seja invertível. Como
  P_t(ŷ=i) = Σ_j C[i, j]·w_j com w_j = π_t(j)/π_s(j), basta resolver um sistema linear. É
  consistente com qualquer classificador (calibrado ou não), mas tem variância maior (usa só o
  argmax) e explode quando C é mal condicionada (classes confundidas entre si).

A função :func:`estimate_target_prior` escolhe/combina os dois com base em
:class:`ValidationSummary` (ECE por classe e número de condição de C). Ver :func:`bbse_weight`.

Estado por talhão
-----------------
:class:`FieldPriorStore` mantém, por ``field_id``, probabilidades recentes com pesos de
esquecimento exponencial (EM ponderado re-rodado a cada lote sobre um buffer recente mais
um resumo em células das amostras antigas; ver :class:`TrackerConfig`), serializável em JSON
com escrita atômica.

Monitor de deriva
-----------------
A correção de prior só vale sob *label shift*. Quando o talhão deixa de parecer o treino
(câmera nova, doença não vista, poeira, outra cultivar), é preciso avisar. :func:`check_drift`
compara uma janela recente com a referência (validação) usando KS, PSI, qui-quadrado e MMD.
Se a referência tiver rótulos e for passado o prior estimado do talhão, ela é *reponderada*
por π_t/π_s antes dos testes: assim uma mudança pura de prevalência (esperada e já tratada)
não dispara alarme, e o que sobra é deriva de aparência.

Só usa stdlib + numpy (sem scipy): o p-valor do KS vem da distribuição assintótica de
Kolmogorov e o do qui-quadrado da função gama incompleta regularizada (Numerical Recipes, 6.2).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

Method = Literal["em", "bbse", "combined", "source"]
Level = Literal["ok", "atenção", "deriva"]

LEVEL_OK: Level = "ok"
LEVEL_WARN: Level = "atenção"
LEVEL_DRIFT: Level = "deriva"
_LEVEL_RANK: dict[str, int] = {LEVEL_OK: 0, LEVEL_WARN: 1, LEVEL_DRIFT: 2}

STATE_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 2) -> str:
    """Formata número com vírgula decimal (pt-BR) para as mensagens de motivo."""
    return f"{x:.{digits}f}".replace(".", ",")


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def _fmt_p(p: float) -> str:
    return "p<0,001" if p < 1e-3 else f"p={_fmt(p, 3)}"


def validate_probabilities(probabilities: ArrayLike, n_classes: int | None = None) -> FloatArray:
    """Valida e normaliza uma matriz (N, K) de probabilidades.

    Aceita um único vetor (K,) e o trata como N=1. Linhas são renormalizadas porque saídas
    float32 de softmax raramente somam exatamente 1; valores negativos ou não finitos são erro
    (indicam bug a montante, não ruído, e não devem ser silenciados).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"esperado array (N, K) com K>=2, recebido shape {p.shape}")
    if n_classes is not None and p.shape[1] != n_classes:
        raise ValueError(f"esperado K={n_classes} classes, recebido {p.shape[1]}")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilidades devem ser finitas e não negativas")
    s = p.sum(axis=1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("linha de probabilidades com soma zero")
    return p / s


def _validate_prior(prior: ArrayLike, n_classes: int | None = None) -> FloatArray:
    pi = np.asarray(prior, dtype=np.float64).ravel()
    if n_classes is not None and pi.size != n_classes:
        raise ValueError(f"prior com {pi.size} classes, esperado {n_classes}")
    if not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        # π_s(y)=0 tornaria a razão π_t/π_s indefinida; é erro de configuração.
        raise ValueError("prior deve ter todas as entradas finitas e > 0")
    return pi / pi.sum()


def apply_floor(prevalences: ArrayLike, floor: float) -> FloatArray:
    """Impõe prevalência mínima ``floor`` a cada classe, preservando soma 1.

    Por que o piso: se o EM zerar uma classe (ex.: nenhuma Cordana nas primeiras 200 fotos),
    a correção p·π_t/π_s zera essa classe para sempre e o sistema fica cego a um surto. O piso
    mantém a classe "viva". As classes abaixo do piso ficam exatamente no piso e o restante da
    massa é redistribuído proporcionalmente entre as demais (iterando até estabilizar).
    """
    p = np.clip(np.asarray(prevalences, dtype=np.float64), 0.0, None)
    p = p / p.sum()
    if floor <= 0:
        return p
    k = p.size
    if floor * k >= 1.0:
        raise ValueError(f"piso {floor} impossível para {k} classes")
    fixed = p < floor
    while True:
        free_mass = 1.0 - floor * fixed.sum()
        free_sum = p[~fixed].sum()
        q = np.where(fixed, floor, p * free_mass / free_sum)
        newly = (q < floor) & ~fixed
        if not newly.any():
            return q
        fixed |= newly


def project_to_simplex(v: ArrayLike) -> FloatArray:
    """Projeção euclidiana no simplex {π >= 0, Σπ = 1}.

    Algoritmo por ordenação de Duchi, Shalev-Shwartz, Singer & Chandra (2008), "Efficient
    projections onto the l1-ball for learning in high dimensions". Necessária no BBSE porque a
    inversão de C, com ruído amostral, pode dar prevalências negativas ou soma != 1.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, v.size + 1)
    rho = int(np.nonzero(u - css / idx > 0)[0][-1])
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def adjust_posteriors(
    probabilities: ArrayLike, source_prior: ArrayLike, target_prior: ArrayLike
) -> FloatArray:
    """Corrige posteriores para o prior do talhão: p'(y|x) ∝ p(y|x)·π_t(y)/π_s(y).

    Saerens et al. (2002), eq. 2.4. Exata sob label shift e modelo calibrado; com modelo
    descalibrado continua movendo a decisão na direção certa (o limiar efetivo passa a
    refletir a prevalência real), mas as probabilidades resultantes herdam a descalibração.
    """
    p = validate_probabilities(probabilities)
    pi_s = _validate_prior(source_prior, p.shape[1])
    pi_t = np.asarray(target_prior, dtype=np.float64).ravel()
    if pi_t.size != p.shape[1] or np.any(pi_t < 0) or not np.all(np.isfinite(pi_t)):
        raise ValueError("prior-alvo inválido")
    w = p * (pi_t / pi_t.sum() / pi_s)
    s = w.sum(axis=1, keepdims=True)
    # Se π_t zerou exatamente as classes com massa (não deveria, pelo piso), mantém o original.
    return np.where(s > 0, w / np.where(s > 0, s, 1.0), p)


# ---------------------------------------------------------------------------
# EM de Saerens, Latinne & Decaestecker (2002)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EMConfig:
    """Salvaguardas do EM.

    Attributes:
        min_samples: abaixo disso não se estima nada e devolve-se o prior do treino. Com poucas
            imagens a verossimilhança é plana e o EM tende a empurrar massa para o canto do
            simplex (classes zeradas); 30 é o mínimo para o erro-padrão de uma prevalência
            de 30% ficar em ~8 p.p.
        dirichlet_alpha: pseudo-contagens por classe (prior Dirichlet simétrico; estimativa
            tipo "média a posteriori"). 0,5 é o prior de Jeffreys para a multinomial:
            suaviza sem puxar muito para o uniforme quando N é grande.
        max_iter / tol: o EM de Saerens converge linearmente e pode ser lento quando classes
            se confundem; limitar evita travar a API. ``tol`` é em norma-máx das prevalências.
        prevalence_floor: ver :func:`apply_floor`.
    """

    min_samples: int = 30
    dirichlet_alpha: float = 0.5
    max_iter: int = 1000
    tol: float = 1e-6
    prevalence_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.min_samples < 1 or self.max_iter < 1:
            raise ValueError("min_samples e max_iter devem ser >= 1")
        if self.dirichlet_alpha < 0 or self.tol <= 0 or not 0 <= self.prevalence_floor < 0.5:
            raise ValueError("parâmetros do EM fora do domínio")


@dataclass(frozen=True)
class PriorEstimate:
    """Resultado de uma estimativa de prevalência-alvo."""

    prevalences: FloatArray
    method: Method
    n_samples: float
    iterations: int = 0
    converged: bool = True
    bbse_weight: float = 0.0
    notes: tuple[str, ...] = ()


def _em_fixed_point(
    probabilities: FloatArray,
    source_prior: FloatArray,
    init: FloatArray,
    config: EMConfig,
    weights: FloatArray | None = None,
) -> tuple[FloatArray, int, bool]:
    """Iteração de ponto fixo do EM de Saerens et al. (2002), com pesos por amostra.

    Passo E: r_ik = p(k|x_i)·π(k)/π_s(k), normalizado por linha.
    Passo M: π(k) = (Σ_i v_i·r_ik + alfa) / (Σ_i v_i + K·alfa), com v_i os pesos de esquecimento
    (v=1 no EM em lote). O termo alfa é a estimativa posterior-média com prior Dirichlet.

    Returns:
        (prevalências, iterações, convergiu)
    """
    k = probabilities.shape[1]
    ratio = probabilities / source_prior
    v = np.ones(probabilities.shape[0]) if weights is None else weights
    alpha = config.dirichlet_alpha
    denom = float(v.sum()) + k * alpha
    pi = init.copy()
    converged = False
    it = 0
    while it < config.max_iter:
        it += 1
        r = ratio * pi
        r /= r.sum(axis=1, keepdims=True)
        new = apply_floor((v @ r + alpha) / denom, config.prevalence_floor)
        delta = float(np.max(np.abs(new - pi)))
        pi = new
        if delta < config.tol:
            converged = True
            break
    return pi, it, converged


def em_prior_estimate(
    probabilities: ArrayLike, source_prior: ArrayLike, config: EMConfig | None = None
) -> PriorEstimate:
    """Estima π_t pelo EM de Saerens et al. (2002) a partir de probabilidades sem rótulo.

    Assume modelo calibrado no domínio de origem e label shift. Com menos de
    ``config.min_samples`` imagens devolve o prior do treino (método "source").
    """
    cfg = config or EMConfig()
    p = validate_probabilities(probabilities)
    pi_s = _validate_prior(source_prior, p.shape[1])
    n = p.shape[0]
    if n < cfg.min_samples:
        return PriorEstimate(
            pi_s.copy(),
            "source",
            n,
            converged=False,
            notes=(f"apenas {n} imagens (< {cfg.min_samples}); mantido o prior do treino",),
        )
    pi, it, conv = _em_fixed_point(p, pi_s, pi_s.copy(), cfg)
    notes = () if conv else (f"EM não convergiu em {cfg.max_iter} iterações",)
    return PriorEstimate(pi, "em", n, iterations=it, converged=conv, notes=notes)


# ---------------------------------------------------------------------------
# BBSE de Lipton, Wang & Smola (2018)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBSEConfig:
    """Parâmetros do BBSE.

    Attributes:
        min_samples: como no EM.
        max_condition: acima deste número de condição de C (forma condicional, ver
            :func:`confusion_condition_number`) a inversão direta amplifica o ruído
            amostral em até ``cond`` vezes (limite de Lipton et al., teorema 3, ∝ 1/sigma_min);
            passa-se a usar Tikhonov encolhendo para "sem mudança" (π_t=π_s), ver
            :func:`_bbse_solve`. 100 significa que um erro de 1 p.p. na distribuição de
            previsões pode virar até 100 p.p. — já inútil sem regularizar.
        prevalence_floor: ver :func:`apply_floor`.
    """

    min_samples: int = 30
    max_condition: float = 100.0
    prevalence_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.min_samples < 1 or self.max_condition <= 1:
            raise ValueError("parâmetros do BBSE fora do domínio")


def joint_confusion(
    labels: ArrayLike, predictions: ArrayLike, n_classes: int, *, soft: bool = False
) -> FloatArray:
    """Matriz de confusão conjunta C[i, j] = P(ŷ=i, y=j), estimada na validação.

    ``predictions`` pode ser vetor de classes previstas ou matriz (N, K) de probabilidades
    (usa-se o argmax, ou as próprias probabilidades se ``soft=True`` — a variante "soft" de
    Garg et al. 2020 / Azizzadenesheli et al. 2019, de menor variância; nesse caso o
    talhão também deve usar a média das probabilidades, não o histograma do argmax).
    """
    y = np.asarray(labels, dtype=np.int64).ravel()
    pred = np.asarray(predictions)
    if pred.ndim == 2:
        probs = validate_probabilities(pred, n_classes)
        onehot = probs if soft else np.eye(n_classes)[probs.argmax(axis=1)]
    else:
        if soft:
            raise ValueError("soft=True exige matriz de probabilidades")
        onehot = np.eye(n_classes)[pred.astype(np.int64).ravel()]
    if onehot.shape[0] != y.size:
        raise ValueError("labels e predictions com tamanhos diferentes")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError("rótulo fora de [0, K)")
    c = onehot.T @ np.eye(n_classes)[y]
    return c / y.size


def predicted_distribution(
    probabilities_or_predictions: ArrayLike, n_classes: int, *, soft: bool = False
) -> FloatArray:
    """μ[i] = P_t(ŷ=i) no talhão (histograma do argmax, ou média das probabilidades se soft)."""
    a = np.asarray(probabilities_or_predictions)
    if a.ndim == 2:
        p = validate_probabilities(a, n_classes)
        if soft:
            return p.mean(axis=0)
        return np.bincount(p.argmax(axis=1), minlength=n_classes) / p.shape[0]
    return np.bincount(a.astype(np.int64).ravel(), minlength=n_classes) / a.size


def confusion_condition_number(confusion: ArrayLike) -> float:
    """Número de condição de P(ŷ=i | y=j) = C[i, j]/π_s(j), a matriz efetivamente invertida.

    Usamos a forma condicional, e não a conjunta, porque o erro que importa é o de π_t:
    π_t = C_cond⁻¹·μ_t, e ||δπ_t|| <= ||C_cond⁻¹||·||δμ_t||. O número de condição da
    conjunta mistura confusão entre classes com o desbalanço do prior da validação
    (C = C_cond·diag(π_s)), que não prejudica a estimativa de π_t.
    """
    c = np.asarray(confusion, dtype=np.float64)
    pi_s = c.sum(axis=0)
    if np.any(pi_s <= 0):
        return math.inf
    sv = np.linalg.svd(c / pi_s, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf


def _bbse_solve(
    confusion: FloatArray, mu: FloatArray, config: BBSEConfig
) -> tuple[FloatArray, float, bool]:
    """Resolve C·w = μ (em termos de π_t) com Tikhonov se mal condicionada.

    Equivalência: C·w = μ com w_j = π_t(j)/π_s(j) é o mesmo que C_cond·π_t = μ. Resolvemos
    em π_t para que a regularização min ||C_cond·π - μ||² + λ||π - π_s||² encolha para "sem
    mudança de prevalência" (π_t = π_s), a hipótese nula segura, com
    λ = sigma_max²/max_condition (limita o número de condição efetivo).

    Returns:
        (π_t no simplex e com piso, número de condição, se regularizou)
    """
    pi_s = confusion.sum(axis=0)
    if np.any(pi_s <= 0):
        raise ValueError("classe ausente na validação: coluna de C zerada, BBSE indefinido")
    c_cond = confusion / pi_s
    sv = np.linalg.svd(c_cond, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    k = confusion.shape[0]
    regularized = cond > config.max_condition
    if regularized:
        lam = sv[0] ** 2 / config.max_condition
        a = c_cond.T @ c_cond + lam * np.eye(k)
        b = c_cond.T @ mu + lam * pi_s
        pi_t = np.linalg.solve(a, b)
    else:
        pi_t = np.linalg.solve(c_cond, mu)
    return apply_floor(project_to_simplex(pi_t), config.prevalence_floor), cond, regularized


def bbse_prior_estimate(
    probabilities_or_predictions: ArrayLike,
    confusion: ArrayLike,
    config: BBSEConfig | None = None,
    *,
    soft: bool = False,
) -> PriorEstimate:
    """Estima π_t por BBSE: w = C⁻¹·μ_t e π_t(j) = w_j·π_s(j), projetado no simplex.

    π_s é tomado das somas das colunas de C (prior da validação), que deve ter o mesmo prior
    do treino — caso contrário o posterior do modelo e C falam de priors diferentes.
    """
    cfg = config or BBSEConfig()
    c = np.asarray(confusion, dtype=np.float64)
    k = c.shape[0]
    if c.shape != (k, k) or np.any(c < 0):
        raise ValueError("matriz de confusão deve ser (K, K) não negativa")
    c = c / c.sum()
    pi_s = c.sum(axis=0)
    a = np.asarray(probabilities_or_predictions)
    n = a.shape[0] if a.ndim == 2 else a.size
    if n < cfg.min_samples:
        return PriorEstimate(
            pi_s.copy(),
            "source",
            n,
            converged=False,
            notes=(f"apenas {n} imagens (< {cfg.min_samples}); mantido o prior da validação",),
        )
    mu = predicted_distribution(a, k, soft=soft)
    pi_t, cond, reg = _bbse_solve(c, mu, cfg)
    notes: tuple[str, ...] = ()
    if reg:
        notes = (
            f"matriz de confusão mal condicionada (cond={_fmt(cond, 0)}); "
            "estimativa regularizada em direção ao prior da validação",
        )
    return PriorEstimate(pi_t, "bbse", n, bbse_weight=1.0, notes=notes)


# ---------------------------------------------------------------------------
# Diagnóstico e escolha/combinação
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probabilities: ArrayLike, labels: ArrayLike, n_bins: int = 15
) -> tuple[float, float]:
    """ECE top-label (Guo et al., 2017) e ECE por classe (Kull et al., 2019), bins iguais.

    O EM usa o vetor inteiro de probabilidades, não só o máximo; por isso a decisão usa o ECE
    por classe (média sobre classes de Σ_b |acc_b - conf_b|·n_b/N no problema um-contra-todos).
    Ambos têm viés positivo de amostra finita (~sqrt(bins/N)), relevante com validação pequena.
    """
    p = validate_probabilities(probabilities)
    y = np.asarray(labels, dtype=np.int64).ravel()
    n, k = p.shape
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _ece(conf: FloatArray, hit: FloatArray) -> float:
        b = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
        sc = np.bincount(b, weights=conf, minlength=n_bins)
        sh = np.bincount(b, weights=hit, minlength=n_bins)
        return float(np.sum(np.abs(sh - sc)) / n)

    top = _ece(p.max(axis=1), (p.argmax(axis=1) == y).astype(np.float64))
    classwise = float(np.mean([_ece(p[:, j], (y == j).astype(np.float64)) for j in range(k)]))
    return top, classwise


@dataclass(frozen=True)
class ValidationSummary:
    """Resumo da validação rotulada (prior do treino) usado para escolher o estimador."""

    confusion: FloatArray
    source_prior: FloatArray
    ece: float
    classwise_ece: float
    condition_number: float
    n_validation: int

    @classmethod
    def from_validation(
        cls, probabilities: ArrayLike, labels: ArrayLike, n_bins: int = 15
    ) -> ValidationSummary:
        p = validate_probabilities(probabilities)
        y = np.asarray(labels, dtype=np.int64).ravel()
        k = p.shape[1]
        c = joint_confusion(y, p, k)
        ece, cw = expected_calibration_error(p, y, n_bins)
        return cls(
            confusion=c,
            source_prior=np.bincount(y, minlength=k) / y.size,
            ece=ece,
            classwise_ece=cw,
            condition_number=confusion_condition_number(c),
            n_validation=int(y.size),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusion": self.confusion.tolist(),
            "source_prior": self.source_prior.tolist(),
            "ece": self.ece,
            "classwise_ece": self.classwise_ece,
            "condition_number": self.condition_number
            if math.isfinite(self.condition_number)
            else None,
            "n_validation": self.n_validation,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ValidationSummary:
        cond = d["condition_number"]
        return cls(
            confusion=np.asarray(d["confusion"], dtype=np.float64),
            source_prior=np.asarray(d["source_prior"], dtype=np.float64),
            ece=float(d["ece"]),
            classwise_ece=float(d["classwise_ece"]),
            condition_number=math.inf if cond is None else float(cond),
            n_validation=int(d["n_validation"]),
        )


@dataclass(frozen=True)
class SelectionConfig:
    """Limiares da escolha EM x BBSE.

    O ECE por classe tem viés positivo de amostra finita: medimos, com o classificador
    sintético EXATAMENTE calibrado dos testes (K=5, 15 bins), ECE ≈ 0,82/√N (0,026 com
    N=1000; 0,018 com N=2000; 0,013 com N=4000). Um limiar fixo confundiria "validação
    pequena" com "modelo descalibrado". Por isso compara-se o excesso
    ``ece - noise_coef/√N`` (``noise_coef`` = 1,0, um pouco acima do 0,82 medido):

    * excesso <= ``excess_good`` (0,01): EM puro;
    * excesso >= ``excess_bad`` (0,04): BBSE puro. Com temperatura T=0,5 (super-confiante)
      o excesso medido foi ~0,04 e o EM errou a prevalência sadia em 20 p.p.;
    * entre os dois, interpolação linear.
    """

    excess_good: float = 0.01
    excess_bad: float = 0.04
    noise_coef: float = 1.0
    max_condition: float = 100.0


def bbse_weight(summary: ValidationSummary, config: SelectionConfig | None = None) -> float:
    """Peso do BBSE na combinação convexa π = w·π_BBSE + (1-w)·π_EM.

    Regra: w cresce linearmente com o excesso de ECE por classe (ver
    :class:`SelectionConfig`) — w=0 é EM puro, mais eficiente se calibrado (Alexandari et al.
    2020; Garg et al. 2020); w=1 é BBSE puro, consistente sem calibração. Se C for mal
    condicionada o BBSE fica ruidoso e o peso é multiplicado por max_condition/cond (e o
    BBSE ainda é regularizado internamente).
    """
    cfg = config or SelectionConfig()
    floor = cfg.noise_coef / math.sqrt(max(summary.n_validation, 1))
    excess = summary.classwise_ece - floor
    span = cfg.excess_bad - cfg.excess_good
    w = min(max((excess - cfg.excess_good) / span, 0.0), 1.0)
    if summary.condition_number > cfg.max_condition:
        w *= cfg.max_condition / summary.condition_number
    return float(w)


def estimate_target_prior(
    probabilities: ArrayLike,
    source_prior: ArrayLike,
    summary: ValidationSummary | None = None,
    *,
    em_config: EMConfig | None = None,
    bbse_config: BBSEConfig | None = None,
    selection: SelectionConfig | None = None,
) -> PriorEstimate:
    """Estima π_t escolhendo/combinando EM e BBSE conforme o diagnóstico da validação.

    Sem ``summary`` usa só EM (não há como montar C nem medir calibração).
    """
    em_cfg = em_config or EMConfig()
    p = validate_probabilities(probabilities)
    em = em_prior_estimate(p, source_prior, em_cfg)
    if summary is None or em.method == "source":
        return em
    w = bbse_weight(summary, selection)
    if w <= 0.0:
        return em
    bb = bbse_prior_estimate(p, summary.confusion, bbse_config or BBSEConfig())
    if w >= 1.0:
        return bb
    pi = apply_floor(w * bb.prevalences + (1.0 - w) * em.prevalences, em_cfg.prevalence_floor)
    return PriorEstimate(
        pi,
        "combined",
        em.n_samples,
        iterations=em.iterations,
        converged=em.converged,
        bbse_weight=w,
        notes=em.notes + bb.notes,
    )


# ---------------------------------------------------------------------------
# Estado por talhão com esquecimento exponencial
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerConfig:
    """Configuração do estado por talhão.

    Attributes:
        half_life_samples: meia-vida em número de imagens. Após esse número de imagens novas,
            o peso das antigas cai à metade. 2000 imagens ~ poucas passadas do rover num
            talhão; em regime o tamanho efetivo de amostra é ~half_life/ln 2 ≈ 2900.
        half_life_seconds: meia-vida em tempo (opcional), para esquecer entre estações mesmo
            sem imagens novas (ex.: 30 dias). Os dois esquecimentos se multiplicam.
        method: "em", "bbse" ou "auto" (combinação por :func:`bbse_weight`; exige summary).
        buffer_size: quantas probabilidades recentes são guardadas exatamente. As que saem do
            buffer vão para um resumo em células (classe prevista x faixa de confiança), que
            guarda o vetor médio de probabilidades e o peso de cada célula.
        summary_bins: faixas de confiança por classe no resumo (Kxbins células).

    Por que não guardar só Σ responsabilidades (EM online "puro", Cappé & Moulines, 2009):
    as responsabilidades ficam congeladas com o π da época; quando classes se confundem
    (Sigatoka x sadia) o viés não some — medimos 4-7 p.p. de erro residual em testes
    sintéticos após uma mudança de estação. Com buffer + células, o EM re-avalia TODAS as
    amostras (exatas ou centróides) sob o π corrente a cada lote. O erro de quantização das
    células é pequeno porque elas separam por classe e confiança, e porque o resumo pesa só
    0,5^(buffer/meia-vida) do total (≈25% no padrão). Custo no JSON: ~(buffer + K·bins)·K
    números por talhão.
    """

    half_life_samples: float | None = 2000.0
    half_life_seconds: float | None = None
    buffer_size: int = 4000
    summary_bins: int = 20
    method: Literal["em", "bbse", "auto"] = "auto"
    em: EMConfig = field(default_factory=EMConfig)
    bbse: BBSEConfig = field(default_factory=BBSEConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def __post_init__(self) -> None:
        for hl in (self.half_life_samples, self.half_life_seconds):
            if hl is not None and hl <= 0:
                raise ValueError("meia-vida deve ser > 0")
        if self.half_life_samples is not None:
            n_eff = self.half_life_samples / math.log(2.0)
            if n_eff < self.em.min_samples:
                raise ValueError("meia-vida curta demais: nunca atingiria min_samples")
        if self.buffer_size < self.em.min_samples or self.summary_bins < 1:
            raise ValueError("buffer_size deve ser >= min_samples e summary_bins >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TrackerConfig:
        return cls(
            half_life_samples=d["half_life_samples"],
            half_life_seconds=d["half_life_seconds"],
            buffer_size=int(d["buffer_size"]),
            summary_bins=int(d["summary_bins"]),
            method=d["method"],
            em=EMConfig(**d["em"]),
            bbse=BBSEConfig(**d["bbse"]),
            selection=SelectionConfig(**d["selection"]),
        )


@dataclass
class FieldState:
    """Estado decaído de um talhão.

    ``buffer``/``buffer_weights``: probabilidades recentes do modelo e pesos de esquecimento;
    ``cell_sums``/``cell_weights``: resumo das amostras que saíram do buffer (soma ponderada
    dos vetores de probabilidade e soma dos pesos, por célula); ``pred_counts``: histograma
    decaído do argmax (BBSE); ``prevalences_em``: π do EM na última atualização.
    """

    buffer: FloatArray
    buffer_weights: FloatArray
    cell_sums: FloatArray
    cell_weights: FloatArray
    pred_counts: FloatArray
    prevalences_em: FloatArray
    n_seen: int = 0
    last_update: float | None = None

    @classmethod
    def empty(cls, source_prior: FloatArray, n_cells: int) -> FieldState:
        k = source_prior.size
        return cls(
            np.zeros((0, k)),
            np.zeros(0),
            np.zeros((n_cells, k)),
            np.zeros(n_cells),
            np.zeros(k),
            source_prior.copy(),
        )

    @property
    def n_eff(self) -> float:
        """Tamanho efetivo: soma dos pesos de esquecimento (buffer + resumo)."""
        return float(self.cell_weights.sum() + self.buffer_weights.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer.tolist(),
            "buffer_weights": self.buffer_weights.tolist(),
            "cell_sums": self.cell_sums.tolist(),
            "cell_weights": self.cell_weights.tolist(),
            "pred_counts": self.pred_counts.tolist(),
            "prevalences_em": self.prevalences_em.tolist(),
            "n_seen": self.n_seen,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FieldState:
        k = len(d["pred_counts"])
        return cls(
            buffer=np.asarray(d["buffer"], dtype=np.float64).reshape(-1, k),
            buffer_weights=np.asarray(d["buffer_weights"], dtype=np.float64),
            cell_sums=np.asarray(d["cell_sums"], dtype=np.float64).reshape(-1, k),
            cell_weights=np.asarray(d["cell_weights"], dtype=np.float64),
            pred_counts=np.asarray(d["pred_counts"], dtype=np.float64),
            prevalences_em=np.asarray(d["prevalences_em"], dtype=np.float64),
            n_seen=int(d["n_seen"]),
            last_update=d["last_update"],
        )


DIVERGENCE_LIMIT = 0.05
"""Distância de variação total entre o EM e a contagem simples acima da qual a
prevalência do talhão é marcada como não confiável.

Escolhido em 30/09/2026 sobre os recortes de campo do PlantSeg (banana e citros,
versões recomendada e comercial, com e sem TTA — 8 casos): nos 6 em que o EM
errava mais de 0,13 da prevalência real, a discordância ficou entre 0,086 e 0,199;
nos 2 em que acertava (erro até 0,03), em até 0,022. É uma regra medida em poucos
casos, não uma garantia: serve para não mostrar ao produtor, como certeza, um
número que o próprio sistema tem motivo para duvidar.
"""


class FieldPriorStore:
    """Estimativa corrente de prevalências por talhão, com esquecimento exponencial.

    Uso::

        store = FieldPriorStore(source_prior, summary=ValidationSummary.from_validation(...))
        store.update("talhao-07", probs_lote)
        probs_corrigidas = store.adjust("talhao-07", probs)
        store.save("estado/priors.json")

    Thread-safe (um lock), pois a API FastAPI pode atender requisições concorrentes.
    """

    def __init__(
        self,
        source_prior: ArrayLike,
        config: TrackerConfig | None = None,
        summary: ValidationSummary | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.source_prior = _validate_prior(source_prior)
        self.n_classes = self.source_prior.size
        self.config = config or TrackerConfig()
        self.summary = summary
        if class_names is not None and len(class_names) != self.n_classes:
            raise ValueError("class_names com tamanho diferente de K")
        self.class_names = list(class_names) if class_names is not None else None
        if summary is not None and summary.confusion.shape != (self.n_classes,) * 2:
            raise ValueError("summary com número de classes diferente")
        if self.config.method in ("bbse",) and summary is None:
            raise ValueError("método 'bbse' exige ValidationSummary")
        self._fields: dict[str, FieldState] = {}
        self._lock = threading.RLock()

    # -- esquecimento -----------------------------------------------------

    def _decay(self, n_new: int, dt: float | None) -> float:
        d = 1.0
        if self.config.half_life_samples is not None:
            d *= 0.5 ** (n_new / self.config.half_life_samples)
        if self.config.half_life_seconds is not None and dt is not None and dt > 0:
            d *= 0.5 ** (dt / self.config.half_life_seconds)
        return d

    # -- API --------------------------------------------------------------

    def update(
        self, field_id: str, probabilities_batch: ArrayLike, timestamp: float | None = None
    ) -> FloatArray:
        """Incorpora um lote de probabilidades (do modelo, NÃO corrigidas) e devolve os priors.

        1. Decai pesos antigos (por número de imagens novas e, se configurado, por tempo).
        2. Move o excedente do buffer para o resumo em células.
        3. Re-roda o EM ponderado sobre buffer + centróides das células, partindo do π
           anterior (poucas iterações em regime).
        """
        p = validate_probabilities(probabilities_batch, self.n_classes)
        ts = time.time() if timestamp is None else float(timestamp)
        n_cells = self.n_classes * self.config.summary_bins
        with self._lock:
            st = self._fields.get(field_id)
            if st is None:
                st = FieldState.empty(self.source_prior, n_cells)
                self._fields[field_id] = st
            dt = None if st.last_update is None else ts - st.last_update
            d = self._decay(p.shape[0], dt)
            buf = np.vstack([st.buffer, p])
            wts = np.concatenate([st.buffer_weights * d, np.ones(p.shape[0])])
            cell_sums, cell_weights = st.cell_sums * d, st.cell_weights * d
            excess = buf.shape[0] - self.config.buffer_size
            if excess > 0:
                cells = self._cell_index(buf[:excess])
                cell_weights += np.bincount(cells, weights=wts[:excess], minlength=n_cells)
                np.add.at(cell_sums, cells, buf[:excess] * wts[:excess, None])
                buf, wts = buf[excess:], wts[excess:]
            occupied = cell_weights > 1e-12
            pts = np.vstack([buf, cell_sums[occupied] / cell_weights[occupied, None]])
            pts_w = np.concatenate([wts, cell_weights[occupied]])
            pi, _, _ = _em_fixed_point(
                pts, self.source_prior, st.prevalences_em, self.config.em, pts_w
            )
            st.buffer, st.buffer_weights = buf, wts
            st.cell_sums, st.cell_weights = cell_sums, cell_weights
            st.prevalences_em = pi
            st.pred_counts = st.pred_counts * d + np.bincount(
                p.argmax(axis=1), minlength=self.n_classes
            )
            st.n_seen += p.shape[0]
            st.last_update = ts
            return self._priors_locked(st)

    def _cell_index(self, probabilities: FloatArray) -> IntArray:
        """Célula = classe prevista x faixa (uniforme em [1/K, 1]) da confiança máxima."""
        b = self.config.summary_bins
        k = self.n_classes
        conf = probabilities.max(axis=1)
        rel = (conf - 1.0 / k) / (1.0 - 1.0 / k)
        bins = np.clip((rel * b).astype(np.int64), 0, b - 1)
        return probabilities.argmax(axis=1) * b + bins

    def priors(self, field_id: str) -> FloatArray:
        """Prevalências correntes do talhão (prior do treino se desconhecido ou pouco visto)."""
        with self._lock:
            st = self._fields.get(field_id)
            if st is None:
                return self.source_prior.copy()
            return self._priors_locked(st)

    def predicted_shares(self, field_id: str) -> FloatArray | None:
        """A fração de cada classe entre os palpites do talhão, com o mesmo esquecimento.

        É a "contagem simples". Ela é enviesada pela confusão do modelo, mas não
        amplifica nada — e por isso serve de testemunha para o EM: quando os dois
        se afastam, o modelo não está calibrado neste talhão (ver
        :data:`DIVERGENCE_LIMIT`). ``None`` sem nada visto.
        """
        with self._lock:
            st = self._fields.get(field_id)
            if st is None or st.pred_counts.sum() <= 0:
                return None
            return st.pred_counts / st.pred_counts.sum()

    def divergence(self, field_id: str) -> float | None:
        """Distância de variação total entre a prevalência estimada e a contagem simples.

        ``None`` enquanto o talhão ainda usa o prior do treino (pouco visto).
        """
        with self._lock:
            st = self._fields.get(field_id)
            if st is None or st.n_eff < self.config.em.min_samples or st.pred_counts.sum() <= 0:
                return None
            shares = st.pred_counts / st.pred_counts.sum()
            return float(np.abs(self._priors_locked(st) - shares).sum() / 2)

    def estimate(self, field_id: str) -> PriorEstimate:
        """Como :meth:`priors`, mas com método, peso do BBSE e tamanho efetivo."""
        with self._lock:
            st = self._fields.get(field_id)
            if st is None or st.n_eff < self.config.em.min_samples:
                n = 0.0 if st is None else st.n_eff
                return PriorEstimate(self.source_prior.copy(), "source", n, converged=False)
            w = self._bbse_w()
            pi = self._priors_locked(st)
            method: Method = "em" if w == 0 else ("bbse" if w == 1 else "combined")
            return PriorEstimate(pi, method, st.n_eff, bbse_weight=w)

    def adjust(self, field_id: str, probabilities: ArrayLike) -> FloatArray:
        """Corrige posteriores para o prior corrente do talhão (não atualiza o estado)."""
        return adjust_posteriors(probabilities, self.source_prior, self.priors(field_id))

    def field_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._fields)

    def reset(self, field_id: str) -> None:
        with self._lock:
            self._fields.pop(field_id, None)

    # -- internos ---------------------------------------------------------

    def _bbse_w(self) -> float:
        if self.summary is None or self.config.method == "em":
            return 0.0
        if self.config.method == "bbse":
            return 1.0
        return bbse_weight(self.summary, self.config.selection)

    def _priors_locked(self, st: FieldState) -> FloatArray:
        if st.n_eff < self.config.em.min_samples:
            return self.source_prior.copy()
        pi_em = st.prevalences_em
        w = self._bbse_w()
        if w == 0.0 or self.summary is None:
            return pi_em
        c = self.summary.confusion / self.summary.confusion.sum()
        pi_bb, _, _ = _bbse_solve(c, st.pred_counts / st.pred_counts.sum(), self.config.bbse)
        return apply_floor(w * pi_bb + (1 - w) * pi_em, self.config.em.prevalence_floor)

    # -- serialização -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format_version": STATE_FORMAT_VERSION,
                "source_prior": self.source_prior.tolist(),
                "class_names": self.class_names,
                "config": self.config.to_dict(),
                "summary": None if self.summary is None else self.summary.to_dict(),
                "fields": {fid: st.to_dict() for fid, st in self._fields.items()},
            }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FieldPriorStore:
        if d.get("format_version") != STATE_FORMAT_VERSION:
            raise ValueError(f"versão de estado não suportada: {d.get('format_version')}")
        store = cls(
            d["source_prior"],
            TrackerConfig.from_dict(d["config"]),
            None if d["summary"] is None else ValidationSummary.from_dict(d["summary"]),
            d["class_names"],
        )
        store._fields = {fid: FieldState.from_dict(s) for fid, s in d["fields"].items()}
        return store

    def save(self, path: str | os.PathLike[str]) -> None:
        """Grava JSON de forma atômica: temporário no mesmo diretório + fsync + os.replace.

        Assim um desligamento abrupto do rover (bateria) deixa o arquivo antigo OU o novo,
        nunca um JSON truncado. O fsync do diretório garante que o rename persista (POSIX).
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=1, allow_nan=False)
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        if hasattr(os, "O_DIRECTORY"):
            dfd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> FieldPriorStore:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Testes estatísticos (sem scipy)
# ---------------------------------------------------------------------------


def kolmogorov_sf(lam: float) -> float:
    """Q_KS(λ) = 2 Σ_{k>=1} (-1)^{k-1} exp(-2k²λ²): P(√n·D > λ) assintótico.

    Para λ < 0,2 a série converge mal e Q ≈ 1 (Numerical Recipes, 14.3).
    """
    if lam < 0.2:
        return 1.0
    s = 0.0
    for k in range(1, 101):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        s += term
        if abs(term) < 1e-12:
            break
    return min(max(s, 0.0), 1.0)


def _weighted_ecdf(sorted_x: FloatArray, cum_w: FloatArray, grid: FloatArray) -> FloatArray:
    idx = np.searchsorted(sorted_x, grid, side="right")
    return np.where(idx > 0, cum_w[np.maximum(idx - 1, 0)], 0.0)


def ks_2samp(
    reference: ArrayLike, sample: ArrayLike, reference_weights: ArrayLike | None = None
) -> tuple[float, float]:
    """KS de duas amostras: D = sup |F_ref - F_amostra| e p-valor assintótico.

    p-valor com a correção de Stephens (1970): λ = (√n_e + 0,12 + 0,11/√n_e)·D, n_e =
    n·m/(n+m). Com pesos na referência (reponderação por label shift) usa-se o tamanho
    efetivo de Kish, (Σw)²/Σw², no lugar de n.
    """
    x = np.asarray(reference, dtype=np.float64).ravel()
    y = np.asarray(sample, dtype=np.float64).ravel()
    wx = np.ones_like(x) if reference_weights is None else np.asarray(reference_weights, float)
    order = np.argsort(x, kind="stable")
    xs, cw = x[order], np.cumsum(wx[order])
    cw = cw / cw[-1]
    ys = np.sort(y)
    grid = np.concatenate([xs, ys])
    fx = _weighted_ecdf(xs, cw, grid)
    fy = np.searchsorted(ys, grid, side="right") / ys.size
    d = float(np.max(np.abs(fx - fy)))
    n = float(wx.sum() ** 2 / np.sum(wx**2))
    m = float(ys.size)
    en = math.sqrt(n * m / (n + m))
    return d, kolmogorov_sf((en + 0.12 + 0.11 / en) * d)


def _gammaincc(a: float, x: float) -> float:
    """Gama incompleta superior regularizada Q(a, x) (Numerical Recipes, 6.2)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, delta = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * 1e-14:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(x: float, df: int) -> float:
    """P(χ²_df > x)."""
    if df <= 0:
        return 1.0
    return _gammaincc(df / 2.0, x / 2.0)


def chi2_homogeneity(
    reference_counts: ArrayLike, sample_counts: ArrayLike
) -> tuple[float, float, int]:
    """Qui-quadrado de homogeneidade 2xK (referência x janela). Devolve (X², p, gl).

    Classes com contagem total zero são descartadas (não contribuem e reduziriam gl à toa).
    """
    r = np.asarray(reference_counts, dtype=np.float64)
    s = np.asarray(sample_counts, dtype=np.float64)
    keep = (r + s) > 0
    r, s = r[keep], s[keep]
    table = np.vstack([r, s])
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    stat = float(np.sum((table - expected) ** 2 / expected))
    df = int(keep.sum()) - 1
    return stat, chi2_sf(stat, df), df


def psi(expected_props: ArrayLike, actual_counts: ArrayLike, pseudo_count: float = 0.5) -> float:
    """Population Stability Index: Σ (a - e)·ln(a/e).

    Bins vazios recebem ``pseudo_count`` (suavização de Laplace/Jeffreys) para evitar ln(0).
    Atenção ao viés de amostra finita: sob H0, PSI ≈ χ²_{B-1}·(1/n + 1/m), média ≈
    (B-1)/n com referência grande — com B=10 e janela de 300 isso dá 0,03, abaixo de 0,1;
    com janela de 50 daria 0,18 (alarme falso sistemático). Daí ``min_window`` do monitor.
    """
    e = np.asarray(expected_props, dtype=np.float64)
    e = np.clip(e, 1e-6, None)
    e = e / e.sum()
    a = np.asarray(actual_counts, dtype=np.float64)
    a = (a + pseudo_count) / (a.sum() + pseudo_count * a.size)
    return float(np.sum((a - e) * np.log(a / e)))


def _rbf_kernel(x: FloatArray, y: FloatArray, gamma: float) -> FloatArray:
    sq = np.sum(x**2, 1)[:, None] + np.sum(y**2, 1)[None, :] - 2.0 * x @ y.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def median_heuristic_gamma(z: FloatArray) -> float:
    """gama = 1/(2·mediana²) das distâncias par-a-par (Gretton et al., 2012, seção 8)."""
    sq = np.sum(z**2, 1)[:, None] + np.sum(z**2, 1)[None, :] - 2.0 * z @ z.T
    iu = np.triu_indices(z.shape[0], k=1)
    med2 = float(np.median(np.maximum(sq[iu], 0.0)))
    return 1.0 / (2.0 * med2) if med2 > 0 else 1.0


def mmd_permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_permutations: int = 1000,
    gamma: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """MMD² não enviesado (estatística U) com kernel RBF e p-valor por permutação.

    Gretton, Borgwardt, Rasch, Schölkopf & Smola (2012), "A kernel two-sample test", eq. 3.
    A matriz de kernel do conjunto unido é calculada uma vez; cada permutação só reindexa.
    p = (1 + #{MMD²_perm >= MMD²_obs}) / (1 + n_perm), válido em amostra finita. Custo O(n²)
    por permutação: use amostras pequenas (o monitor subamostra para <= 150 por lado).
    """
    gen = rng or np.random.default_rng()
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    z = np.vstack([xa, ya])
    n, m = xa.shape[0], ya.shape[0]
    g = median_heuristic_gamma(z) if gamma is None else gamma
    kmat = _rbf_kernel(z, z, g)
    np.fill_diagonal(kmat, 0.0)

    def _mmd2(ix: IntArray, iy: IntArray) -> float:
        kxx = kmat[np.ix_(ix, ix)].sum() / (n * (n - 1))
        kyy = kmat[np.ix_(iy, iy)].sum() / (m * (m - 1))
        kxy = kmat[np.ix_(ix, iy)].sum() / (n * m)
        return float(kxx + kyy - 2.0 * kxy)

    idx = np.arange(n + m)
    obs = _mmd2(idx[:n], idx[n:])
    exceed = 0
    for _ in range(n_permutations):
        perm = gen.permutation(n + m)
        if _mmd2(perm[:n], perm[n:]) >= obs:
            exceed += 1
    return obs, (1.0 + exceed) / (1.0 + n_permutations)


# ---------------------------------------------------------------------------
# Monitor de deriva
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftThresholds:
    """Limiares do monitor.

    Racional (além da convenção PSI 0,1/0,25 — Siddiqi, 2006, "Credit Risk Scorecards"):

    * p-valores 0,01 (atenção) e 0,001 (deriva): o monitor roda ~6 testes por avaliação;
      0,01 por teste mantém o alarme falso de "atenção" em ~5% por avaliação (Bonferroni
      aproximado) e 0,001 torna "deriva" espúria rara (<1%).
    * Tamanho de efeito mínimo: com janelas grandes qualquer diferença trivial vira p≈0.
      Exigimos D do KS >= 0,10/0,20 (10/20 p.p. de diferença máxima entre CDFs) e distância
      de variação total >= 0,10/0,20 no histograma de classes — mudanças menores não mudam a
      decisão agronômica.
    * MMD: não há escala natural para MMD², então só o p-valor da permutação conta; o teste já
      é pouco sensível com 150 embeddings por lado, o que o torna conservador.
    * ``min_window``: abaixo de 300 imagens o viés do PSI sob H0 (~(B-1)/n) se aproxima de 0,1;
      ver :func:`psi`.
    * ``novelty_threshold``: novelty_score > 1 significa "estranho ao treino" (definição do
      projeto); relatamos a fração acima dele como motivo legível.
    """

    alpha_warn: float = 0.01
    alpha_drift: float = 0.001
    ks_effect_warn: float = 0.10
    ks_effect_drift: float = 0.20
    psi_warn: float = 0.10
    psi_drift: float = 0.25
    tv_effect_warn: float = 0.10
    tv_effect_drift: float = 0.20
    min_window: int = 300
    novelty_threshold: float = 1.0
    mmd_max_per_side: int = 150
    mmd_permutations: int = 1000


@dataclass(frozen=True)
class TestResult:
    """Resultado de um teste individual do monitor."""

    name: str
    statistic: float
    p_value: float | None
    level: Level
    reason: str


@dataclass(frozen=True)
class DriftReport:
    """Relatório agregado: nível = pior nível entre os testes."""

    level: Level
    reasons: tuple[str, ...]
    tests: tuple[TestResult, ...]
    n_window: int
    reweighted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "reasons": list(self.reasons),
            "n_window": self.n_window,
            "reweighted": self.reweighted,
            "tests": [asdict(t) for t in self.tests],
        }


def _worst(levels: Sequence[Level]) -> Level:
    return max(levels, key=lambda lv: _LEVEL_RANK[lv], default=LEVEL_OK)


@dataclass(frozen=True)
class ReferenceProfile:
    """Distribuições de referência (validação com o prior do treino).

    ``labels`` é opcional; com ele o monitor pode reponderar a referência pelo prior estimado
    do talhão. Os bins do PSI são fixados aqui (decis da referência), como manda a convenção:
    bins recalculados na janela esconderiam a deriva.
    """

    confidence: FloatArray
    novelty: FloatArray
    predicted: IntArray
    n_classes: int
    labels: IntArray | None = None
    embeddings: FloatArray | None = None
    confidence_edges: FloatArray = field(default_factory=lambda: np.zeros(0))
    novelty_edges: FloatArray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def build(
        cls,
        confidence: ArrayLike,
        novelty: ArrayLike,
        predicted: ArrayLike,
        n_classes: int,
        *,
        labels: ArrayLike | None = None,
        embeddings: ArrayLike | None = None,
        n_bins: int = 10,
    ) -> ReferenceProfile:
        conf = np.asarray(confidence, dtype=np.float64).ravel()
        nov = np.asarray(novelty, dtype=np.float64).ravel()
        pred = np.asarray(predicted, dtype=np.int64).ravel()
        if not conf.size == nov.size == pred.size:
            raise ValueError("confidence, novelty e predicted com tamanhos diferentes")
        lab = None if labels is None else np.asarray(labels, dtype=np.int64).ravel()
        if lab is not None and lab.size != conf.size:
            raise ValueError("labels com tamanho diferente")
        emb = None if embeddings is None else np.asarray(embeddings, dtype=np.float64)
        if emb is not None and emb.shape[0] != conf.size:
            raise ValueError("embeddings com número de linhas diferente")
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        return cls(
            confidence=conf,
            novelty=nov,
            predicted=pred,
            n_classes=n_classes,
            labels=lab,
            embeddings=emb,
            confidence_edges=np.unique(np.quantile(conf, qs)),
            novelty_edges=np.unique(np.quantile(nov, qs)),
        )

    def weights_for(self, target_prior: ArrayLike | None) -> FloatArray | None:
        """Pesos de importância π_t(y)/π_s(y) por exemplo da referência (label shift)."""
        if target_prior is None or self.labels is None:
            return None
        pi_t = np.asarray(target_prior, dtype=np.float64).ravel()
        pi_t = pi_t / pi_t.sum()
        pi_s = np.bincount(self.labels, minlength=self.n_classes) / self.labels.size
        ratio = np.divide(pi_t, pi_s, out=np.zeros_like(pi_t), where=pi_s > 0)
        return ratio[self.labels]


def _hist(x: FloatArray, edges: FloatArray, w: FloatArray | None = None) -> FloatArray:
    b = np.searchsorted(edges, x, side="right")
    return np.bincount(b, weights=w, minlength=edges.size + 1).astype(np.float64)


def _level_from(
    p: float, effect: float, th: DriftThresholds, eff_warn: float, eff_drift: float
) -> Level:
    if p < th.alpha_drift and effect >= eff_drift:
        return LEVEL_DRIFT
    if p < th.alpha_warn and effect >= eff_warn:
        return LEVEL_WARN
    return LEVEL_OK


def _wmean(x: FloatArray, w: FloatArray | None) -> float:
    return float(np.average(x, weights=w))


def check_drift(
    reference: ReferenceProfile,
    confidence: ArrayLike,
    novelty: ArrayLike,
    predicted: ArrayLike,
    embeddings: ArrayLike | None = None,
    *,
    target_prior: ArrayLike | None = None,
    thresholds: DriftThresholds | None = None,
    rng: np.random.Generator | None = None,
) -> DriftReport:
    """Compara a janela recente de um talhão com a referência.

    Testes: KS e PSI sobre confiança máxima e novelty_score; qui-quadrado no histograma de
    classes previstas; MMD (RBF, mediana) nos embeddings, se fornecidos.

    ``target_prior``: prevalência estimada do talhão (ex.: ``store.priors(fid)``). Se a
    referência tiver rótulos, ela é reponderada por π_t/π_s, e só sobra o que a mudança de
    prevalência não explica. Sem reponderação, uma mudança de prevalência legítima também
    altera o histograma de classes (e um pouco a confiança); por isso, nesse caso, o teste de
    classes sozinho é limitado a "atenção": é o sintoma esperado de label shift, que a
    correção de prior já trata, e não prova que o talhão deixou de parecer o treino.
    """
    th = thresholds or DriftThresholds()
    gen = rng or np.random.default_rng(0)
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    nov = np.asarray(novelty, dtype=np.float64).ravel()
    pred = np.asarray(predicted, dtype=np.int64).ravel()
    n = conf.size
    if not n == nov.size == pred.size:
        raise ValueError("confidence, novelty e predicted com tamanhos diferentes")
    if n < th.min_window:
        return DriftReport(
            LEVEL_OK,
            (f"janela com {n} imagens (< {th.min_window}); testes não executados",),
            (),
            n,
        )
    w = reference.weights_for(target_prior)
    reweighted = w is not None
    tests: list[TestResult] = []

    # --- KS + PSI na confiança máxima ------------------------------------------------
    d, p = ks_2samp(reference.confidence, conf, w)
    lv = _level_from(p, d, th, th.ks_effect_warn, th.ks_effect_drift)
    m_ref, m_win = _wmean(reference.confidence, w), float(conf.mean())
    verb = "caiu" if m_win < m_ref else "subiu"
    tests.append(
        TestResult(
            "ks_confianca",
            d,
            p,
            lv,
            f"confiança média {verb} de {_fmt(m_ref)} para {_fmt(m_win)} "
            f"(KS D={_fmt(d)}, {_fmt_p(p)})",
        )
    )

    e = _hist(reference.confidence, reference.confidence_edges, w)
    v = psi(e / e.sum(), _hist(conf, reference.confidence_edges))
    lv = LEVEL_DRIFT if v >= th.psi_drift else LEVEL_WARN if v >= th.psi_warn else LEVEL_OK
    tests.append(TestResult("psi_confianca", v, None, lv, f"PSI da confiança {_fmt(v)}"))

    # --- KS + PSI no novelty_score ----------------------------------------------------
    d, p = ks_2samp(reference.novelty, nov, w)
    lv = _level_from(p, d, th, th.ks_effect_warn, th.ks_effect_drift)
    f_ref = _wmean((reference.novelty > th.novelty_threshold).astype(float), w)
    f_win = float(np.mean(nov > th.novelty_threshold))
    verb = "subiu" if f_win > f_ref else "caiu"
    tests.append(
        TestResult(
            "ks_novidade",
            d,
            p,
            lv,
            f"fração com novidade > {_fmt(th.novelty_threshold, 1)} {verb} de {_fmt_pct(f_ref)} "
            f"para {_fmt_pct(f_win)} (KS D={_fmt(d)}, {_fmt_p(p)})",
        )
    )

    e = _hist(reference.novelty, reference.novelty_edges, w)
    v = psi(e / e.sum(), _hist(nov, reference.novelty_edges))
    lv = LEVEL_DRIFT if v >= th.psi_drift else LEVEL_WARN if v >= th.psi_warn else LEVEL_OK
    tests.append(TestResult("psi_novidade", v, None, lv, f"PSI da novidade {_fmt(v)}"))

    # --- qui-quadrado nas classes previstas ------------------------------------------
    k = reference.n_classes
    ref_hist = np.bincount(reference.predicted, weights=w, minlength=k).astype(np.float64)
    n_ref_eff = reference.predicted.size if w is None else float(w.sum() ** 2 / np.sum(w**2))
    ref_counts = ref_hist / ref_hist.sum() * n_ref_eff
    win_counts = np.bincount(pred, minlength=k).astype(np.float64)
    x2, p, df = chi2_homogeneity(ref_counts, win_counts)
    tv = 0.5 * float(np.abs(ref_hist / ref_hist.sum() - win_counts / n).sum())
    lv = _level_from(p, tv, th, th.tv_effect_warn, th.tv_effect_drift)
    if not reweighted and lv == LEVEL_DRIFT:
        lv = LEVEL_WARN
    tests.append(
        TestResult(
            "qui2_classes",
            x2,
            p,
            lv,
            f"distribuição de classes previstas mudou (variação total {_fmt(tv)}, "
            f"χ²={_fmt(x2, 1)}, gl={df}, {_fmt_p(p)})"
            + ("" if reweighted else "; compatível com mudança de prevalência"),
        )
    )

    # --- MMD nos embeddings -----------------------------------------------------------
    if embeddings is not None and reference.embeddings is not None:
        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.shape[0] != n:
            raise ValueError("embeddings com número de linhas diferente da janela")
        s = th.mmd_max_per_side
        ref_n = reference.embeddings.shape[0]
        pr = None if w is None else w / w.sum()
        n_nonzero = ref_n if pr is None else int(np.count_nonzero(pr))
        ri = gen.choice(ref_n, size=min(s, n_nonzero), replace=False, p=pr)
        wi = gen.choice(n, size=min(s, n), replace=False)
        mmd2, p = mmd_permutation_test(
            reference.embeddings[ri], emb[wi], n_permutations=th.mmd_permutations, rng=gen
        )
        lv = LEVEL_DRIFT if p < th.alpha_drift else LEVEL_WARN if p < th.alpha_warn else LEVEL_OK
        tests.append(
            TestResult(
                "mmd_embeddings",
                mmd2,
                p,
                lv,
                f"embeddings diferentes da referência (MMD²={_fmt(mmd2, 4)}, {_fmt_p(p)})",
            )
        )

    level = _worst([t.level for t in tests])
    reasons = tuple(t.reason for t in tests if t.level != LEVEL_OK)
    if level == LEVEL_OK:
        reasons = ("janela compatível com a referência",)
    return DriftReport(level, reasons, tuple(tests), n, reweighted)


class DriftMonitor:
    """Janela deslizante por talhão sobre :func:`check_drift`.

    Guarda as últimas ``window_size`` imagens (confiança, novidade, classe e, opcionalmente,
    embedding). Embeddings são opcionais por imagem: o MMD só roda quando todas as imagens da
    janela os têm.
    """

    def __init__(
        self,
        reference: ReferenceProfile,
        window_size: int = 500,
        thresholds: DriftThresholds | None = None,
    ) -> None:
        self.reference = reference
        self.thresholds = thresholds or DriftThresholds()
        self._window: deque[tuple[float, float, int, FloatArray | None]] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def observe(
        self,
        probabilities: ArrayLike,
        novelty: ArrayLike,
        embeddings: ArrayLike | None = None,
    ) -> None:
        p = validate_probabilities(probabilities, self.reference.n_classes)
        nov = np.asarray(novelty, dtype=np.float64).ravel()
        emb = None if embeddings is None else np.asarray(embeddings, dtype=np.float64)
        with self._lock:
            for i in range(p.shape[0]):
                e_i = None if emb is None else emb[i]
                self._window.append((float(p[i].max()), float(nov[i]), int(p[i].argmax()), e_i))

    def evaluate(
        self, target_prior: ArrayLike | None = None, rng: np.random.Generator | None = None
    ) -> DriftReport:
        with self._lock:
            items = list(self._window)
        conf = np.array([it[0] for it in items])
        nov = np.array([it[1] for it in items])
        pred = np.array([it[2] for it in items], dtype=np.int64)
        embs = [it[3] for it in items]
        emb = np.vstack(embs) if items and all(e is not None for e in embs) else None
        return check_drift(
            self.reference,
            conf,
            nov,
            pred,
            emb,
            target_prior=target_prior,
            thresholds=self.thresholds,
            rng=rng,
        )
