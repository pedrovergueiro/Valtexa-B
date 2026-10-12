"""Severidade da Sigatoka na escala de campo (Stover modificada por Gauhl) e índice DSI.

Este módulo traduz as classificações por recorte produzidas pelo rover VALTEXA-B para a
linguagem que o agrônomo usa no campo: o grau de Stover modificado por Gauhl (0 a 6) por
folha e o índice de severidade da doença (DSI, "disease severity index") por planta.

Referências
-----------
- Stover, R. H. (1971). A proposed international scale for estimating intensity of banana
  leaf spot (*Mycosphaerella musicola* Leach). Tropical Agriculture (Trinidad) 48: 185-196.
  (Escala original, que a de Gauhl modifica.)
- Gauhl, F. (1989). Untersuchungen zur Epidemiologie und Ökologie der Schwarzen
  Sigatoka-Krankheit (*Mycosphaerella fijiensis* Morelet) an Kochbananen (*Musa* sp.) in
  Costa Rica. Göttinger Beiträge zur Land- und Forstwirtschaft in den Tropen und
  Subtropen 42. (Origem da modificação de Gauhl.)
- Gauhl, F., Pasberg-Gauhl, C., Vuylsteke, D. & Ortiz, R. (1993). Multilocational
  evaluation of black Sigatoka resistance in banana and plantain. IITA Research Guide 47.
- Carlier, J., De Waele, D. & Escalant, J.-V. (2002). Global evaluation of *Musa* germplasm
  for resistance to Fusarium wilt, *Mycosphaerella* leaf spot diseases and nematodes:
  in-depth evaluation. INIBAP Technical Guidelines 6. INIBAP, Montpellier.
  (Guia técnico com a tabela de graus e a fórmula do índice de severidade usadas aqui.)

AVISO — NÃO VALIDADO
--------------------
Os limites abaixo foram transcritos da literatura de memória, sem conferência contra o PDF
original, e a conversão automática "recortes do rover -> % de área -> grau" é uma aproximação
nossa. ``VALIDATED = False`` até que um agrônomo confira os limites e a estimativa seja
comparada com avaliações manuais em campo (ver ``agreement_report``). Não usar em decisão
comercial (laudo, recomendação de pulverização) enquanto ``VALIDATED`` for ``False``.

Limitações conhecidas da estimativa automática
----------------------------------------------
- O rover só vê as folhas baixas (as mais velhas), onde a Sigatoka é mais severa. O DSI de
  campo é calculado sobre TODAS as folhas em pé da planta; o nosso é calculado só sobre as
  folhas/faixas visíveis e, portanto, tende a SUPERESTIMAR o DSI da planta inteira. Os dois
  números não são diretamente comparáveis sem calibração.
- Sem ``leaf_id`` as "folhas" são faixas de altura: uma faixa pode misturar pedaços de
  várias folhas e uma folha pode atravessar várias faixas. Os dois lados da planta (câmeras
  esquerda e direita) são somados na mesma faixa. A altura só aproxima a posição/idade da
  folha (folhas mais baixas ~ mais velhas), e o número de faixas não é o número de folhas.
- Sem ``lesion_fraction`` a % de área é aproximada pela proporção de recortes com
  Sigatoka. Um recorte "com Sigatoka" raramente está 100% lesionado, então essa aproximação
  tende a superestimar a área afetada; é um indicador, não uma medida de área.
- O grau 1 da escala também admite a condição "só estrias e/ou até 10 manchas", que não dá
  para verificar sem contar lesões; usamos apenas o critério de % de área.
- O classificador tem uma classe única "Sigatoka" (não separa Sigatoka-negra de
  Sigatoka-amarela). A escala de Gauhl foi descrita para a Sigatoka-negra, mas é usada na
  prática para as duas.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------------------
# Escala de Stover modificada por Gauhl
# ---------------------------------------------------------------------------------------

#: Enquanto ``False``, os resultados deste módulo são experimentais: os limites da escala e a
#: conversão automática ainda não foram conferidos por um agrônomo.
VALIDATED: bool = False

#: Classe do classificador que corresponde à Sigatoka.
SIGATOKA_CLASS: str = "Sigatoka"

#: Número de graus da escala (0 a 6), o "N" da fórmula do DSI em Carlier et al. (2002).
N_GRADES: int = 7

#: Descrição de cada grau como aparece na literatura (Gauhl 1989; Gauhl et al. 1993;
#: Carlier et al. 2002), em % da área do limbo foliar com sintomas.
#: Grau de certeza: os intervalos 1-5, 6-15, 16-33, 34-50 e 51-100 % são os mais citados e
#: temos confiança alta neles; a menção "até 10 manchas" no grau 1 tem confiança média
#: quanto à redação exata. Precisa de conferência do agrônomo (ver ``VALIDATED``).
GRADE_DESCRIPTIONS: dict[int, str] = {
    0: "sem sintomas",
    1: "menos de 1% do limbo com sintomas (só estrias e/ou até 10 manchas)",
    2: "1 a 5% do limbo com sintomas",
    3: "6 a 15% do limbo com sintomas",
    4: "16 a 33% do limbo com sintomas",
    5: "34 a 50% do limbo com sintomas",
    6: "51 a 100% do limbo com sintomas",
}

#: Limite superior (inclusivo, em %) dos graus 2 a 5; acima de 50% é grau 6.
#: A literatura dá faixas em números inteiros (1-5, 6-15, 16-33, 34-50, 51-100) e deixa
#: lacunas (ex.: 5,5%). Não encontramos na literatura uma regra para essas lacunas; a
#: convenção adotada aqui é que cada grau vai até o limite superior citado, inclusive
#: (5% -> grau 2; 5,01% -> grau 3). Essa convenção de fronteira é nossa e precisa ser
#: confirmada pelo agrônomo.
GRADE_UPPER_BOUNDS_PCT: dict[int, float] = {2: 5.0, 3: 15.0, 4: 33.0, 5: 50.0}

#: Abaixo deste valor (em %) e acima de zero é grau 1 ("menos de 1%"); 1% exato é grau 2.
GRADE_1_UPPER_EXCLUSIVE_PCT: float = 1.0


def grade_from_percent(affected_pct: float) -> int:
    """Converte a % da área foliar com sintomas no grau de Stover modificado por Gauhl.

    Fronteiras (ver ``GRADE_UPPER_BOUNDS_PCT``): 0% -> 0; (0, 1) -> 1; [1, 5] -> 2;
    (5, 15] -> 3; (15, 33] -> 4; (33, 50] -> 5; (50, 100] -> 6. A inclusividade nas
    fronteiras é uma convenção nossa (a literatura usa faixas inteiras com lacunas).
    """
    if not math.isfinite(affected_pct) or affected_pct < 0.0 or affected_pct > 100.0:
        raise ValueError(f"porcentagem fora de [0, 100]: {affected_pct!r}")
    if affected_pct == 0.0:
        return 0
    if affected_pct < GRADE_1_UPPER_EXCLUSIVE_PCT:
        return 1
    for grade in sorted(GRADE_UPPER_BOUNDS_PCT):
        if affected_pct <= GRADE_UPPER_BOUNDS_PCT[grade]:
            return grade
    return 6


def _grades_from_percent_array(affected_pct: np.ndarray) -> np.ndarray:
    """Versão vetorizada de ``grade_from_percent`` (usada no bootstrap, sem validação)."""
    conditions = [
        affected_pct <= 0.0,
        affected_pct < GRADE_1_UPPER_EXCLUSIVE_PCT,
        *(affected_pct <= GRADE_UPPER_BOUNDS_PCT[g] for g in sorted(GRADE_UPPER_BOUNDS_PCT)),
    ]
    choices = [0, 1, *sorted(GRADE_UPPER_BOUNDS_PCT)]
    return np.select(conditions, choices, default=6).astype(int)


# ---------------------------------------------------------------------------------------
# Índice de severidade (DSI)
# ---------------------------------------------------------------------------------------


def disease_severity_index(
    leaves_per_grade: Mapping[int, int] | Sequence[int], n_grades: int = N_GRADES
) -> float:
    """Calcula o índice de severidade (DSI, em %) a partir do nº de folhas em cada grau.

    Fórmula de Carlier et al. (2002, INIBAP Technical Guidelines 6):

        DSI = Σ (n_b · b) / ((N - 1) · T) x 100

    onde n_b é o nº de folhas no grau b, N é o nº de graus da escala (7, graus 0 a 6) e T é
    o nº total de folhas avaliadas. O DSI vai de 0 (todas as folhas sem sintomas) a 100
    (todas no grau máximo). Função pura: serve tanto para a estimativa automática quanto para
    o formulário manual do agrônomo.

    ``leaves_per_grade`` pode ser um dict ``{grau: nº de folhas}`` ou uma sequência em que o
    índice é o grau.
    """
    if n_grades < 2:
        raise ValueError("a escala precisa de pelo menos 2 graus")
    items = (
        leaves_per_grade.items()
        if isinstance(leaves_per_grade, Mapping)
        else enumerate(leaves_per_grade)
    )
    weighted_sum = 0
    total = 0
    for grade, count in items:
        if not 0 <= grade < n_grades:
            raise ValueError(f"grau fora da escala 0..{n_grades - 1}: {grade!r}")
        if count < 0:
            raise ValueError(f"nº de folhas negativo no grau {grade}: {count!r}")
        weighted_sum += grade * count
        total += count
    if total == 0:
        raise ValueError("nenhuma folha avaliada (T = 0)")
    return weighted_sum / ((n_grades - 1) * total) * 100.0


def disease_severity_index_from_grades(grades: Iterable[int], n_grades: int = N_GRADES) -> float:
    """DSI a partir de uma lista de graus (um por folha); atalho para o formulário manual."""
    counts = [0] * n_grades
    for grade in grades:
        if not 0 <= grade < n_grades:
            raise ValueError(f"grau fora da escala 0..{n_grades - 1}: {grade!r}")
        counts[grade] += 1
    return disease_severity_index(counts, n_grades)


def _dsi_from_grade_matrix(grades: np.ndarray) -> np.ndarray:
    """DSI vetorizado: ``grades`` tem forma (n_reamostragens, n_folhas)."""
    return grades.sum(axis=1) / ((N_GRADES - 1) * grades.shape[1]) * 100.0


# ---------------------------------------------------------------------------------------
# Estimativa automática por planta a partir dos recortes do rover
# ---------------------------------------------------------------------------------------

#: Zona do recorte onde a Sigatoka é avaliada (a doença é foliar; o pseudocaule é ignorado).
LEAF_ZONES: tuple[str, ...] = ("folhas_baixas",)

#: Largura e origem das faixas de altura usadas quando não há ``leaf_id``. A origem coincide
#: com o início da zona "folhas_baixas" (900 mm). 300 mm = 2 recortes de 150 mm: estreita o
#: bastante para separar níveis de folhas, larga o bastante para juntar alguns recortes.
#: Valor escolhido por nós, sem base na literatura; ajustar com dados de campo.
HEIGHT_BAND_WIDTH_MM: float = 300.0
HEIGHT_BAND_ORIGIN_MM: float = 900.0

#: Limiares dos avisos de evidência fraca (escolhas nossas, conservadoras).
MIN_CROPS_PER_PLANT: int = 10
MIN_CROPS_PER_LEAF: int = 3
MAX_OUT_OF_SCOPE_FRACTION: float = 0.3

GroupingKind = Literal["leaf_id", "height_band"]
AreaMethod = Literal["lesion_fraction", "crop_proportion"]


@dataclass(frozen=True, slots=True)
class CropObservation:
    """Um recorte de 150 mm já classificado pelo modelo, como chega do pipeline do rover.

    ``in_scope`` False significa "não é folha" e o recorte é ignorado; None (escopo
    desconhecido) é tratado como dentro do escopo. ``lesion_fraction`` (0..1) vem do modelo
    de segmentação e pode faltar; ``leaf_id`` só existe quando a associação a uma folha
    específica foi feita.
    """

    predicted_class: str
    probabilities: Mapping[str, float]
    height_mm: float
    zone: str | None = None
    in_scope: bool | None = None
    lesion_fraction: float | None = None
    leaf_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CropObservation:
        """Constrói a partir do dict que o backend já produz (chaves com os mesmos nomes)."""
        leaf_id = data.get("leaf_id")
        lesion = data.get("lesion_fraction")
        return cls(
            predicted_class=str(data["predicted_class"]),
            probabilities=dict(data.get("probabilities") or {}),
            height_mm=float(data["height_mm"]),
            zone=data.get("zone"),
            in_scope=data.get("in_scope"),
            lesion_fraction=None if lesion is None else float(lesion),
            leaf_id=None if leaf_id is None else str(leaf_id),
        )

    def sigatoka_probability(self) -> float:
        """Probabilidade de Sigatoka; se faltar no dict e a classe prevista for Sigatoka, 1."""
        prob = self.probabilities.get(SIGATOKA_CLASS)
        if prob is None:
            return 1.0 if self.predicted_class == SIGATOKA_CLASS else 0.0
        return float(prob)


@dataclass(frozen=True, slots=True)
class LeafSeverity:
    """Grau estimado para uma folha (``leaf_id``) ou faixa de altura (aproximação de folha)."""

    key: str
    grouping: GroupingKind
    n_crops: int
    method: AreaMethod
    affected_pct: float
    grade: int
    height_min_mm: float
    height_max_mm: float


@dataclass(frozen=True, slots=True)
class PlantSeverityEstimate:
    """Resultado da estimativa automática para uma planta.

    ``dsi`` é None quando não houve nenhum recorte utilizável. ``dsi_ci_low``/``dsi_ci_high``
    formam o intervalo de incerteza por bootstrap (nível ``confidence``).
    """

    leaves: list[LeafSeverity]
    dsi: float | None
    dsi_ci_low: float | None
    dsi_ci_high: float | None
    confidence: float
    n_crops_total: int
    n_crops_used: int
    n_crops_out_of_scope: int
    n_crops_other_zone: int
    warnings: list[str] = field(default_factory=list)
    scale_validated: bool = VALIDATED

    def to_dict(self) -> dict[str, Any]:
        """Serializa para JSON (resposta da API)."""
        return asdict(self)


def _crop_affected_fraction(crop: CropObservation, method: AreaMethod) -> float:
    """Contribuição de um recorte para a fração de área afetada por Sigatoka (0..1).

    - ``lesion_fraction``: fração lesionada x P(Sigatoka). A segmentação marca lesão de
      qualquer doença; ponderar pela probabilidade atribui à Sigatoka só a parte que o
      classificador acredita ser dela.
    - ``crop_proportion``: P(Sigatoka) se a classe prevista for Sigatoka, senão 0. A média
      sobre os recortes é a "fração de recortes com Sigatoka ponderada pela probabilidade".
    """
    p_sig = min(max(crop.sigatoka_probability(), 0.0), 1.0)
    if method == "lesion_fraction":
        assert crop.lesion_fraction is not None
        return min(max(crop.lesion_fraction, 0.0), 1.0) * p_sig
    return p_sig if crop.predicted_class == SIGATOKA_CLASS else 0.0


def _group_key(crop: CropObservation, band_width_mm: float, band_origin_mm: float) -> str:
    if crop.leaf_id is not None:
        return f"folha_{crop.leaf_id}"
    band = math.floor((crop.height_mm - band_origin_mm) / band_width_mm)
    low = band_origin_mm + band * band_width_mm
    return f"faixa_{low:.0f}-{low + band_width_mm:.0f}mm"


def estimate_plant_severity(
    crops: Iterable[CropObservation | Mapping[str, Any]],
    *,
    leaf_zones: Sequence[str] = LEAF_ZONES,
    band_width_mm: float = HEIGHT_BAND_WIDTH_MM,
    band_origin_mm: float = HEIGHT_BAND_ORIGIN_MM,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PlantSeverityEstimate:
    """Estima grau por folha/faixa e o DSI de uma planta a partir dos recortes do rover.

    Passos:
    1. Descarta recortes fora do escopo (``in_scope is False``) e de zonas que não são folha
       (recortes sem zona são mantidos).
    2. Agrupa em folhas por ``leaf_id``; recortes sem ``leaf_id`` vão para faixas de altura
       de ``band_width_mm`` (aproximação da posição/idade da folha — ver limitações no
       docstring do módulo).
    3. Por grupo, estima a % de área com Sigatoka: se algum recorte do grupo tem
       ``lesion_fraction``, usa só esses (média de lesion_fraction x P(Sigatoka)); senão usa
       a proporção de recortes com Sigatoka ponderada pela probabilidade.
    4. Converte em grau (``grade_from_percent``) e calcula o DSI sobre os grupos.
    5. Intervalo de incerteza por bootstrap estratificado: reamostra os recortes com
       reposição DENTRO de cada grupo (o nº de folhas/faixas fica fixo, como no campo) e
       recalcula o DSI; intervalo percentil com seed fixa (reprodutível). Como o grau é
       discreto, o intervalo percentil pode, em casos raros, não conter a estimativa
       pontual; nesse caso ele é estendido até ela.
    """
    if band_width_mm <= 0:
        raise ValueError("band_width_mm deve ser positivo")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence deve estar em (0, 1)")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap deve ser >= 1")

    all_crops = [
        c if isinstance(c, CropObservation) else CropObservation.from_mapping(c) for c in crops
    ]
    out_of_scope = [c for c in all_crops if c.in_scope is False]
    in_scope = [c for c in all_crops if c.in_scope is not False]
    other_zone = [c for c in in_scope if c.zone is not None and c.zone not in leaf_zones]
    usable = [c for c in in_scope if c.zone is None or c.zone in leaf_zones]

    warnings: list[str] = []
    n_total = len(all_crops)
    if n_total and len(out_of_scope) / n_total > MAX_OUT_OF_SCOPE_FRACTION:
        warnings.append(
            f"{len(out_of_scope)} de {n_total} recortes fora do escopo (não são folha); "
            "a imagem pode estar obstruída ou desenquadrada."
        )

    if not usable:
        warnings.append("Nenhum recorte de folha utilizável; não foi possível estimar o DSI.")
        return PlantSeverityEstimate(
            leaves=[],
            dsi=None,
            dsi_ci_low=None,
            dsi_ci_high=None,
            confidence=confidence,
            n_crops_total=n_total,
            n_crops_used=0,
            n_crops_out_of_scope=len(out_of_scope),
            n_crops_other_zone=len(other_zone),
            warnings=warnings,
        )

    if len(usable) < MIN_CROPS_PER_PLANT:
        warnings.append(
            f"Poucos recortes de folha ({len(usable)} < {MIN_CROPS_PER_PLANT}); "
            "estimativa pouco confiável."
        )
    if any(c.leaf_id is None for c in usable):
        warnings.append(
            "Folhas aproximadas por faixas de altura (sem leaf_id); o nº de faixas não é o "
            "nº de folhas e o DSI não é diretamente comparável ao DSI de campo."
        )
        if any(c.leaf_id is not None for c in usable):
            warnings.append("Mistura de recortes com e sem leaf_id na mesma planta.")

    groups: dict[str, list[CropObservation]] = {}
    for crop in usable:
        groups.setdefault(_group_key(crop, band_width_mm, band_origin_mm), []).append(crop)

    leaves: list[LeafSeverity] = []
    group_values: list[np.ndarray] = []
    groups_without_lesion = 0
    for key in sorted(groups, key=lambda k: min(c.height_mm for c in groups[k])):
        members = groups[key]
        with_lesion = [c for c in members if c.lesion_fraction is not None]
        method: AreaMethod = "lesion_fraction" if with_lesion else "crop_proportion"
        used = with_lesion or members
        if not with_lesion:
            groups_without_lesion += 1
        values = np.array([_crop_affected_fraction(c, method) for c in used], dtype=float)
        group_values.append(values)
        pct = float(values.mean() * 100.0)
        leaves.append(
            LeafSeverity(
                key=key,
                grouping="leaf_id" if members[0].leaf_id is not None else "height_band",
                n_crops=len(used),
                method=method,
                affected_pct=pct,
                grade=grade_from_percent(pct),
                height_min_mm=min(c.height_mm for c in members),
                height_max_mm=max(c.height_mm for c in members),
            )
        )
        if len(used) < MIN_CROPS_PER_LEAF:
            warnings.append(
                f"{key}: só {len(used)} recorte(s); grau desta folha/faixa pouco confiável."
            )

    if groups_without_lesion:
        warnings.append(
            f"{groups_without_lesion} de {len(leaves)} folha(s)/faixa(s) sem lesion_fraction; "
            "% de área aproximada pela proporção de recortes com Sigatoka "
            "(tende a superestimar)."
        )

    dsi = disease_severity_index_from_grades(leaf.grade for leaf in leaves)

    # Bootstrap estratificado por folha/faixa (ver docstring).
    rng = np.random.default_rng(seed)
    boot_grades = np.empty((n_bootstrap, len(group_values)), dtype=int)
    for j, values in enumerate(group_values):
        idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        boot_grades[:, j] = _grades_from_percent_array(values[idx].mean(axis=1) * 100.0)
    boot_dsi = _dsi_from_grade_matrix(boot_grades)
    alpha = (1.0 - confidence) / 2.0
    ci_low = min(float(np.quantile(boot_dsi, alpha)), dsi)
    ci_high = max(float(np.quantile(boot_dsi, 1.0 - alpha)), dsi)

    return PlantSeverityEstimate(
        leaves=leaves,
        dsi=dsi,
        dsi_ci_low=ci_low,
        dsi_ci_high=ci_high,
        confidence=confidence,
        n_crops_total=n_total,
        n_crops_used=sum(leaf.n_crops for leaf in leaves),
        n_crops_out_of_scope=len(out_of_scope),
        n_crops_other_zone=len(other_zone),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------------------
# Concordância com o agrônomo
# ---------------------------------------------------------------------------------------


def quadratic_weighted_kappa(
    rater_a: Sequence[int], rater_b: Sequence[int], n_categories: int = N_GRADES
) -> float:
    """Kappa de Cohen com pesos quadráticos entre dois avaliadores (ex.: rover x agrônomo).

    κ_w = 1 - Σ w_ij·O_ij / Σ w_ij·E_ij, com w_ij = (i - j)² / (k - 1)², O a matriz de
    confusão observada (proporções) e E = produto externo das marginais (Cohen 1968).
    Pesos quadráticos são o padrão para escalas ordinais como a de Gauhl: errar um grau
    custa pouco, errar três custa nove vezes mais. As categorias são fixas em
    0..``n_categories``-1 (não inferidas dos dados), para que valores de estudos diferentes
    sejam comparáveis. Retorna ``nan`` se a discordância esperada for zero (ex.: os dois
    avaliadores deram sempre o mesmo grau), caso em que o kappa é indefinido.

    Referência: Cohen, J. (1968). Weighted kappa. Psychological Bulletin 70(4): 213-220.
    """
    a = np.asarray(rater_a, dtype=int)
    b = np.asarray(rater_b, dtype=int)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("as duas listas de graus precisam ter o mesmo tamanho")
    if a.size == 0:
        raise ValueError("listas de graus vazias")
    if n_categories < 2:
        raise ValueError("são necessárias pelo menos 2 categorias")
    for arr in (a, b):
        if arr.min() < 0 or arr.max() >= n_categories:
            raise ValueError(f"grau fora de 0..{n_categories - 1}")

    observed = np.zeros((n_categories, n_categories), dtype=float)
    np.add.at(observed, (a, b), 1.0)
    observed /= a.size
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    cats = np.arange(n_categories)
    weights = (cats[:, None] - cats[None, :]) ** 2 / (n_categories - 1) ** 2
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0.0:
        return float("nan")
    return 1.0 - float((weights * observed).sum()) / expected_disagreement


def dsi_mean_absolute_error(dsi_auto: Sequence[float], dsi_reference: Sequence[float]) -> float:
    """Erro absoluto médio (pontos percentuais de DSI) entre estimativa automática e agrônomo."""
    auto = np.asarray(dsi_auto, dtype=float)
    ref = np.asarray(dsi_reference, dtype=float)
    if auto.shape != ref.shape or auto.ndim != 1 or auto.size == 0:
        raise ValueError("as listas de DSI precisam ter o mesmo tamanho (> 0)")
    return float(np.abs(auto - ref).mean())


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """Resumo da concordância rover x agrônomo."""

    weighted_kappa: float
    n_leaves: int
    dsi_mae: float | None
    n_plants: int


def agreement_report(
    auto_grades: Sequence[int],
    agronomist_grades: Sequence[int],
    dsi_auto: Sequence[float] | None = None,
    dsi_agronomist: Sequence[float] | None = None,
) -> AgreementReport:
    """Concordância por folha (kappa ponderado quadrático) e por planta (MAE do DSI).

    Os graus precisam estar pareados (mesma folha/faixa avaliada pelos dois). O MAE só é
    calculado se as duas listas de DSI forem fornecidas.
    """
    kappa = quadratic_weighted_kappa(auto_grades, agronomist_grades)
    mae: float | None = None
    n_plants = 0
    if dsi_auto is not None and dsi_agronomist is not None:
        mae = dsi_mean_absolute_error(dsi_auto, dsi_agronomist)
        n_plants = len(dsi_auto)
    return AgreementReport(
        weighted_kappa=kappa, n_leaves=len(auto_grades), dsi_mae=mae, n_plants=n_plants
    )
