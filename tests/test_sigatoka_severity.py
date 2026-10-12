"""Testes da severidade da Sigatoka (escala de Stover modificada por Gauhl e DSI)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.sigatoka_severity import (
    VALIDATED,
    CropObservation,
    agreement_report,
    disease_severity_index,
    disease_severity_index_from_grades,
    dsi_mean_absolute_error,
    estimate_plant_severity,
    grade_from_percent,
    quadratic_weighted_kappa,
)


def _crop(
    cls: str = "Healthy",
    p: float = 1.0,
    height: float = 1000.0,
    *,
    in_scope: bool | None = True,
    lesion: float | None = None,
    leaf_id: str | None = None,
    zone: str | None = "folhas_baixas",
) -> CropObservation:
    """Recorte sintético: a classe prevista recebe probabilidade ``p``."""
    probs = {
        "Cordana": 0.0,
        "Fusarium_Wilt": 0.0,
        "Healthy": 0.0,
        "Pestalotiopsis": 0.0,
        "Sigatoka": 0.0,
    }
    probs[cls] = p
    return CropObservation(
        predicted_class=cls,
        probabilities=probs,
        height_mm=height,
        zone=zone,
        in_scope=in_scope,
        lesion_fraction=lesion,
        leaf_id=leaf_id,
    )


class TestGradeFromPercent:
    """Conversão % de área -> grau, com atenção às fronteiras da escala."""

    @pytest.mark.parametrize(
        ("pct", "grade"),
        [
            (0.0, 0),
            (0.01, 1),
            (0.999, 1),
            (1.0, 2),
            (5.0, 2),
            (5.01, 3),
            (15.0, 3),
            (15.01, 4),
            (33.0, 4),
            (33.01, 5),
            (50.0, 5),
            (50.01, 6),
            (100.0, 6),
        ],
    )
    def test_fronteiras(self, pct: float, grade: int) -> None:
        """Cada limite superior citado na literatura pertence ao grau de baixo."""
        assert grade_from_percent(pct) == grade

    @pytest.mark.parametrize("pct", [-0.1, 100.1, math.nan])
    def test_valores_invalidos(self, pct: float) -> None:
        """Porcentagens fora de [0, 100] são rejeitadas."""
        with pytest.raises(ValueError):
            grade_from_percent(pct)

    def test_escala_marcada_como_nao_validada(self) -> None:
        """A escala precisa de validação agronômica antes de uso comercial."""
        assert VALIDATED is False


class TestDiseaseSeverityIndex:
    """Fórmula DSI = Σ n_b·b / ((N-1)·T) x 100."""

    def test_exemplo_calculado_a_mao(self) -> None:
        """10 folhas: 2xg0, 1xg1, 3xg2, 2xg3, 1xg4, 1xg5 -> Σ=22; 22/(6·10)·100 = 36,67."""
        counts = {0: 2, 1: 1, 2: 3, 3: 2, 4: 1, 5: 1, 6: 0}
        assert disease_severity_index(counts) == pytest.approx(22 / 60 * 100)
        grades = [0, 0, 1, 2, 2, 2, 3, 3, 4, 5]
        assert disease_severity_index_from_grades(grades) == pytest.approx(36.6666667)

    def test_extremos(self) -> None:
        """Todas as folhas sadias -> 0; todas no grau 6 -> 100."""
        assert disease_severity_index([5, 0, 0, 0, 0, 0, 0]) == 0.0
        assert disease_severity_index({6: 4}) == 100.0

    def test_sem_folhas_e_erro(self) -> None:
        """T = 0 não tem DSI definido."""
        with pytest.raises(ValueError):
            disease_severity_index({})


class TestEstimatePlantSeverity:
    """Estimativa automática por planta a partir dos recortes do rover."""

    def test_planta_sadia_dsi_zero(self) -> None:
        """Só recortes Healthy -> todas as faixas no grau 0 e DSI 0 com intervalo [0, 0]."""
        crops = [_crop("Healthy", 0.95, h) for h in range(900, 2400, 75)]
        est = estimate_plant_severity(crops)
        assert est.dsi == 0.0
        assert est.dsi_ci_low == 0.0
        assert est.dsi_ci_high == 0.0
        assert all(leaf.grade == 0 for leaf in est.leaves)
        assert est.n_crops_used == len(crops)

    def test_recortes_fora_do_escopo_ignorados(self) -> None:
        """Recortes com in_scope=False não entram na estimativa, mesmo se "Sigatoka"."""
        healthy = [_crop("Healthy", 0.9, 1000 + i) for i in range(12)]
        junk = [_crop("Sigatoka", 1.0, 1000 + i, in_scope=False) for i in range(3)]
        est = estimate_plant_severity(healthy + junk)
        assert est.dsi == 0.0
        assert est.n_crops_used == 12
        assert est.n_crops_out_of_scope == 3

    def test_muitos_fora_do_escopo_gera_aviso(self) -> None:
        """Mais de 30% fora do escopo gera aviso em português."""
        crops = [_crop("Healthy", 0.9, 1000)] * 5 + [_crop(in_scope=False)] * 5
        est = estimate_plant_severity(crops)
        assert any("fora do escopo" in w for w in est.warnings)

    def test_in_scope_none_tratado_como_folha(self) -> None:
        """Escopo desconhecido (None) conta como folha."""
        est = estimate_plant_severity([_crop("Healthy", 0.9, 1000, in_scope=None)] * 10)
        assert est.n_crops_used == 10

    def test_pseudocaule_ignorado(self) -> None:
        """A Sigatoka é foliar: recortes da zona pseudocaule não entram."""
        crops = [_crop("Healthy", 0.9, 1000)] * 10 + [
            _crop("Sigatoka", 1.0, 500, zone="pseudocaule")
        ] * 5
        est = estimate_plant_severity(crops)
        assert est.dsi == 0.0
        assert est.n_crops_other_zone == 5

    def test_lesion_fraction_tem_precedencia(self) -> None:
        """Todos os recortes são Sigatoka (proporção = 100% -> grau 6), mas a segmentação
        diz 3% de lesão -> grau 2. A lesion_fraction deve prevalecer."""
        crops = [_crop("Sigatoka", 1.0, 1000, lesion=0.03) for _ in range(10)]
        est = estimate_plant_severity(crops)
        (leaf,) = est.leaves
        assert leaf.method == "lesion_fraction"
        assert leaf.affected_pct == pytest.approx(3.0)
        assert leaf.grade == 2
        without = estimate_plant_severity([_crop("Sigatoka", 1.0, 1000) for _ in range(10)])
        assert without.leaves[0].method == "crop_proportion"
        assert without.leaves[0].grade == 6
        assert any("lesion_fraction" in w for w in without.warnings)

    def test_lesion_fraction_parcial_usa_so_recortes_com_segmentacao(self) -> None:
        """Na mesma folha, recortes sem lesion_fraction não diluem os que têm."""
        crops = [_crop("Sigatoka", 1.0, 1000, lesion=0.10)] * 5 + [_crop("Healthy")] * 20
        est = estimate_plant_severity(crops)
        assert est.leaves[0].affected_pct == pytest.approx(10.0)
        assert est.leaves[0].n_crops == 5

    def test_proporcao_ponderada_pela_probabilidade(self) -> None:
        """Sem segmentação: 2 de 10 recortes com Sigatoka a p=0,5 -> 10% -> grau 3."""
        crops = [_crop("Sigatoka", 0.5, 1000)] * 2 + [_crop("Healthy", 0.9, 1000)] * 8
        est = estimate_plant_severity(crops)
        assert est.leaves[0].affected_pct == pytest.approx(10.0)
        assert est.leaves[0].grade == 3

    def test_agrupamento_por_leaf_id_e_por_faixa(self) -> None:
        """leaf_id define a folha; sem ele, faixas de 300 mm a partir de 900 mm."""
        by_id = [_crop(height=1000 + 400 * (i % 2), leaf_id="A") for i in range(6)]
        est = estimate_plant_severity(by_id + [_crop(leaf_id="B")] * 6)
        assert sorted(leaf.key for leaf in est.leaves) == ["folha_A", "folha_B"]
        assert all(leaf.grouping == "leaf_id" for leaf in est.leaves)

        bands = [_crop(height=h) for h in (950, 1100, 1250, 1450, 2000)] * 3
        est = estimate_plant_severity(bands)
        assert [leaf.key for leaf in est.leaves] == [
            "faixa_900-1200mm",
            "faixa_1200-1500mm",
            "faixa_1800-2100mm",
        ]
        assert any("faixas de altura" in w for w in est.warnings)

    def test_dsi_da_planta_a_partir_das_faixas(self) -> None:
        """Faixa baixa grau 4 (20%), faixa alta grau 2 (2%) -> DSI = 6/(6·2)·100 = 50."""
        low = [_crop("Sigatoka", 1.0, 1000, lesion=0.20)] * 10
        high = [_crop("Sigatoka", 1.0, 1300, lesion=0.02)] * 10
        est = estimate_plant_severity(low + high)
        assert [leaf.grade for leaf in est.leaves] == [4, 2]
        assert est.dsi == pytest.approx(50.0)

    def test_poucos_recortes_e_sem_recortes(self) -> None:
        """Poucos recortes geram aviso; nenhum recorte utilizável -> DSI None."""
        est = estimate_plant_severity([_crop()] * 3)
        assert any("Poucos recortes" in w for w in est.warnings)
        empty = estimate_plant_severity([_crop(in_scope=False)] * 4)
        assert empty.dsi is None
        assert empty.leaves == []
        assert empty.warnings

    def test_aceita_dicts_do_backend(self) -> None:
        """Os dicts já produzidos pelo backend podem ser passados diretamente."""
        data = {
            "predicted_class": "Sigatoka",
            "probabilities": {"Sigatoka": 0.8, "Healthy": 0.2},
            "height_mm": 1000,
            "zone": "folhas_baixas",
            "in_scope": True,
        }
        est = estimate_plant_severity([data] * 10)
        assert est.leaves[0].affected_pct == pytest.approx(80.0)
        assert est.to_dict()["scale_validated"] is False


class TestUncertaintyInterval:
    """Intervalo de incerteza por bootstrap estratificado."""

    @staticmethod
    def _plant(repeats: int) -> list[CropObservation]:
        """Duas faixas com lesões variadas (média 10% -> grau 3 em ambas)."""
        pattern = [0.0, 0.05, 0.10, 0.15, 0.20]
        crops = []
        for height in (1000, 1300):
            crops += [_crop("Sigatoka", 1.0, height, lesion=lf) for lf in pattern * repeats]
        return crops

    def test_intervalo_contem_estimativa(self) -> None:
        """O intervalo sempre contém o DSI pontual."""
        for repeats in (1, 2, 10):
            est = estimate_plant_severity(self._plant(repeats))
            assert est.dsi_ci_low <= est.dsi <= est.dsi_ci_high

    def test_intervalo_encolhe_com_mais_recortes(self) -> None:
        """Mesma distribuição, 40x mais recortes -> intervalo estritamente mais estreito."""
        small = estimate_plant_severity(self._plant(1))
        large = estimate_plant_severity(self._plant(40))
        assert small.dsi == large.dsi == pytest.approx(50.0)
        width_small = small.dsi_ci_high - small.dsi_ci_low
        width_large = large.dsi_ci_high - large.dsi_ci_low
        assert width_large < width_small

    def test_reprodutivel_com_seed_fixa(self) -> None:
        """Mesma seed -> mesmo intervalo."""
        a = estimate_plant_severity(self._plant(1), seed=7)
        b = estimate_plant_severity(self._plant(1), seed=7)
        assert (a.dsi_ci_low, a.dsi_ci_high) == (b.dsi_ci_low, b.dsi_ci_high)


class TestAgreement:
    """Concordância rover x agrônomo."""

    def test_kappa_concordancia_perfeita(self) -> None:
        """Graus idênticos -> kappa 1."""
        grades = [0, 1, 2, 3, 4, 5, 6, 3, 2]
        assert quadratic_weighted_kappa(grades, grades) == pytest.approx(1.0)

    def test_kappa_aleatorio_proximo_de_zero(self) -> None:
        """Graus independentes e aleatórios -> kappa ~0 em várias seeds."""
        kappas = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            a = rng.integers(0, 7, size=2000)
            b = rng.integers(0, 7, size=2000)
            kappas.append(quadratic_weighted_kappa(a, b))
        assert all(abs(k) < 0.1 for k in kappas)
        assert abs(float(np.mean(kappas))) < 0.03

    def test_kappa_exemplo_calculado_a_mao(self) -> None:
        """A=[0,1,2], B=[0,2,2], k=3, w=(i-j)²/4.
        Σw·O = (1/4)(1/3) = 1/12; marginais A=(1/3,1/3,1/3), B=(1/3,0,2/3);
        Σw·E = (1/9)(5/4) + (2/9)(5/4) = 5/12; κ = 1 - (1/12)/(5/12) = 0,8."""
        assert quadratic_weighted_kappa([0, 1, 2], [0, 2, 2], n_categories=3) == (
            pytest.approx(0.8)
        )

    def test_kappa_indefinido_e_entradas_invalidas(self) -> None:
        """Sem variação nos dois avaliadores -> nan; tamanhos diferentes -> erro."""
        assert math.isnan(quadratic_weighted_kappa([2, 2], [2, 2]))
        with pytest.raises(ValueError):
            quadratic_weighted_kappa([1, 2], [1])
        with pytest.raises(ValueError):
            quadratic_weighted_kappa([7], [0])

    def test_mae_e_relatorio(self) -> None:
        """MAE do DSI: |10-12| e |50-44| -> média 4."""
        assert dsi_mean_absolute_error([10, 50], [12, 44]) == pytest.approx(4.0)
        report = agreement_report([0, 1, 2], [0, 2, 2], [10, 50], [12, 44])
        assert report.dsi_mae == pytest.approx(4.0)
        assert report.n_leaves == 3
        assert report.n_plants == 2
