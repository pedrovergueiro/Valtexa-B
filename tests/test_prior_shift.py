"""Testes de app.inference.prior_shift com um classificador sintético de posteriores exatos.

O "modelo" é uma mistura de gaussianas isotrópicas 2D com K=5 classes (análogo à banana:
Cordana, Fusarium_Wilt, Healthy, Pestalotiopsis, Sigatoka). Como as densidades de classe são
conhecidas, p_s(y|x) ∝ π_s(y)·N(x; μ_y, I) é exatamente calibrado; uma temperatura T != 1
produz um modelo descalibrado controlado.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from app.inference.prior_shift import (
    DIVERGENCE_LIMIT,
    LEVEL_DRIFT,
    LEVEL_OK,
    LEVEL_WARN,
    BBSEConfig,
    DriftMonitor,
    DriftThresholds,
    EMConfig,
    FieldPriorStore,
    ReferenceProfile,
    TrackerConfig,
    ValidationSummary,
    adjust_posteriors,
    apply_floor,
    bbse_prior_estimate,
    bbse_weight,
    check_drift,
    chi2_sf,
    em_prior_estimate,
    estimate_target_prior,
    joint_confusion,
    kolmogorov_sf,
    ks_2samp,
    mmd_permutation_test,
    project_to_simplex,
    psi,
)

K = 5
HEALTHY = 2
MEANS = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2], [1.1, 2.75]])
# Prior "de dataset público": Sigatoka 35%, sadia 30%.
PI_S = np.array([0.15, 0.15, 0.30, 0.05, 0.35])
# Talhão real: 95% sadia.
PI_FARM = np.array([0.01, 0.01, 0.95, 0.01, 0.02])
NOVELTY_SCALE = -2.0 * math.log(0.01)  # quantil 99% de χ²₂: novelty > 1 é "estranho"


def sample(pi: np.ndarray, n: int, rng: np.random.Generator, noise: float = 1.0):
    """Amostra (x, y) da mistura; ``noise`` > 1 simula imagens borradas (deriva de aparência)."""
    y = rng.choice(K, size=n, p=pi)
    x = MEANS[y] + noise * rng.normal(size=(n, 2))
    return x, y


def posteriors(x: np.ndarray, prior: np.ndarray = PI_S, temperature: float = 1.0) -> np.ndarray:
    """Posterior exato (T=1) ou descalibrado por temperatura (T != 1)."""
    loglik = -0.5 * ((x[:, None, :] - MEANS[None]) ** 2).sum(-1)
    logits = (loglik + np.log(prior)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def novelty(x: np.ndarray) -> np.ndarray:
    """Distância de Mahalanobis² à classe mais próxima, normalizada pelo quantil 99%."""
    d2 = ((x[:, None, :] - MEANS[None]) ** 2).sum(-1).min(axis=1)
    return d2 / NOVELTY_SCALE


EMBED_PROJ = np.random.default_rng(1234).normal(size=(2, 8)) / math.sqrt(2)


def embeddings(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Embedding 8D: projeção linear fixa de x mais ruído (como uma penúltima camada)."""
    return x @ EMBED_PROJ + 0.3 * rng.normal(size=(x.shape[0], 8))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], 1e-12, None))))


# ---------------------------------------------------------------------------


class TestUtilities:
    """Funções numéricas auxiliares."""

    def test_project_to_simplex_keeps_simplex_points(self):
        """Um ponto já no simplex é seu próprio projetado."""
        v = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(project_to_simplex(v), v)

    def test_project_to_simplex_negative_entries(self):
        """Entradas negativas (típicas da inversão do BBSE) viram zero e a soma vira 1."""
        q = project_to_simplex(np.array([1.2, -0.3, 0.4]))
        assert q.min() >= 0 and math.isclose(q.sum(), 1.0)
        np.testing.assert_allclose(q, [0.9, 0.0, 0.1])

    def test_apply_floor(self):
        """Classes abaixo do piso ficam exatamente no piso e a soma continua 1."""
        q = apply_floor(np.array([0.0, 0.0005, 0.9995]), 0.01)
        np.testing.assert_allclose(q, [0.01, 0.01, 0.98])

    def test_adjust_identity_when_priors_equal(self):
        """Com π_t = π_s a correção não altera nada."""
        rng = np.random.default_rng(0)
        p = posteriors(sample(PI_S, 50, rng)[0])
        np.testing.assert_allclose(adjust_posteriors(p, PI_S, PI_S), p)

    def test_adjust_equals_bayes_with_true_prior(self):
        """Corrigir com o prior verdadeiro dá exatamente o posterior de Bayes do talhão."""
        rng = np.random.default_rng(1)
        x, _ = sample(PI_FARM, 100, rng)
        np.testing.assert_allclose(
            adjust_posteriors(posteriors(x), PI_S, PI_FARM), posteriors(x, PI_FARM), atol=1e-12
        )

    def test_kolmogorov_and_chi2_known_values(self):
        """Valores de tabela: Q_KS(1,358)≈0,05; χ²₁(3,841)≈0,05; χ²₅(11,07)≈0,05."""
        assert kolmogorov_sf(1.358) == pytest.approx(0.05, abs=1e-3)
        assert chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
        assert chi2_sf(11.070, 5) == pytest.approx(0.05, abs=1e-3)
        assert chi2_sf(43.773, 30) == pytest.approx(0.05, abs=1e-3)
        assert chi2_sf(0.5, 4) == pytest.approx(0.97350, abs=1e-4)

    def test_invalid_probabilities_rejected(self):
        """Probabilidades negativas indicam bug a montante e devem gerar erro."""
        with pytest.raises(ValueError):
            em_prior_estimate(np.array([[0.5, -0.1, 0.6]] * 40), [1 / 3] * 3)


class TestEM:
    """EM de Saerens et al. (2002) com classificador calibrado."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_recovers_target_prevalence(self, seed):
        """Com 5000 imagens sem rótulo o EM recupera a prevalência com erro < 2 p.p."""
        rng = np.random.default_rng(seed)
        x, _ = sample(PI_FARM, 5000, rng)
        est = em_prior_estimate(posteriors(x), PI_S)
        assert est.method == "em" and est.converged
        assert np.max(np.abs(est.prevalences - PI_FARM)) < 0.02

    def test_recovers_moderate_shift(self):
        """Mudança moderada (Sigatoka 35% → 10%) também é recuperada."""
        rng = np.random.default_rng(3)
        target = np.array([0.2, 0.1, 0.5, 0.1, 0.1])
        x, _ = sample(target, 5000, rng)
        est = em_prior_estimate(posteriors(x), PI_S)
        assert np.max(np.abs(est.prevalences - target)) < 0.03

    def test_correction_improves_accuracy_and_log_loss(self):
        """Sob mudança de prior, corrigir com o π_t estimado melhora acurácia e log-loss."""
        rng = np.random.default_rng(4)
        x, y = sample(PI_FARM, 5000, rng)
        p = posteriors(x)
        est = em_prior_estimate(p, PI_S)
        q = adjust_posteriors(p, PI_S, est.prevalences)
        acc_raw = np.mean(p.argmax(1) == y)
        acc_adj = np.mean(q.argmax(1) == y)
        assert acc_adj > acc_raw + 0.05
        assert log_loss(q, y) < log_loss(p, y) - 0.1
        # Falsos alarmes de doença (sadia prevista como doente) caem drasticamente.
        fa_raw = np.mean(p[y == HEALTHY].argmax(1) != HEALTHY)
        fa_adj = np.mean(q[y == HEALTHY].argmax(1) != HEALTHY)
        assert fa_adj < fa_raw / 3

    def test_min_samples_fallback(self):
        """Com poucas imagens devolve o prior do treino, sinalizado como 'source'."""
        rng = np.random.default_rng(5)
        est = em_prior_estimate(posteriors(sample(PI_FARM, 10, rng)[0]), PI_S)
        assert est.method == "source"
        np.testing.assert_allclose(est.prevalences, PI_S)

    def test_floor_keeps_absent_class_alive(self):
        """Classe ausente do talhão não vai a zero: fica no piso."""
        rng = np.random.default_rng(6)
        target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        x, _ = sample(target, 2000, rng)
        est = em_prior_estimate(posteriors(x), PI_S, EMConfig(prevalence_floor=0.002))
        assert est.prevalences.min() >= 0.002 - 1e-12

    def test_max_iter_reports_non_convergence(self):
        """Limite de iterações é respeitado e a não convergência é reportada."""
        rng = np.random.default_rng(7)
        x, _ = sample(PI_FARM, 500, rng)
        est = em_prior_estimate(posteriors(x), PI_S, EMConfig(max_iter=2, tol=1e-12))
        assert not est.converged and est.iterations == 2 and est.notes


class TestBBSE:
    """BBSE de Lipton et al. (2018) e escolha EM x BBSE."""

    @pytest.fixture
    def miscalibrated(self):
        """Modelo super-suavizado (T=3): mesma fronteira de decisão, probabilidades erradas."""
        rng = np.random.default_rng(10)
        xv, yv = sample(PI_S, 4000, rng)
        summary = ValidationSummary.from_validation(posteriors(xv, temperature=3.0), yv)
        xt, _ = sample(PI_FARM, 5000, rng)
        return summary, posteriors(xt, temperature=3.0)

    def test_joint_confusion_sums(self):
        """C é conjunta: soma 1 e colunas somam o prior da validação."""
        rng = np.random.default_rng(11)
        x, y = sample(PI_S, 3000, rng)
        c = joint_confusion(y, posteriors(x), K)
        assert c.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(c.sum(axis=0), np.bincount(y, minlength=K) / y.size)

    def test_bbse_recovers_where_em_fails(self, miscalibrated):
        """Com modelo descalibrado o EM erra muito e o BBSE acerta."""
        summary, pt = miscalibrated
        em = em_prior_estimate(pt, PI_S)
        bb = bbse_prior_estimate(pt, summary.confusion)
        err_em = np.max(np.abs(em.prevalences - PI_FARM))
        err_bb = np.max(np.abs(bb.prevalences - PI_FARM))
        assert err_em > 0.2
        assert err_bb < 0.05

    def test_selection_picks_bbse_when_miscalibrated(self, miscalibrated):
        """O diagnóstico detecta a descalibração e a escolha automática usa o BBSE."""
        summary, pt = miscalibrated
        assert bbse_weight(summary) == 1.0
        est = estimate_target_prior(pt, PI_S, summary)
        assert est.method == "bbse"
        assert np.max(np.abs(est.prevalences - PI_FARM)) < 0.05

    def test_selection_picks_em_when_calibrated(self):
        """Com modelo calibrado a escolha automática fica com o EM (mais eficiente)."""
        rng = np.random.default_rng(12)
        xv, yv = sample(PI_S, 4000, rng)
        summary = ValidationSummary.from_validation(posteriors(xv), yv)
        assert bbse_weight(summary) < 0.5
        xt, _ = sample(PI_FARM, 5000, rng)
        est = estimate_target_prior(posteriors(xt), PI_S, summary)
        # Sadia x Sigatoka se confundem (médias próximas): tolerância um pouco maior.
        assert np.max(np.abs(est.prevalences - PI_FARM)) < 0.035

    @pytest.mark.parametrize("n_val", [500, 1000])
    def test_small_validation_not_mistaken_for_miscalibration(self, n_val):
        """ECE tem viés ~1/√N: validação pequena de modelo calibrado não deve virar BBSE."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            xv, yv = sample(PI_S, n_val, rng)
            assert bbse_weight(ValidationSummary.from_validation(posteriors(xv), yv)) == 0.0

    def test_ill_conditioned_is_regularized(self):
        """Classes quase indistinguíveis: a estimativa é regularizada, fica no simplex e avisa."""
        c_cond = np.array([[0.5, 0.49, 0.0], [0.5, 0.51, 0.0], [0.0, 0.0, 1.0]])
        pi_s = np.array([0.4, 0.4, 0.2])
        c = c_cond * pi_s
        preds = np.repeat([0, 1, 2], [300, 200, 500])
        est = bbse_prior_estimate(preds, c, BBSEConfig(max_condition=50))
        assert est.notes and "mal condicionada" in est.notes[0]
        assert est.prevalences.min() >= 0 and est.prevalences.sum() == pytest.approx(1.0)
        # A classe bem separada é estimada corretamente mesmo assim.
        assert est.prevalences[2] == pytest.approx(0.5, abs=0.05)


class TestFieldPriorStore:
    """Estado por talhão com esquecimento exponencial e persistência."""

    def _stream(self, store, fid, pi, n_batches, rng, t0=0.0):
        for i in range(n_batches):
            x, _ = sample(pi, 200, rng)
            store.update(fid, posteriors(x), timestamp=t0 + i)

    def test_unknown_field_returns_source_prior(self):
        """Talhão novo usa o prior do treino e ``adjust`` é a identidade."""
        store = FieldPriorStore(PI_S)
        np.testing.assert_allclose(store.priors("novo"), PI_S)
        p = posteriors(sample(PI_S, 5, np.random.default_rng(0))[0])
        np.testing.assert_allclose(store.adjust("novo", p), p)

    def test_tracks_prevalence_change(self):
        """Após mudança de estação (doença 70% → 5%), o estado com esquecimento acompanha.

        Com meia-vida de 1500 imagens e 8000 imagens após a mudança, o regime antigo ainda
        pesa 0,5^(8000/1500) ≈ 2,5%: esse é o atraso inerente ao esquecimento, daí a
        tolerância de 3 p.p. Sem esquecimento a estimativa fica presa no meio do caminho.
        """
        before = np.array([0.1, 0.1, 0.3, 0.1, 0.4])
        after = np.array([0.01, 0.01, 0.95, 0.01, 0.02])
        forgetful = FieldPriorStore(PI_S, TrackerConfig(half_life_samples=1500, method="em"))
        frozen = FieldPriorStore(PI_S, TrackerConfig(half_life_samples=None, method="em"))
        for store in (forgetful, frozen):
            r = np.random.default_rng(20)
            self._stream(store, "t1", before, 30, r)
            # Sanidade (Sadia x Sigatoka se confundem; ~2000 imagens efetivas).
            np.testing.assert_allclose(store.priors("t1"), before, atol=0.05)
            self._stream(store, "t1", after, 40, r, t0=100)
        err_forget = np.max(np.abs(forgetful.priors("t1") - after))
        err_frozen = np.max(np.abs(frozen.priors("t1") - after))
        assert err_forget < 0.03
        assert err_frozen > 0.2

    def test_divergence_is_unknown_before_enough_images(self):
        store = FieldPriorStore(np.array([0.9, 0.1]), TrackerConfig(method="em"))
        assert store.divergence("novo") is None
        store.update("t", np.array([[1.0, 0.0]] * 5))
        assert store.divergence("t") is None, "abaixo do mínimo ainda vale o prior do treino"

    def test_divergence_is_zero_when_the_model_is_sure(self):
        store = FieldPriorStore(np.array([0.9, 0.1]), TrackerConfig(method="em"))
        store.update("t", np.array([[1.0, 0.0]] * 60 + [[0.0, 1.0]] * 40))
        np.testing.assert_allclose(store.predicted_shares("t"), [0.6, 0.4])
        assert store.divergence("t") < DIVERGENCE_LIMIT

    def test_divergence_flags_the_em_amplifying_a_doubt(self):
        """Todo palpite diz a classe 0 a 60%, mas o treino tinha 10% da classe 1: o EM
        empurra a prevalência para a classe 1, que nenhum palpite escolheu. Foi o que
        aconteceu com Sigatoka e Cordana no campo da banana."""
        store = FieldPriorStore(np.array([0.9, 0.1]), TrackerConfig(method="em"))
        store.update("t", np.array([[0.6, 0.4]] * 100))
        np.testing.assert_allclose(store.predicted_shares("t"), [1.0, 0.0])
        assert store.divergence("t") > DIVERGENCE_LIMIT

    def test_fields_are_independent(self):
        """Talhões diferentes mantêm estimativas diferentes."""
        rng = np.random.default_rng(21)
        store = FieldPriorStore(PI_S, TrackerConfig(method="em"))
        self._stream(store, "a", PI_FARM, 10, rng)
        self._stream(store, "b", PI_S, 10, rng)
        assert store.priors("a")[HEALTHY] > 0.9
        assert store.priors("b")[HEALTHY] == pytest.approx(0.30, abs=0.05)

    def test_time_based_forgetting(self):
        """Com meia-vida em tempo, um intervalo longo sem imagens reduz o peso do passado."""
        store = FieldPriorStore(
            PI_S, TrackerConfig(half_life_samples=None, half_life_seconds=10.0, method="em")
        )
        rng = np.random.default_rng(22)
        store.update("t", posteriors(sample(PI_S, 1000, rng)[0]), timestamp=0.0)
        store.update("t", posteriors(sample(PI_FARM, 1000, rng)[0]), timestamp=100.0)
        # Passado decaído por 2^-10: prevalência quase só do segundo lote.
        assert store.priors("t")[HEALTHY] > 0.9

    def test_auto_method_with_miscalibrated_model(self):
        """Modo auto com summary descalibrado usa BBSE online e acerta onde o EM erraria."""
        rng = np.random.default_rng(23)
        xv, yv = sample(PI_S, 4000, rng)
        summary = ValidationSummary.from_validation(posteriors(xv, temperature=3.0), yv)
        store = FieldPriorStore(PI_S, TrackerConfig(method="auto"), summary)
        for _ in range(20):
            store.update("t", posteriors(sample(PI_FARM, 250, rng)[0], temperature=3.0))
        assert store.estimate("t").method == "bbse"
        assert np.max(np.abs(store.priors("t") - PI_FARM)) < 0.05

    def test_json_round_trip(self, tmp_path):
        """Salvar e carregar reproduz priors e correção; escrita atômica sem temporários."""
        rng = np.random.default_rng(24)
        xv, yv = sample(PI_S, 2000, rng)
        summary = ValidationSummary.from_validation(posteriors(xv), yv)
        names = ["Cordana", "Fusarium_Wilt", "Healthy", "Pestalotiopsis", "Sigatoka"]
        store = FieldPriorStore(PI_S, TrackerConfig(half_life_seconds=86400.0), summary, names)
        self._stream(store, "talhão-7", PI_FARM, 5, rng)
        self._stream(store, "t2", PI_S, 3, rng)
        path = tmp_path / "estado" / "priors.json"
        store.save(path)
        store.save(path)  # sobrescrita também é atômica
        assert [p.name for p in path.parent.iterdir()] == ["priors.json"]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["format_version"] == 1 and "talhão-7" in data["fields"]
        loaded = FieldPriorStore.load(path)
        assert loaded.field_ids() == store.field_ids()
        assert loaded.class_names == names
        for fid in store.field_ids():
            np.testing.assert_allclose(loaded.priors(fid), store.priors(fid))
        p = posteriors(sample(PI_FARM, 20, rng)[0])
        np.testing.assert_allclose(loaded.adjust("talhão-7", p), store.adjust("talhão-7", p))
        # Continuar atualizando o carregado dá o mesmo que continuar o original.
        batch = posteriors(sample(PI_FARM, 200, rng)[0])
        np.testing.assert_allclose(
            loaded.update("t2", batch, timestamp=1e9), store.update("t2", batch, timestamp=1e9)
        )

    def test_rejects_unknown_format_version(self):
        """Estado de versão desconhecida não é carregado silenciosamente."""
        d = FieldPriorStore(PI_S).to_dict()
        d["format_version"] = 99
        with pytest.raises(ValueError):
            FieldPriorStore.from_dict(d)


class TestDriftMonitor:
    """KS, PSI, qui-quadrado e MMD: sem alarme sem deriva, com alarme com deriva."""

    @staticmethod
    def _profile(rng, n=3000):
        x, y = sample(PI_S, n, rng)
        p = posteriors(x)
        return ReferenceProfile.build(
            p.max(1), novelty(x), p.argmax(1), K, labels=y, embeddings=embeddings(x, rng)
        )

    @staticmethod
    def _window(pi, n, rng, noise=1.0):
        x, _ = sample(pi, n, rng, noise)
        p = posteriors(x)
        return p.max(1), novelty(x), p.argmax(1), embeddings(x, rng)

    def test_ks_null_calibrated(self):
        """KS: taxa de rejeição a 5% sob H0 próxima de 5% (p-valor assintótico)."""
        rng = np.random.default_rng(30)
        rej = [ks_2samp(rng.normal(size=400), rng.normal(size=300))[1] < 0.05 for _ in range(400)]
        assert 0.02 < np.mean(rej) < 0.09

    def test_mmd_detects_shift_and_not_null(self):
        """MMD por permutação: p grande sob H0, p mínimo com deslocamento de média."""
        rng = np.random.default_rng(31)
        a, b = rng.normal(size=(100, 4)), rng.normal(size=(100, 4))
        _, p0 = mmd_permutation_test(a, b, n_permutations=300, rng=rng)
        _, p1 = mmd_permutation_test(a, b + 0.8, n_permutations=300, rng=rng)
        assert p0 > 0.01
        assert p1 < 0.01

    def test_psi_zero_for_identical(self):
        """PSI de distribuições iguais é ~0."""
        assert psi([0.25] * 4, [250, 250, 250, 250], pseudo_count=0) == pytest.approx(0.0)

    def test_false_alarm_controlled(self):
        """Sem deriva: nenhum 'deriva' e no máximo 10% de 'atenção' em 20 seeds."""
        levels = []
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            ref = self._profile(rng)
            conf, nov, pred, emb = self._window(PI_S, 500, rng)
            th = DriftThresholds(mmd_permutations=500)
            rep = check_drift(ref, conf, nov, pred, emb, thresholds=th, rng=rng)
            levels.append(rep.level)
        assert LEVEL_DRIFT not in levels
        assert levels.count(LEVEL_WARN) <= 2

    def test_detects_low_contrast_drift(self):
        """Câmera com baixo contraste (imagens "achatadas" no centro): confiança cai → deriva."""
        rng = np.random.default_rng(40)
        ref = self._profile(rng)
        x, _ = sample(PI_S, 500, rng)
        center = MEANS.mean(axis=0)
        x = center + 0.35 * (x - center)
        p = posteriors(x)
        rep = check_drift(ref, p.max(1), novelty(x), p.argmax(1), embeddings(x, rng), rng=rng)
        assert rep.level == LEVEL_DRIFT
        by_name = {t.name: t for t in rep.tests}
        for name in ("ks_confianca", "psi_confianca", "mmd_embeddings"):
            assert by_name[name].level == LEVEL_DRIFT, name
        assert any(r.startswith("confiança média caiu de 0,") for r in rep.reasons)

    def test_detects_unknown_disease(self):
        """Doença fora do treino (novo aglomerado distante) em 30% da janela → novidade dispara.

        Note que a confiança máxima NÃO cai nesse cenário (softmax extrapola com confiança
        alta longe dos dados — Hein et al., 2019); por isso o monitor não depende só dela.
        """
        rng = np.random.default_rng(41)
        ref = self._profile(rng)
        x, _ = sample(PI_S, 500, rng)
        n_new = 150
        x[:n_new] = np.array([5.0, -2.5]) + 0.7 * rng.normal(size=(n_new, 2))
        p = posteriors(x)
        rep = check_drift(ref, p.max(1), novelty(x), p.argmax(1), embeddings(x, rng), rng=rng)
        assert rep.level == LEVEL_DRIFT
        by_name = {t.name: t for t in rep.tests}
        for name in ("ks_novidade", "psi_novidade", "mmd_embeddings"):
            assert by_name[name].level == LEVEL_DRIFT, name
        text = " | ".join(rep.reasons)
        assert "PSI da novidade" in text and "fração com novidade > 1,0 subiu" in text

    def test_detects_class_histogram_change(self):
        """Qui-quadrado detecta mudança nas classes previstas (limitado a 'atenção')."""
        rng = np.random.default_rng(45)
        ref = self._profile(rng)
        conf, nov, pred, _ = self._window(PI_FARM, 500, rng)
        rep = check_drift(ref, conf, nov, pred)
        chi = next(t for t in rep.tests if t.name == "qui2_classes")
        assert chi.level == LEVEL_WARN and chi.p_value < 1e-3
        assert "compatível com mudança de prevalência" in chi.reason

    def test_label_shift_alone_is_not_drift_after_reweighting(self):
        """Mudança só de prevalência, com a referência reponderada pelo π_t estimado: 'ok'."""
        rng = np.random.default_rng(42)
        ref = self._profile(rng)
        target = np.array([0.05, 0.05, 0.8, 0.05, 0.05])
        x, _ = sample(target, 600, rng)
        p = posteriors(x)
        pi_hat = em_prior_estimate(p, PI_S).prevalences
        rep = check_drift(
            ref,
            p.max(1),
            novelty(x),
            p.argmax(1),
            embeddings(x, rng),
            target_prior=pi_hat,
            rng=rng,
        )
        assert rep.reweighted
        assert rep.level == LEVEL_OK, rep.reasons

    def test_small_window_skips_tests(self):
        """Janela abaixo do mínimo não roda testes (evita viés do PSI) e diz por quê."""
        rng = np.random.default_rng(43)
        ref = self._profile(rng, 500)
        conf, nov, pred, _ = self._window(PI_S, 50, rng)
        rep = check_drift(ref, conf, nov, pred)
        assert rep.level == LEVEL_OK and not rep.tests and "janela com 50" in rep.reasons[0]

    def test_rolling_monitor(self):
        """DriftMonitor acumula a janela e avalia; relatório serializa em JSON."""
        rng = np.random.default_rng(44)
        mon = DriftMonitor(self._profile(rng), window_size=400)
        for _ in range(4):
            x, _ = sample(PI_S, 150, rng, noise=1.8)
            mon.observe(posteriors(x), novelty(x))
        rep = mon.evaluate(rng=rng)
        assert rep.n_window == 400 and rep.level == LEVEL_DRIFT
        json.dumps(rep.to_dict(), ensure_ascii=False)
