"""Testes da predição conformal."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.inference.conformal import (
    ConformalReference,
    conformal_quantile,
    coverage_report,
    nonconformity,
)
from app.registry import get_spec, registry
from tests.conftest import DEFAULT_MODEL, attach_fake, make_image_bytes

CLASSES = get_spec(DEFAULT_MODEL).class_names


def calibrated_world(n: int, seed: int, classes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Um modelo perfeitamente calibrado: o rótulo é sorteado das próprias probabilidades.

    Com concentração baixa, parte das imagens sai ambígua — o caso em que um
    conjunto com mais de uma classe é a resposta honesta.
    """
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.full(classes, 0.4), size=n)
    labels = np.array([rng.choice(classes, p=row) for row in probabilities])
    return probabilities, labels


def _reference(alpha: float = 0.1) -> ConformalReference:
    probabilities, labels = calibrated_world(3000, seed=1)
    return ConformalReference.fit(
        {"plain": probabilities, "tta": probabilities}, labels, CLASSES, alpha=alpha
    )


class TestQuantile:
    def test_uses_the_finite_sample_rank(self):
        scores = np.arange(1, 20) / 20  # 19 escores
        # ceil((19 + 1) * 0.9) = 18 -> o 18º menor
        assert conformal_quantile(scores, alpha=0.1) == pytest.approx(18 / 20)

    def test_without_enough_samples_nothing_is_ruled_out(self):
        """Com 5 imagens não há como garantir 95%: a classe entra sempre."""
        assert conformal_quantile(np.full(5, 0.2), alpha=0.05) == 1.0

    def test_nonconformity_is_one_minus_the_true_class(self):
        probabilities = np.array([[0.7, 0.3], [0.1, 0.9]])
        assert nonconformity(probabilities, np.array([0, 0])) == pytest.approx([0.3, 0.9])


class TestGuarantee:
    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_covers_each_class_at_the_requested_level(self, alpha):
        """A promessa: a classe certa está no conjunto, classe por classe.

        A garantia é sobre a média das calibrações possíveis — uma calibração só
        oscila uns 1,5 ponto em volta dela. Por isso o teste repete a calibração
        com sementes independentes e confere a média, com folga de 3 erros-padrão.
        """
        per_class = []
        for seed in range(10):
            probabilities, labels = calibrated_world(3000, seed=100 + seed)
            reference = ConformalReference.fit(
                {"plain": probabilities}, labels, CLASSES, alpha=alpha
            )
            test_probabilities, test_labels = calibrated_world(4000, seed=900 + seed)
            report = coverage_report(reference, test_probabilities, test_labels, tta=False)
            per_class.append(list(report["coverage_per_class"].values()))

        for name, coverage in zip(CLASSES, np.mean(per_class, axis=0), strict=True):
            assert coverage >= 1 - alpha - 0.015, name

    def test_ambiguous_images_get_more_than_one_class(self):
        reference = _reference(0.05)
        [candidates] = reference.prediction_sets(np.array([[0.48, 0.47, 0.03, 0.02]]), tta=False)

        assert candidates[:2] == [CLASSES[0], CLASSES[1]]

    def test_a_clear_image_gets_a_single_class(self):
        reference = _reference(0.2)
        [candidates] = reference.prediction_sets(
            np.array([[0.001, 0.001, 0.001, 0.997]]), tta=False
        )

        assert candidates == [CLASSES[3]]

    def test_sets_are_ordered_from_most_to_least_likely(self):
        reference = ConformalReference(
            class_names=CLASSES,
            alpha=0.05,
            thresholds={"plain": [1.0, 1.0, 1.0, 1.0]},
            calibrated_on=0,
            class_counts=[0, 0, 0, 0],
        )
        [candidates] = reference.prediction_sets(np.array([[0.1, 0.2, 0.3, 0.4]]), tta=False)

        assert candidates == list(reversed(CLASSES))

    def test_nothing_plausible_gives_an_empty_set(self):
        reference = ConformalReference(
            class_names=CLASSES,
            alpha=0.05,
            thresholds={"plain": [0.1, 0.1, 0.1, 0.1]},
            calibrated_on=100,
            class_counts=[25] * 4,
        )
        assert reference.prediction_sets(np.array([[0.25] * 4]), tta=False) == [[]]

    def test_refuses_a_mode_that_was_not_calibrated(self):
        reference = ConformalReference(
            class_names=CLASSES,
            alpha=0.05,
            thresholds={"plain": [0.5] * 4},
            calibrated_on=0,
            class_counts=[0] * 4,
        )
        with pytest.raises(ValueError, match="tta"):
            reference.prediction_sets(np.full((1, 4), 0.25), tta=True)

    def test_refuses_mismatched_shapes(self):
        probabilities, labels = calibrated_world(100, seed=3)
        with pytest.raises(ValueError, match="forma"):
            ConformalReference.fit({"plain": probabilities[:, :3]}, labels, CLASSES)


class TestFieldWidening:
    """O campo alarga os limiares das classes que aparecem nele, e só delas."""

    def _field(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Campo pior que o dataset: na classe 0, o modelo tira 60% da probabilidade
        da classe certa e espalha nas outras."""
        probabilities, labels = calibrated_world(n, seed)
        keep = labels == 0
        shifted = probabilities[keep].copy()
        moved = 0.6 * shifted[:, 0]
        shifted[:, 0] -= moved
        shifted[:, 1:] += moved[:, None] / (shifted.shape[1] - 1)
        return shifted, labels[keep]

    def test_recovers_the_coverage_in_the_field(self):
        reference = _reference()
        calibration, labels = self._field(20000, seed=2)
        held_out, held_labels = self._field(20000, seed=3)

        before = coverage_report(reference, held_out, held_labels, tta=False)["coverage"]
        widened = reference.widened({"plain": calibration, "tta": calibration}, labels)
        after = coverage_report(widened, held_out, held_labels, tta=False)["coverage"]

        assert before < 0.9 - 0.05
        assert after >= 0.9 - 0.03

    def test_only_widens_and_only_the_classes_seen_in_the_field(self):
        reference = _reference()
        calibration, labels = self._field(2000, seed=4)

        widened = reference.widened({"plain": calibration}, labels)

        for mode in ("plain", "tta"):
            pairs = zip(reference.thresholds[mode], widened.thresholds[mode], strict=True)
            for before, after in pairs:
                assert after >= before
        assert widened.thresholds["tta"] == reference.thresholds["tta"], "modo sem campo"
        assert widened.thresholds["plain"][1:] == reference.thresholds["plain"][1:]
        assert widened.field_calibrated_on == len(labels)

    def test_the_field_count_survives_a_round_trip(self, tmp_path):
        calibration, labels = self._field(500, seed=5)
        widened = _reference().widened({"plain": calibration}, labels)
        path = tmp_path / "ref.conformal.json"
        widened.save(path)

        assert ConformalReference.load(path) == widened

    def test_widening_keeps_the_original_thresholds_for_the_fusion(self):
        """A resposta traz o conjunto honesto; a fusão por planta segue com o original."""
        reference = _reference()
        calibration, labels = self._field(3000, seed=6)
        widened = reference.widened({"plain": calibration, "tta": calibration}, labels)
        probabilities, _ = calibrated_world(500, seed=7)

        assert reference.base_thresholds is None
        assert widened.base_thresholds == reference.thresholds
        assert widened.prediction_sets(
            probabilities, False, base=True
        ) == reference.prediction_sets(probabilities, False)
        honest = widened.prediction_sets(probabilities, False)
        assert sum(len(s) for s in honest) > sum(
            len(s) for s in widened.prediction_sets(probabilities, False, base=True)
        ), "o conjunto honesto é maior onde o modelo duvida"

    def test_widening_twice_keeps_the_first_calibration_as_the_base(self):
        reference = _reference()
        calibration, labels = self._field(1500, seed=8)
        once = reference.widened({"plain": calibration}, labels)
        twice = once.widened({"plain": calibration}, labels)

        assert twice.base_thresholds == reference.thresholds

    def test_without_widening_base_sets_are_the_same_sets(self):
        reference = _reference()
        probabilities, _ = calibrated_world(300, seed=9)

        assert reference.prediction_sets(probabilities, True, base=True) == (
            reference.prediction_sets(probabilities, True)
        )

    def test_the_base_thresholds_survive_a_round_trip(self, tmp_path):
        calibration, labels = self._field(800, seed=10)
        widened = _reference().widened({"plain": calibration}, labels)
        path = tmp_path / "ref.conformal.json"
        widened.save(path)

        assert ConformalReference.load(path).base_thresholds == widened.base_thresholds

    def test_photos_never_straddle_the_two_halves(self):
        from pathlib import Path

        from training.fit_conformal import split_photos

        paths = [Path(f"Sigatoka/foto{i // 3}_r{i % 3}.jpg") for i in range(30)]
        first, second = split_photos(paths, [0] * 30, seed=1)

        def photos(indices):
            return {paths[i].name.rsplit("_r", 1)[0] for i in indices}

        assert sorted(first + second) == list(range(30))
        assert not photos(first) & photos(second)
        assert len(photos(first)) == len(photos(second)) == 5


class TestPersistence:
    def test_survives_a_round_trip(self, tmp_path):
        path = tmp_path / "m.conformal.json"
        _reference().save(path)
        assert ConformalReference.load(path) == _reference()

    def test_reports_an_incomplete_file(self, tmp_path):
        path = tmp_path / "m.conformal.json"
        path.write_text('{"class_names": ["a"]}', encoding="utf-8")
        with pytest.raises(ValueError, match="ilegível"):
            ConformalReference.load(path)


class TestInThePrediction:
    @pytest.fixture
    def calibrated(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        _reference(0.05).save(get_spec(DEFAULT_MODEL).conformal_path)
        return attach_fake(DEFAULT_MODEL, confidence=0.4)

    def test_is_null_without_calibration(self, client, fake_model):
        body = client.post(
            "/predict", files={"file": ("x.jpg", make_image_bytes(), "image/jpeg")}
        ).json()

        assert body["prediction_set"] is None
        assert body["set_coverage"] is None

    def test_returns_the_set_and_its_guarantee(self, client, calibrated):
        body = client.post(
            "/predict", files={"file": ("x.jpg", make_image_bytes(), "image/jpeg")}
        ).json()

        assert body["set_coverage"] == pytest.approx(0.95)
        assert body["prediction_set"][0] == body["predicted_class"]
        assert len(body["prediction_set"]) > 1, "0,4 contra 0,2: a dúvida aparece"

    def test_uses_the_thresholds_of_the_mode_that_ran(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        ConformalReference(
            class_names=CLASSES,
            alpha=0.05,
            thresholds={"plain": [1.0] * 4, "tta": [0.0] * 4},
            calibrated_on=0,
            class_counts=[0] * 4,
        ).save(get_spec(DEFAULT_MODEL).conformal_path)
        attach_fake(DEFAULT_MODEL, confidence=0.9)

        def post(**params):
            return client.post(
                "/predict",
                params=params,
                files={"file": ("x.jpg", make_image_bytes(), "image/jpeg")},
            ).json()

        assert len(post()["prediction_set"]) == 4
        assert post(tta=True)["prediction_set"] == []

    def test_counts_ambiguous_images_in_a_batch(self, client, calibrated):
        files = [("files", (f"v{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(3)]
        body = client.post("/predict/batch", files=files).json()

        assert body["ambiguous_count"] == 3
        assert body["empty_set_count"] == 0

    def test_ignores_a_calibration_for_other_classes(self, client, tmp_path, monkeypatch):
        """Limiar da classe errada é garantia da classe errada."""
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        probabilities, labels = calibrated_world(500, seed=4, classes=3)
        ConformalReference.fit({"plain": probabilities}, labels, ("A", "B", "C")).save(
            get_spec(DEFAULT_MODEL).conformal_path
        )
        attach_fake(DEFAULT_MODEL)

        assert registry.get(DEFAULT_MODEL)._conformal is None
