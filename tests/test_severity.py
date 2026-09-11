"""Severidade: fração do quadro com lesão, ao lado da classificação."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from PIL import Image

from app.severity import SeverityModel, lesion_fraction, severity_model
from tests.conftest import attach_fake


class FakeSegmenter:
    """Marca como lesão uma faixa fixa do quadro."""

    def __init__(self, fraction: float, img_size: int = 64) -> None:
        self.input_shape = (None, img_size, img_size, 3)
        self.fraction = fraction
        self.batch_shapes: list[tuple[int, ...]] = []

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        self.batch_shapes.append(batch.shape)
        size = batch.shape[1]
        masks = np.zeros((batch.shape[0], size, size, 1), dtype=np.float32)
        masks[:, : round(size * self.fraction), :, 0] = 0.9
        return masks


@pytest.fixture
def segmenter() -> Iterator[FakeSegmenter]:
    double = FakeSegmenter(fraction=0.25)
    severity_model.attach(double)
    yield double
    severity_model.unload()


def _upload(client, image_bytes: bytes, **params):
    return client.post(
        "/predict", params=params, files={"file": ("folha.jpg", image_bytes, "image/jpeg")}
    )


class TestLesionFraction:
    def test_counts_pixels_above_the_threshold(self):
        probabilities = np.zeros((2, 4, 4, 1))
        probabilities[0, :2] = 0.8
        probabilities[1, 0, 0] = 0.4
        assert lesion_fraction(probabilities) == [0.5, 0.0]

    def test_accepts_masks_without_channel_axis(self):
        assert lesion_fraction(np.ones((1, 3, 3))) == [1.0]

    def test_rejects_a_single_mask_without_batch_axis(self):
        with pytest.raises(ValueError, match="lote"):
            lesion_fraction(np.ones((3, 3)))


class TestSeverityModel:
    def test_without_weights_it_stays_off(self, tmp_path):
        model = SeverityModel(path=tmp_path / "nao-existe.keras")
        model.load()
        assert not model.is_loaded

    def test_unloaded_model_answers_none_for_each_image(self, tmp_path):
        model = SeverityModel(path=tmp_path / "nao-existe.keras")
        assert model.estimate([Image.new("RGB", (10, 10))] * 2) == [None, None]

    def test_resizes_to_the_segmenter_input(self, tmp_path):
        model = SeverityModel(path=tmp_path / "x.keras")
        double = FakeSegmenter(fraction=0.5, img_size=32)
        model.attach(double)
        assert model.estimate([Image.new("RGB", (100, 70))]) == [0.5]
        assert double.batch_shapes == [(1, 32, 32, 3)]


class TestPredictReportsSeverity:
    def test_leaf_model_reports_the_lesion_fraction(self, client, image_bytes, segmenter):
        attach_fake("leaf-foliar")
        body = _upload(client, image_bytes).json()
        assert body["severity"] == pytest.approx(0.25)

    def test_second_architecture_reports_it_too(self, client, image_bytes, segmenter):
        attach_fake("leaf-foliar-b")
        body = _upload(client, image_bytes, model="leaf-foliar-b").json()
        assert body["severity"] == pytest.approx(0.25)

    def test_models_that_do_not_measure_severity_answer_null(self, client, image_bytes, segmenter):
        """A rede aprendeu lesão de milho; aplicá-la a café seria inventar número."""
        attach_fake("coffee-foliar")
        body = _upload(client, image_bytes, model="coffee-foliar").json()
        assert body["severity"] is None
        assert segmenter.batch_shapes == []

    def test_without_the_segmenter_severity_is_null(self, client, image_bytes, fake_model):
        body = _upload(client, image_bytes).json()
        assert body["severity"] is None
