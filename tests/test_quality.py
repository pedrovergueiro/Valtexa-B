"""Testes da verificação de qualidade da captura."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.inference.quality import (
    BLURRY,
    DARK,
    LOW_CONTRAST,
    OVEREXPOSED,
    CaptureMetrics,
    QualityReference,
    measure,
)
from app.registry import get_spec, registry
from tests.conftest import DEFAULT_MODEL, attach_fake, make_image_bytes


def _unit(pixels: np.ndarray) -> np.ndarray:
    """Pixels ``[0, 255]`` na escala do modelo, como lote de uma imagem."""
    return (np.asarray(pixels, dtype=np.float64) / 127.5 - 1.0)[None]


def _textured_bytes(seed: int, size: int = 256, base: int = 110) -> bytes:
    """Uma imagem com detalhe fino, bem exposta — o que uma folha boa parece."""
    import io

    rng = np.random.default_rng(seed)
    pixels = np.clip(base + rng.normal(0, 40, (size, size, 3)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _reference(**overrides) -> QualityReference:
    values = {
        "img_size": 224,
        "min_sharpness": 50.0,
        "min_brightness": 60.0,
        "max_brightness": 190.0,
        "min_contrast": 15.0,
        "max_highlights": 0.05,
        "percentile": 0.5,
        "fitted_on": 1000,
    }
    values.update(overrides)
    return QualityReference(**values)


GOOD = CaptureMetrics(sharpness=400.0, brightness=120.0, contrast=45.0, highlights=0.0)


class TestMeasure:
    def test_flat_color_has_no_detail_and_no_contrast(self):
        [metrics] = measure(_unit(np.full((32, 32, 3), 140)))

        assert metrics.sharpness == pytest.approx(0.0)
        assert metrics.contrast == pytest.approx(0.0)
        assert metrics.brightness == pytest.approx(140.0, abs=0.01)

    def test_fine_pattern_is_sharp(self):
        checker = (np.indices((32, 32)).sum(axis=0) % 2 * 255)[..., None].repeat(3, axis=2)
        [metrics] = measure(_unit(checker))

        assert metrics.sharpness > 10_000
        assert metrics.contrast == pytest.approx(127.5, abs=0.5)

    def test_counts_blown_out_pixels(self):
        half = np.zeros((32, 32, 3))
        half[:, :16] = 255
        [metrics] = measure(_unit(half))

        assert metrics.highlights == pytest.approx(0.5)

    def test_measures_each_image_of_the_batch(self):
        batch = np.concatenate([_unit(np.full((8, 8, 3), v)) for v in (20, 200)])
        dark, bright = measure(batch)

        assert dark.brightness < bright.brightness

    @pytest.mark.parametrize("shape", [(8, 8, 3), (1, 8, 8), (1, 2, 2, 3)])
    def test_rejects_what_is_not_a_measurable_batch(self, shape):
        with pytest.raises(ValueError):
            measure(np.zeros(shape))


class TestReference:
    def test_good_capture_has_no_issue(self):
        assert _reference().assess(GOOD) == []

    @pytest.mark.parametrize(
        ("change", "issue"),
        [
            ({"sharpness": 10.0}, BLURRY),
            ({"brightness": 30.0}, DARK),
            ({"brightness": 230.0}, OVEREXPOSED),
            ({"contrast": 3.0}, LOW_CONTRAST),
        ],
    )
    def test_flags_each_problem_on_its_own(self, change, issue):
        metrics = CaptureMetrics(**{**GOOD.__dict__, **change})
        assert _reference().assess(metrics) == [issue]

    def test_blown_out_sky_counts_even_with_normal_average(self):
        """Folha bem exposta e céu queimado num canto: a média engana."""
        metrics = CaptureMetrics(**{**GOOD.__dict__, "highlights": 0.3})
        assert _reference().assess(metrics) == [OVEREXPOSED]

    def test_limits_come_from_the_tails_of_the_training_set(self):
        metrics = [
            CaptureMetrics(sharpness=float(i), brightness=100.0, contrast=40.0, highlights=0.0)
            for i in range(1, 201)
        ]
        reference = QualityReference.fit(metrics, img_size=224, percentile=1.0)

        assert reference.min_sharpness == pytest.approx(np.percentile(range(1, 201), 1.0))
        assert reference.fitted_on == 200
        assert reference.img_size == 224

    def test_refuses_a_tail_smaller_than_one_image(self):
        with pytest.raises(ValueError, match="percentil"):
            QualityReference.fit([GOOD] * 50, img_size=224, percentile=0.5)

    @pytest.mark.parametrize("percentile", [0.0, 50.0, -1.0])
    def test_refuses_a_meaningless_percentile(self, percentile):
        with pytest.raises(ValueError):
            QualityReference.fit([GOOD] * 1000, img_size=224, percentile=percentile)

    def test_survives_a_round_trip_to_disk(self, tmp_path):
        path = tmp_path / "ref.quality.json"
        _reference().save(path)
        assert QualityReference.load(path) == _reference()

    def test_reports_an_unreadable_file(self, tmp_path):
        path = tmp_path / "ref.quality.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="ilegível"):
            QualityReference.load(path)


class TestQualityInThePrediction:
    @pytest.fixture
    def with_reference(self, client, tmp_path, monkeypatch):
        """Grava uma referência onde o catálogo a procura e carrega o modelo."""
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        _reference().save(get_spec(DEFAULT_MODEL).quality_path)
        attach_fake(DEFAULT_MODEL)

    def test_is_null_without_a_calibrated_reference(self, client, fake_model):
        body = client.post(
            "/predict", files={"file": ("x.jpg", make_image_bytes(), "image/jpeg")}
        ).json()

        assert body["quality"] is None

    def test_flags_a_flat_capture_and_says_what_to_do(self, client, with_reference):
        """A cor chapada que escapava da novidade e da confiança."""
        body = client.post(
            "/predict", files={"file": ("x.jpg", make_image_bytes(), "image/jpeg")}
        ).json()

        assert body["quality"]["ok"] is False
        assert LOW_CONTRAST in body["quality"]["issues"]
        assert len(body["quality"]["actions"]) == len(body["quality"]["issues"])

    def test_passes_a_detailed_capture(self, client, with_reference):
        body = client.post(
            "/predict", files={"file": ("x.png", _textured_bytes(0), "image/png")}
        ).json()

        assert body["quality"]["ok"] is True
        assert body["quality"]["issues"] == []

    def test_counts_captures_to_redo_in_a_batch(self, client, with_reference):
        files = [
            ("files", ("chapada.jpg", make_image_bytes(), "image/jpeg")),
            ("files", ("boa.png", _textured_bytes(1), "image/png")),
        ]
        body = client.post("/predict/batch", files=files).json()

        assert body["quality_issue_count"] == 1

    def test_ignores_a_reference_measured_at_another_resolution(
        self, client, tmp_path, monkeypatch
    ):
        """Nitidez a 384 px não se compara com nitidez a 224 px."""
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        _reference(img_size=384).save(get_spec(DEFAULT_MODEL).quality_path)
        attach_fake(DEFAULT_MODEL)

        assert registry.get(DEFAULT_MODEL).checks_quality is False


class TestFitQualityCli:
    def test_calibrates_and_writes_next_to_the_weights(self, tmp_path, monkeypatch, capsys):
        from training import fit_quality

        data_dir = tmp_path / "data"
        for index, class_name in enumerate(("Healthy", "Rust")):
            folder = data_dir / class_name
            folder.mkdir(parents=True)
            for seed in range(25):
                (folder / f"{seed}.png").write_bytes(_textured_bytes(seed + 100 * index))
        (data_dir / "Rust" / "quebrada.jpg").write_bytes(b"isto nao e imagem")

        weights = tmp_path / "experimento.keras"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fit_quality",
                "--model",
                DEFAULT_MODEL,
                "--data-dir",
                str(data_dir),
                "--weights",
                str(weights),
                "--percentile",
                "5",
            ],
        )
        fit_quality.main()

        written = Path(weights.with_suffix(".quality.json"))
        reference = json.loads(written.read_text(encoding="utf-8"))
        assert reference["img_size"] == get_spec(DEFAULT_MODEL).img_size
        # 10 das 51 ficam de fora para medir; a quebrada é pulada onde cair.
        assert reference["fitted_on"] in (40, 41)

        report = capsys.readouterr().out
        assert "Alarme falso" in report
        assert "desfoque" in report
