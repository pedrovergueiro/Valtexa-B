"""Testes do TTA na inferência."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.registry import get_spec, registry
from tests.conftest import DEFAULT_MODEL, FakeKerasModel, attach_fake, make_image_bytes
from tests.test_novelty import FAR_AWAY, KNOWN, build_reference

CLASSES = get_spec(DEFAULT_MODEL).class_names


class OrientationSensitiveModel(FakeKerasModel):
    """Responde a última classe quando a metade esquerda é a mais clara.

    Espelhar na horizontal troca a resposta; na vertical, não. É o modelo que
    muda de ideia só porque a foto foi girada.
    """

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        self.batch_shapes.append(batch.shape)
        half = batch.shape[2] // 2
        left_brighter = batch[:, :, :half].mean(axis=(1, 2, 3)) > batch[:, :, half:].mean(
            axis=(1, 2, 3)
        )
        rows = np.full((len(batch), 4), 0.1)
        rows[left_brighter, 3] = 0.7
        rows[~left_brighter, 0] = 0.7
        return rows

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Estranho quando a esquerda é a mais clara; conhecido no espelho."""
        half = batch.shape[2] // 2
        left_brighter = batch[:, :, :half].mean(axis=(1, 2, 3)) > batch[:, :, half:].mean(
            axis=(1, 2, 3)
        )
        return np.where(left_brighter[:, None], FAR_AWAY, KNOWN)


def _half_bright_image() -> bytes:
    pixels = np.zeros((240, 320, 3), dtype=np.uint8)
    pixels[:, :160] = 220
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _post(client, data: bytes | None = None, content_type: str = "image/jpeg", **params):
    return client.post(
        "/predict",
        params=params,
        files={"file": ("folha", data or make_image_bytes(), content_type)},
    ).json()


class TestTta:
    def test_runs_the_four_views_in_a_single_forward_pass(self, client, fake_model):
        files = [("files", (f"v{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(3)]
        client.post("/predict/batch", params={"tta": True}, files=files)

        assert [shape[0] for shape in fake_model.batch_shapes] == [12]

    def test_is_off_by_default(self, client, fake_model):
        body = _post(client)

        assert fake_model.batch_shapes[0][0] == 1
        assert body["view_agreement"] is None

    def test_follows_the_service_default_when_the_request_is_silent(
        self, client, fake_model, monkeypatch
    ):
        monkeypatch.setattr(settings, "inference_tta", True)

        assert _post(client)["view_agreement"] == 1.0
        assert fake_model.batch_shapes[0][0] == 4

    def test_the_request_overrides_the_service_default(self, client, fake_model, monkeypatch):
        monkeypatch.setattr(settings, "inference_tta", True)

        assert _post(client, tta=False)["view_agreement"] is None

    def test_reports_when_the_views_disagree(self, client):
        """Metade das orientações vota numa classe e metade na outra."""
        double = OrientationSensitiveModel()
        registry.get(DEFAULT_MODEL).attach(double, CLASSES)

        body = _post(client, _half_bright_image(), "image/png", tta=True)

        assert body["view_agreement"] == pytest.approx(0.5)
        assert body["confidence"] == pytest.approx(0.4)

    def test_without_tta_the_same_image_looks_certain(self, client):
        """É o que o TTA revela: sem ele, a dúvida não aparece."""
        registry.get(DEFAULT_MODEL).attach(OrientationSensitiveModel(), CLASSES)

        body = _post(client, _half_bright_image(), "image/png")

        assert body["predicted_class"] == CLASSES[-1]
        assert body["confidence"] == pytest.approx(0.7)

    def test_novelty_is_scored_on_the_original_view_only(self, client):
        """A referência foi calibrada sem espelhamento; o score tem de bater.

        O dublê devolve um embedding estranho só na orientação original. Uma
        média das quatro vistas o puxaria para perto do conhecido, e a imagem
        deixaria de ir para o agrônomo só porque o TTA foi ligado.
        """
        registry.get(DEFAULT_MODEL).attach(OrientationSensitiveModel(), CLASSES)
        registry.get(DEFAULT_MODEL)._novelty = build_reference()

        plain = _post(client, _half_bright_image(), "image/png")
        with_tta = _post(client, _half_bright_image(), "image/png", tta=True)

        assert plain["needs_review"] is True
        assert with_tta["novelty_score"] == pytest.approx(plain["novelty_score"])
        assert with_tta["needs_review"] is True

    def test_a_plain_model_agrees_with_itself(self, client):
        attach_fake(DEFAULT_MODEL)

        assert _post(client, tta=True)["view_agreement"] == 1.0


class TestForwardBatchLimit:
    """O teto de imagens por passada — o que evita a VRAM transbordar e travar a máquina."""

    def test_splits_the_tta_views_into_bounded_passes(self, client, fake_model, monkeypatch):
        monkeypatch.setattr(settings, "max_forward_batch", 5)
        files = [("files", (f"v{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(3)]

        client.post("/predict/batch", params={"tta": True}, files=files)

        assert [shape[0] for shape in fake_model.batch_shapes] == [5, 5, 2]

    def test_splitting_does_not_change_any_answer(self, client, monkeypatch):
        registry.get(DEFAULT_MODEL).attach(OrientationSensitiveModel(), CLASSES)
        registry.get(DEFAULT_MODEL)._novelty = build_reference()
        files = [("files", (f"v{i}.png", _half_bright_image(), "image/png")) for i in range(3)]

        whole = client.post("/predict/batch", params={"tta": True}, files=files).json()
        monkeypatch.setattr(settings, "max_forward_batch", 1)
        split = client.post("/predict/batch", params={"tta": True}, files=files).json()

        for a, b in zip(whole["results"], split["results"], strict=True):
            assert a["probabilities"] == b["probabilities"]
            assert a["view_agreement"] == b["view_agreement"]
            assert a["novelty_score"] == pytest.approx(b["novelty_score"])
