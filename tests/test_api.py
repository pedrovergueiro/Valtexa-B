"""Testes das rotas HTTP."""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.model import CornDiseaseModel, ModelConfigurationError
from app.registry import get_spec
from tests.conftest import DEFAULT_MODEL, FakeKerasModel, attach_fake, make_image_bytes

LEAF_CLASSES = list(get_spec(DEFAULT_MODEL).class_names)
ROVER = "rover-understory"


def _upload(name: str = "folha.jpg", content_type: str = "image/jpeg", data: bytes | None = None):
    return (name, data if data is not None else make_image_bytes(), content_type)


class TestHealth:
    def test_reports_ok_even_without_weights(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "model_loaded": False,
            "default_model": DEFAULT_MODEL,
            "loaded_models": [],
        }

    def test_lists_only_the_models_that_are_ready(self, client, fake_model):
        assert client.get("/health").json()["loaded_models"] == [DEFAULT_MODEL]


class TestModelCatalogue:
    def test_lists_every_model(self, client):
        catalogue = client.get("/models").json()
        assert {entry["key"] for entry in catalogue} == {
            DEFAULT_MODEL,
            "leaf-foliar-b",
            "coffee-foliar",
            ROVER,
            "aerial-canopy",
        }

    def test_reports_platform_and_readiness(self, client, fake_model):
        catalogue = {entry["key"]: entry for entry in client.get("/models").json()}

        assert catalogue[ROVER]["platform"] == "ground"
        assert catalogue["aerial-canopy"]["platform"] == "aerial"
        assert catalogue[DEFAULT_MODEL]["loaded"] is True
        assert catalogue[ROVER]["loaded"] is False

    def test_flags_models_that_need_field_data(self, client):
        catalogue = {entry["key"]: entry for entry in client.get("/models").json()}
        assert catalogue[DEFAULT_MODEL]["needs_field_data"] is False
        assert catalogue[ROVER]["needs_field_data"] is True


class TestClasses:
    def test_defaults_to_the_default_model(self, client):
        assert client.get("/classes").json() == LEAF_CLASSES

    def test_returns_classes_of_the_requested_model(self, client):
        rover_classes = client.get("/classes", params={"model": ROVER}).json()
        assert "Stalk_Rot" in rover_classes
        assert len(rover_classes) == 8

    def test_selects_by_platform(self, client):
        by_platform = client.get("/classes", params={"platform": "ground"}).json()
        assert by_platform == client.get("/classes", params={"model": ROVER}).json()

    def test_rejects_unknown_model(self, client):
        response = client.get("/classes", params={"model": "nao-existe"})
        assert response.status_code == 404
        assert "Disponíveis" in response.json()["detail"]

    def test_rejects_unknown_platform(self, client):
        assert client.get("/classes", params={"platform": "submarino"}).status_code == 422


class TestPredict:
    def test_returns_prediction(self, client, fake_model, image_bytes):
        response = client.post("/predict", files={"file": _upload(data=image_bytes)})

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "folha.jpg"
        assert body["model_key"] == DEFAULT_MODEL
        # O dublê concentra a massa na última classe.
        assert body["predicted_class"] == LEAF_CLASSES[-1]
        assert body["confidence"] == pytest.approx(1.0)
        assert body["abstained"] is False
        assert [p["class_name"] for p in body["probabilities"]] == LEAF_CLASSES
        assert sum(p["probability"] for p in body["probabilities"]) == pytest.approx(1.0)
        assert body["inference_ms"] >= 0

    def test_routes_to_the_requested_model(self, client, fake_model):
        attach_fake(ROVER)
        response = client.post("/predict", files={"file": _upload()}, params={"model": ROVER})
        assert response.json()["model_key"] == ROVER

    def test_routes_by_platform(self, client, fake_model):
        attach_fake(ROVER)
        response = client.post("/predict", files={"file": _upload()}, params={"platform": "ground"})
        assert response.json()["model_key"] == ROVER

    def test_rejects_when_the_requested_model_is_missing(self, client, fake_model):
        """O modelo padrão estar pronto não torna os outros disponíveis."""
        response = client.post("/predict", files={"file": _upload()}, params={"model": ROVER})
        assert response.status_code == 503
        assert ROVER in response.json()["detail"]

    def test_rejects_when_no_model_is_loaded(self, client):
        assert client.post("/predict", files={"file": _upload()}).status_code == 503

    def test_rejects_unknown_model(self, client, fake_model):
        response = client.post("/predict", files={"file": _upload()}, params={"model": "xpto"})
        assert response.status_code == 404

    def test_rejects_unsupported_content_type(self, client, fake_model):
        response = client.post("/predict", files={"file": _upload(content_type="application/pdf")})
        assert response.status_code == 415

    def test_rejects_upload_above_size_limit(self, client, fake_model, monkeypatch):
        monkeypatch.setattr(settings, "max_upload_mb", 1)
        oversized = b"\xff\xd8\xff" + b"0" * (2 * 1024 * 1024)
        response = client.post("/predict", files={"file": _upload(data=oversized)})
        assert response.status_code == 413

    def test_rejects_corrupt_image(self, client, fake_model):
        response = client.post("/predict", files={"file": _upload(data=b"nao sou uma imagem")})
        assert response.status_code == 400
        assert "folha.jpg" in response.json()["detail"]


class TestAbstention:
    def test_flags_low_confidence_results(self, client):
        """Sem isso, uma foto de solo sai como doença com confiança alta."""
        settings.min_confidence = 0.9
        attach_fake(DEFAULT_MODEL, confidence=0.4)

        body = client.post("/predict", files={"file": _upload()}).json()
        assert body["abstained"] is True
        assert body["confidence"] == pytest.approx(0.4)

    def test_does_not_flag_confident_results(self, client):
        settings.min_confidence = 0.9
        attach_fake(DEFAULT_MODEL, confidence=0.99)
        assert client.post("/predict", files={"file": _upload()}).json()["abstained"] is False

    def test_is_disabled_by_default(self, client, fake_model):
        assert settings.min_confidence == 0.0
        assert client.post("/predict", files={"file": _upload()}).json()["abstained"] is False


class TestPredictBatch:
    def test_runs_one_forward_pass_for_the_whole_batch(self, client, fake_model):
        files = [("files", _upload(f"voo_{i}.jpg")) for i in range(3)]
        response = client.post("/predict/batch", files=files)

        assert response.status_code == 200
        assert len(fake_model.batch_shapes) == 1, "o lote deveria ir ao modelo de uma só vez"
        assert fake_model.batch_shapes[0][0] == 3

    def test_summarises_predictions_per_class(self, client, fake_model):
        files = [("files", _upload(f"voo_{i}.jpg")) for i in range(3)]
        body = client.post("/predict/batch", files=files).json()

        assert body["model_key"] == DEFAULT_MODEL
        assert [r["filename"] for r in body["results"]] == ["voo_0.jpg", "voo_1.jpg", "voo_2.jpg"]
        assert body["summary"] == {
            name: (3 if name == LEAF_CLASSES[-1] else 0) for name in LEAF_CLASSES
        }
        assert body["abstained_count"] == 0

    def test_counts_abstentions(self, client):
        settings.min_confidence = 0.9
        attach_fake(DEFAULT_MODEL, confidence=0.3)
        files = [("files", _upload(f"voo_{i}.jpg")) for i in range(2)]
        assert client.post("/predict/batch", files=files).json()["abstained_count"] == 2

    def test_rejects_batch_above_limit(self, client, fake_model, monkeypatch):
        monkeypatch.setattr(settings, "max_batch_size", 2)
        files = [("files", _upload(f"voo_{i}.jpg")) for i in range(3)]
        assert client.post("/predict/batch", files=files).status_code == 413

    def test_rejects_batch_without_files(self, client, fake_model):
        assert client.post("/predict/batch").status_code == 422

    def test_fails_the_batch_when_one_image_is_corrupt(self, client, fake_model):
        files = [
            ("files", _upload("boa.jpg")),
            ("files", _upload("ruim.jpg", data=b"lixo")),
        ]
        response = client.post("/predict/batch", files=files)
        assert response.status_code == 400
        assert "ruim.jpg" in response.json()["detail"]


class TestLabels:
    def test_reads_labels_written_by_training(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "corn_disease_model.labels.json"
        labels = ["A", "B", "C", "D"]
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        monkeypatch.setattr(settings, "labels_path", labels_path)

        model = CornDiseaseModel(get_spec(DEFAULT_MODEL))
        model.attach(FakeKerasModel(), model._read_class_names())

        assert model.class_names == labels

    def test_falls_back_to_the_catalogue_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "labels_path", tmp_path / "ausente.json")
        model = CornDiseaseModel(get_spec(DEFAULT_MODEL))
        assert model._read_class_names() == tuple(LEAF_CLASSES)

    def test_falls_back_when_malformed(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "labels.json"
        labels_path.write_text('{"nao": "e uma lista"}', encoding="utf-8")
        monkeypatch.setattr(settings, "labels_path", labels_path)

        model = CornDiseaseModel(get_spec(DEFAULT_MODEL))
        assert model._read_class_names() == tuple(LEAF_CLASSES)

    def test_refuses_labels_that_do_not_match_the_model_outputs(self):
        """Rótulos a mais ou a menos trocariam as classes sem gerar erro."""
        model = CornDiseaseModel(get_spec(DEFAULT_MODEL))
        with pytest.raises(ModelConfigurationError, match="4 saídas"):
            model.attach(FakeKerasModel(num_classes=4), ["Blight", "Healthy"])

    def test_adopts_input_size_declared_by_the_model(self):
        model = CornDiseaseModel(get_spec(DEFAULT_MODEL))
        model.attach(FakeKerasModel(img_size=160), LEAF_CLASSES)
        assert model.img_size == 160
