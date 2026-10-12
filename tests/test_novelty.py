"""Testes da detecção de novidade e do ciclo de dados próprio."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.config import settings
from app.inference.novelty import NoveltyReference
from app.registry import SOURCE_CLASS_ALIASES, get_spec, normalize_class_name, registry
from app.review import InvalidClassNameError, ReviewQueue, UnknownReviewItemError
from tests.conftest import DEFAULT_MODEL, EMBEDDING_DIM, attach_fake, make_image_bytes
from training.prepare_dataset import execute_plan, plan_dataset

KNOWN = np.zeros(EMBEDDING_DIM)
"""Embedding que cai no centro da classe conhecida da referência de teste."""

FAR_AWAY = np.full(EMBEDDING_DIM, 50.0)
"""Embedding distante de tudo — o que um sintoma desconhecido produziria."""


def build_reference(num_classes: int = 4, samples: int = 400) -> NoveltyReference:
    """Referência sintética com classes bem separadas em torno da origem."""
    rng = np.random.default_rng(7)
    centers = rng.normal(0, 5, size=(num_classes, EMBEDDING_DIM))
    centers[0] = 0.0  # a primeira classe fica na origem, para KNOWN cair nela

    embeddings = np.vstack(
        [center + rng.normal(0, 1, size=(samples, EMBEDDING_DIM)) for center in centers]
    )
    labels = np.repeat(np.arange(num_classes), samples)
    names = tuple(get_spec(DEFAULT_MODEL).class_names)[:num_classes]
    return NoveltyReference.fit(embeddings, labels, names)


class TestNoveltyReference:
    def test_known_data_scores_below_the_threshold(self):
        reference = build_reference()
        assert reference.score(KNOWN[None, :])[0] < 1.0

    def test_unknown_data_scores_far_above_the_threshold(self):
        """É este caso que vira material para o agrônomo."""
        reference = build_reference()
        assert reference.score(FAR_AWAY[None, :])[0] > 5.0

    def test_threshold_matches_the_requested_percentile(self):
        """O limiar é um orçamento: 99 => ~1% do treino marcado."""
        rng = np.random.default_rng(3)
        embeddings = rng.normal(0, 1, size=(2000, EMBEDDING_DIM))
        labels = np.zeros(2000, dtype=int)
        reference = NoveltyReference.fit(embeddings, labels, ("Healthy",), percentile=99.0)

        flagged = (reference.score(embeddings) > 1.0).mean()
        assert flagged == pytest.approx(0.01, abs=0.005)

    def test_scores_a_batch_in_one_call(self):
        reference = build_reference()
        scores = reference.score(np.vstack([KNOWN, FAR_AWAY, KNOWN]))
        assert scores.shape == (3,)
        assert scores[1] > scores[0] and scores[1] > scores[2]

    def test_survives_a_save_load_round_trip(self, tmp_path):
        reference = build_reference()
        path = tmp_path / "modelo.novelty.npz"
        reference.save(path)
        restored = NoveltyReference.load(path)

        assert restored.class_names == reference.class_names
        assert restored.fitted_on == reference.fitted_on
        np.testing.assert_allclose(
            restored.score(FAR_AWAY[None, :]), reference.score(FAR_AWAY[None, :])
        )

    def test_rejects_a_corrupt_reference_file(self, tmp_path):
        path = tmp_path / "quebrado.novelty.npz"
        path.write_bytes(b"isto nao e um npz")
        with pytest.raises(ValueError, match="inválida"):
            NoveltyReference.load(path)

    def test_refuses_to_fit_with_fewer_samples_than_dimensions(self):
        """Menos amostras que dimensões dá uma covariância sem sentido."""
        embeddings = np.zeros((5, EMBEDDING_DIM))
        with pytest.raises(ValueError, match="Amostras de menos"):
            NoveltyReference.fit(embeddings, np.zeros(5, dtype=int), ("Healthy",))

    def test_refuses_mismatched_labels(self):
        with pytest.raises(ValueError, match="rótulos"):
            NoveltyReference.fit(np.zeros((100, 4)), np.zeros(50, dtype=int), ("Healthy",))


class TestNoveltyThroughTheApi:
    def _load(self, embedding, num_classes: int = 4):
        model = attach_fake(DEFAULT_MODEL)
        model.embedding = embedding
        from app.registry import registry

        registry.get(DEFAULT_MODEL)._novelty = build_reference(num_classes)
        return model

    def test_flags_an_unknown_image_for_review(self, client):
        self._load(FAR_AWAY)
        body = client.post(
            "/predict", files={"file": ("nova.jpg", make_image_bytes(), "image/jpeg")}
        ).json()

        assert body["needs_review"] is True
        assert body["novelty_score"] > 1.0

    def test_flagged_image_without_client_consent_stays_out_of_the_queue(
        self, client, tmp_path, monkeypatch
    ):
        """Brief, seção 32: X-Training-Consent: 0 mantém a imagem fora do treino."""
        import app.main

        queue = ReviewQueue(tmp_path / "revisao")
        monkeypatch.setattr(app.main, "review_queue", queue)
        self._load(FAR_AWAY)
        arquivo = {"file": ("nova.jpg", make_image_bytes(), "image/jpeg")}

        recusado = client.post("/predict", files=arquivo, headers={"X-Training-Consent": "0"})
        assert recusado.json()["needs_review"] is True
        assert queue.pending() == []

        arquivo = {"file": ("nova.jpg", make_image_bytes(), "image/jpeg")}
        client.post("/predict", files=arquivo, headers={"X-Training-Consent": "1"})
        assert len(queue.pending()) == 1

    def test_does_not_flag_a_familiar_image(self, client):
        self._load(KNOWN)
        body = client.post(
            "/predict", files={"file": ("ok.jpg", make_image_bytes(), "image/jpeg")}
        ).json()

        assert body["needs_review"] is False
        assert body["novelty_score"] < 1.0

    def test_sensitivity_widens_the_review_queue(self, client):
        """O botão que regula o tempo do agrônomo, sem recalibrar nada."""
        self._load(KNOWN)
        settings.novelty_sensitivity = 0.01

        body = client.post(
            "/predict", files={"file": ("ok.jpg", make_image_bytes(), "image/jpeg")}
        ).json()
        assert body["needs_review"] is True

    def test_reports_null_when_no_reference_is_calibrated(self, client, fake_model):
        body = client.post(
            "/predict", files={"file": ("x.jpg", make_image_bytes(), "image/jpeg")}
        ).json()

        assert body["novelty_score"] is None
        assert body["needs_review"] is False

    def test_counts_flagged_images_in_a_batch(self, client):
        self._load(FAR_AWAY)
        files = [("files", (f"v{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(3)]
        body = client.post("/predict/batch", files=files).json()

        assert body["needs_review_count"] == 3

    def test_embedding_comes_from_the_same_forward_pass(self, client):
        """Extrair a assinatura visual não pode custar uma segunda inferência."""
        model = self._load(FAR_AWAY)
        files = [("files", (f"v{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(4)]
        client.post("/predict/batch", files=files)

        assert len(model.batch_shapes) == 1
        assert model.batch_shapes[0][0] == 4


class TestReviewQueue:
    def _prediction(self, client):
        from app.registry import registry

        attach_fake(DEFAULT_MODEL).embedding = FAR_AWAY
        registry.get(DEFAULT_MODEL)._novelty = build_reference()
        return registry.get(DEFAULT_MODEL).predict("folha.jpg", make_image_bytes())

    def test_is_disabled_without_a_directory(self):
        assert ReviewQueue(None).enabled is False
        assert ReviewQueue(None).enqueue(b"x", None, "image/jpeg") is None

    def test_writes_the_image_and_its_metadata(self, client, tmp_path):
        prediction = self._prediction(client)
        queue = ReviewQueue(tmp_path)

        item_id = queue.enqueue(make_image_bytes(), prediction, "image/jpeg")

        assert item_id is not None
        images = list((tmp_path / "pending").rglob("*.jpg"))
        sidecars = list((tmp_path / "pending").rglob("*.json"))
        assert len(images) == 1 and len(sidecars) == 1

        metadata = json.loads(sidecars[0].read_text(encoding="utf-8"))
        assert metadata["id"] == item_id
        assert metadata["original_filename"] == "folha.jpg"
        assert metadata["novelty_score"] > 1.0
        assert metadata["reviewed_label"] is None, "o agrônomo ainda não diagnosticou"

    def test_keeps_the_extension_of_the_uploaded_type(self, client, tmp_path):
        prediction = self._prediction(client)
        queue = ReviewQueue(tmp_path)
        queue.enqueue(make_image_bytes(image_format="PNG"), prediction, "image/png")

        assert list((tmp_path / "pending").rglob("*.png"))

    def test_survives_an_unwritable_directory(self, client, tmp_path):
        """Disco cheio não pode virar erro 500 para o cliente."""
        blocker = tmp_path / "bloqueado"
        blocker.write_text("sou um arquivo, nao um diretorio", encoding="utf-8")

        assert ReviewQueue(blocker).enqueue(b"x", self._prediction(client), "image/jpeg") is None


class TestReviewApi:
    """As rotas que a interface de revisão vai consumir."""

    def _queued(self, client, tmp_path):
        """Enfileira um item e devolve a fila e o id dele."""
        attach_fake(DEFAULT_MODEL).embedding = FAR_AWAY
        registry.get(DEFAULT_MODEL)._novelty = build_reference()
        prediction = registry.get(DEFAULT_MODEL).predict("folha.jpg", make_image_bytes())

        queue = ReviewQueue(tmp_path)
        return queue, queue.enqueue(make_image_bytes(), prediction, "image/jpeg")

    def test_lists_what_is_waiting_for_the_agronomist(self, client, tmp_path):
        queue, item_id = self._queued(client, tmp_path)

        pending = queue.pending()

        assert [item["id"] for item in pending] == [item_id]
        assert pending[0]["model_guess"], "o agrônomo decide melhor sabendo o palpite do modelo"
        assert pending[0]["novelty_score"] > 1.0

    def test_counts_the_queue(self, client, tmp_path):
        queue, _ = self._queued(client, tmp_path)

        stats = queue.stats()

        assert stats == {
            "enabled": True,
            "pending": 1,
            "labeled": 0,
            "oldest_pending": stats["oldest_pending"],
        }

    def test_labelling_moves_the_image_into_the_training_layout(self, client, tmp_path):
        queue, item_id = self._queued(client, tmp_path)

        metadata = queue.label(item_id, "Cigarrinha", reviewed_by="agrônoma Ana")

        assert metadata["reviewed_label"] == "Cigarrinha"
        assert metadata["reviewed_by"] == "agrônoma Ana"
        assert list((tmp_path / "labeled" / "Cigarrinha").glob("*.jpg"))
        assert queue.stats() == {
            "enabled": True,
            "pending": 0,
            "labeled": 1,
            "oldest_pending": None,
        }

    def test_accepts_a_class_the_catalogue_does_not_have(self, client, tmp_path):
        """Descobrir uma doença fora do catálogo é o objetivo da fila."""
        queue, item_id = self._queued(client, tmp_path)
        assert "Cigarrinha" not in get_spec(DEFAULT_MODEL).class_names

        queue.label(item_id, "Cigarrinha")

        assert (tmp_path / "labeled" / "Cigarrinha").is_dir()

    @pytest.mark.parametrize(
        "malicioso", ["../../etc", "..", "a/b", "com espaço", "", "_comeca_errado", "x" * 80]
    )
    def test_refuses_a_label_that_is_not_a_safe_folder_name(self, client, tmp_path, malicioso):
        """O rótulo vira diretório — sem isto um `../` escreveria fora da fila."""
        queue, item_id = self._queued(client, tmp_path)

        with pytest.raises(InvalidClassNameError):
            queue.label(item_id, malicioso)

    @pytest.mark.parametrize("malicioso", ["../../../etc/passwd", "nao-existe", "..", ""])
    def test_refuses_an_id_that_is_not_a_queued_item(self, client, tmp_path, malicioso):
        """O id vem da URL e vira caminho: precisa casar com o formato do uuid4."""
        queue, _ = self._queued(client, tmp_path)

        with pytest.raises(UnknownReviewItemError):
            queue.image_path(malicioso)

    def test_reports_the_queue_as_off_when_there_is_no_directory(self):
        assert ReviewQueue(None).stats() == {
            "enabled": False,
            "pending": 0,
            "labeled": 0,
            "oldest_pending": None,
        }
        assert ReviewQueue(None).pending() == []

    def test_the_api_answers_503_while_the_queue_is_off(self, client):
        """Sem REVIEW_DIR não há o que listar — e isso não é erro do cliente."""
        for rota in ("/review", "/review/stats"):
            assert client.get(rota).status_code in (200, 503)

    def test_a_corrupt_item_does_not_break_the_listing(self, client, tmp_path):
        queue, _ = self._queued(client, tmp_path)
        quebrado = next((tmp_path / "pending").rglob("*.json")).parent / "quebrado.json"
        quebrado.write_text("{ nao e json", encoding="utf-8")

        assert len(queue.pending()) == 1, "o item bom continua listado"


class TestFlywheelClosesTheLoop:
    """O ciclo só fecha se o `prepare_dataset` ler o que o agrônomo rotulou."""

    def test_every_catalogue_class_is_a_valid_folder_name(self):
        """O agrônomo move o arquivo para uma pasta com o nome da classe."""
        from app.registry import MODEL_SPECS

        for spec in MODEL_SPECS.values():
            for name in spec.class_names:
                assert SOURCE_CLASS_ALIASES[normalize_class_name(name)] == name

    def test_reviewed_images_flow_back_into_the_training_set(self, tmp_path):
        labeled = tmp_path / "review" / "labeled" / "Common_Rust"
        labeled.mkdir(parents=True)
        for index in range(3):
            (labeled / f"{index}.jpg").write_bytes(make_image_bytes(size=(8, 8)))
        # O sidecar do agrônomo não pode virar imagem de treino.
        (labeled / "0.json").write_text("{}", encoding="utf-8")

        plan, unmapped = plan_dataset([tmp_path / "review" / "labeled"], get_spec(DEFAULT_MODEL))
        counts = execute_plan(plan, tmp_path / "data")

        assert unmapped == []
        assert counts == {"Common_Rust": 3}
