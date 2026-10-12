"""A fila de revisão em ordem de valor: o agrônomo rotula primeiro o que mais ensina."""

from __future__ import annotations

import json
import uuid

import pytest

from app.review import ReviewQueue

MODEL = "banana-foliar"


def _pending(root, probabilities: dict[str, float], novelty: float = 0.2, **extra) -> str:
    """Grava um item pendente como a fila grava: imagem + JSON ao lado."""
    item_id = uuid.uuid4().hex
    folder = root / "pending" / "2026-09-27"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{item_id}.jpg").write_bytes(b"\xff\xd8\xff")
    guess = max(probabilities, key=probabilities.get)
    metadata = {
        "id": item_id,
        "original_filename": f"{item_id}.jpg",
        "captured_at": "2026-09-27T12:00:00+00:00",
        "model_key": MODEL,
        "model_guess": guess,
        "confidence": probabilities[guess],
        "novelty_score": novelty,
        "probabilities": probabilities,
        "reviewed_label": None,
        "reviewed_by": None,
        **extra,
    }
    (folder / f"{item_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
    return item_id


CONFIDENT = {"Sigatoka": 0.97, "Healthy": 0.01, "Cordana": 0.01, "Fusarium_Wilt": 0.01}
AMBIGUOUS = {"Sigatoka": 0.46, "Cordana": 0.44, "Healthy": 0.05, "Fusarium_Wilt": 0.05}


class TestPrioritized:
    def test_the_ambiguous_leaf_comes_before_the_confident_one(self, tmp_path):
        confident = _pending(tmp_path, CONFIDENT)
        ambiguous = _pending(tmp_path, AMBIGUOUS)

        ranked = ReviewQueue(tmp_path).prioritized(budget=2)

        assert [meta["id"] for meta, _ in ranked] == [ambiguous, confident]

    def test_every_choice_says_why(self, tmp_path):
        _pending(tmp_path, AMBIGUOUS)
        (_, choice), *_ = ReviewQueue(tmp_path).prioritized(budget=1)

        assert choice.position == 1
        assert choice.reasons, "o agrônomo precisa saber por que o item está no topo"

    def test_a_strange_image_rises(self, tmp_path):
        usual = _pending(tmp_path, CONFIDENT, novelty=0.1)
        strange = _pending(tmp_path, CONFIDENT, novelty=3.0)

        ranked = ReviewQueue(tmp_path).prioritized(budget=2)

        assert ranked[0][0]["id"] == strange and ranked[1][0]["id"] == usual

    def test_an_empty_queue_gives_nothing(self, tmp_path):
        assert ReviewQueue(tmp_path).prioritized(budget=5) == []


class TestReviewRoute:
    @pytest.fixture
    def queue_dir(self, tmp_path, monkeypatch):
        import app.main

        monkeypatch.setattr(app.main, "review_queue", ReviewQueue(tmp_path))
        return tmp_path

    def test_order_by_value(self, client, queue_dir):
        _pending(queue_dir, CONFIDENT)
        ambiguous = _pending(queue_dir, AMBIGUOUS)

        body = client.get("/review", params={"order": "valor"}).json()

        assert body[0]["id"] == ambiguous
        assert body[0]["priority"] == 1 and body[0]["priority_reasons"]

    def test_the_default_order_is_still_by_date(self, client, queue_dir):
        _pending(queue_dir, CONFIDENT)
        body = client.get("/review").json()
        assert body[0]["priority"] is None
