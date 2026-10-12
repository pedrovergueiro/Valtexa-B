"""Testes da visibilidade das salvaguardas e da trava do fit_novelty.

Nasceram de um defeito real: a referência de novidade do ``leaf-foliar`` foi
sobrescrita por uma calibração de 3 classes, a API passou a recusá-la — e a
única pista era uma linha de log. O modelo serviu semanas sem novidade.
"""

from __future__ import annotations

import json

import pytest

from app.registry import registry
from tests.conftest import DEFAULT_MODEL, attach_fake
from tests.test_novelty import build_reference


def _catalogue(client) -> dict:
    return {entry["key"]: entry for entry in client.get("/models").json()}


class TestSafeguardsInTheCatalogue:
    def test_an_unloaded_model_reports_nothing(self, client):
        entry = _catalogue(client)[DEFAULT_MODEL]

        assert entry["runtime"] is None
        assert entry["safeguards"] == {}

    def test_reports_what_is_active_after_loading(self, client):
        attach_fake(DEFAULT_MODEL)
        registry.get(DEFAULT_MODEL)._novelty = build_reference()

        entry = _catalogue(client)[DEFAULT_MODEL]

        assert entry["runtime"] == "keras"
        assert entry["safeguards"] == {
            "novelty": True,
            "quality": False,
            "conformal": False,
            "framing": False,
            "species": False,
            "diagnosis_head": False,
        }

    def test_a_refused_reference_shows_up_as_off(self, client):
        """O caso real: pesos carregados, novidade recusada, tudo 'pronto'."""
        attach_fake(DEFAULT_MODEL)  # sem .novelty.npz válido

        entry = _catalogue(client)[DEFAULT_MODEL]

        assert entry["loaded"] is True
        assert entry["safeguards"]["novelty"] is False


class TestFitNoveltyRefusesMismatchedClasses:
    def test_refuses_to_calibrate_with_other_classes(self, tmp_path):
        from training.fit_novelty import _refuse_class_mismatch

        weights = tmp_path / "corn_disease_model.keras"
        weights.with_suffix(".labels.json").write_text(
            json.dumps(["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]), encoding="utf-8"
        )

        with pytest.raises(SystemExit, match="ficaria sem novidade"):
            _refuse_class_mismatch(weights, ["Blight", "Common_Rust", "Healthy"])

    def test_accepts_the_classes_of_the_weights(self, tmp_path):
        from training.fit_novelty import _refuse_class_mismatch

        weights = tmp_path / "m.keras"
        weights.with_suffix(".labels.json").write_text(json.dumps(["A", "B"]), encoding="utf-8")

        _refuse_class_mismatch(weights, ["A", "B"])  # não levanta

    def test_without_labels_there_is_nothing_to_compare(self, tmp_path):
        from training.fit_novelty import _refuse_class_mismatch

        _refuse_class_mismatch(tmp_path / "sem_rotulos.keras", ["A"])  # não levanta
