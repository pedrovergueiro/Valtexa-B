"""O seletor de modelos: cada cultura com as suas versões, escolhidas na chamada."""

from __future__ import annotations

import pytest

from app.config import settings
from app.registry import default_for_crop, get_spec, registry, specs_by_crop
from tests.conftest import FakeKerasModel, make_image_bytes


@pytest.fixture
def rover_models(monkeypatch):
    """Troca o ROVER_MODELS durante o teste."""

    def set_models(mapping: dict[str, str]) -> None:
        monkeypatch.setattr(settings, "rover_models", mapping)

    set_models({})
    return set_models


def _attach(key: str) -> None:
    spec = get_spec(key)
    registry.get(key).attach(FakeKerasModel(num_classes=spec.num_classes), spec.class_names)


class TestCatalogue:
    def test_every_crop_has_exactly_one_recommended_version(self):
        for crop, options in specs_by_crop().items():
            assert sum(spec.recommended for spec in options) == 1, crop

    def test_the_recommended_version_comes_first(self):
        for options in specs_by_crop().values():
            assert options[0].recommended

    def test_banana_and_citrus_have_a_commercial_version(self):
        for crop in ("banana", "citrus"):
            variants = {spec.variant: spec for spec in specs_by_crop()[crop]}
            assert variants["comercial"].license == "CC BY 4.0"
            assert "não comercial" in variants["campo"].license

    def test_versions_of_a_crop_share_the_classes(self):
        for options in specs_by_crop().values():
            assert len({spec.class_names for spec in options}) == 1


class TestRoverDefault:
    def test_is_the_recommended_version(self, rover_models):
        assert default_for_crop("banana").key == "banana-foliar"

    def test_rover_models_switches_the_version(self, rover_models):
        rover_models({"banana": "banana-foliar-cc"})
        assert default_for_crop("banana").key == "banana-foliar-cc"

    def test_refuses_a_model_of_another_crop(self, rover_models):
        rover_models({"banana": "citrus-foliar"})
        with pytest.raises(KeyError, match="citrus"):
            default_for_crop("banana")

    def test_refuses_an_unknown_crop(self, rover_models):
        with pytest.raises(KeyError, match="soja"):
            default_for_crop("soja")


class TestChoicesRoute:
    def test_groups_versions_by_crop(self, client, rover_models):
        body = {entry["crop"]: entry for entry in client.get("/models/choices").json()}

        banana = body["banana"]
        assert [o["key"] for o in banana["options"]] == ["banana-foliar", "banana-foliar-cc"]
        assert banana["rover_default"] == "banana-foliar"
        assert banana["options"][0]["recommended"] is True
        assert banana["options"][0]["field_accuracy"] > banana["options"][1]["field_accuracy"]

    def test_says_which_versions_are_ready(self, client, rover_models):
        _attach("citrus-foliar-cc")
        citrus = next(e for e in client.get("/models/choices").json() if e["crop"] == "citrus")
        loaded = {o["key"]: o["loaded"] for o in citrus["options"]}
        assert loaded == {"citrus-foliar": False, "citrus-foliar-cc": True}

    def test_models_carries_the_picker_fields(self, client):
        entry = next(e for e in client.get("/models").json() if e["key"] == "banana-foliar-cc")
        assert entry["display_name"] == "Banana · comercial"
        assert entry["variant"] == "comercial"
        assert entry["crop"] == "banana"


class TestRoverFrameByCrop:
    def _post(self, client, **params):
        query = {"camera_height_mm": 450, "distance_mm": 200, **params}
        return client.post(
            "/predict/rover-frame",
            params=query,
            files={"file": ("q.jpg", make_image_bytes(size=(1000, 750)), "image/jpeg")},
        )

    def test_crop_uses_the_configured_version(self, client, rover_models):
        rover_models({"banana": "banana-foliar-cc"})
        _attach("banana-foliar-cc")

        body = self._post(client, crop="banana").json()

        assert body["model_key"] == "banana-foliar-cc"

    def test_an_explicit_model_wins_over_crop(self, client, rover_models):
        _attach("banana-foliar")
        body = self._post(client, crop="citrus", model="banana-foliar").json()
        assert body["model_key"] == "banana-foliar"

    def test_needs_model_or_crop(self, client):
        response = self._post(client)
        assert response.status_code == 422
        assert "crop" in response.json()["detail"]

    def test_unknown_crop_is_404(self, client, rover_models):
        assert self._post(client, crop="soja").status_code == 404
