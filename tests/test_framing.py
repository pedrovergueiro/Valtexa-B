"""Testes do portão de enquadramento."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from PIL import Image

from app.framing import SCENES, FramingModel, FramingReference, framing_model, judge
from tests.conftest import DEFAULT_MODEL, attach_fake, make_image_bytes

KEYS = tuple(scene.key for scene in SCENES)


def _reference(threshold: float = 0.5) -> FramingReference:
    """Cada cena é um eixo: um embedding alinhado a ela é aquela cena, sem dúvida."""
    return FramingReference(
        model_id="dublê",
        keys=KEYS,
        prompts=tuple(scene.prompt for scene in SCENES),
        in_scope=np.array([scene.in_scope for scene in SCENES]),
        crops=tuple(scene.crop or "" for scene in SCENES),
        text_embeddings=np.eye(len(SCENES)),
        logit_scale=100.0,
        threshold=threshold,
        calibrated_on=10,
    )


def _axis(key: str) -> np.ndarray:
    vector = np.zeros(len(SCENES))
    vector[KEYS.index(key)] = 1.0
    return vector


def brightness_encoder(images: list[Image.Image]) -> np.ndarray:
    """Dublê da torre de visão: foto clara é vista aérea; escura, folha de milho."""
    rows = []
    for image in images:
        bright = np.asarray(image.convert("L")).mean() > 128
        rows.append(_axis("aereo") if bright else _axis("folha_milho"))
    return np.array(rows)


@pytest.fixture
def gate() -> Iterator[FramingModel]:
    framing_model.attach(brightness_encoder, _reference())
    yield framing_model
    framing_model.unload()


def _post(client, color, model: str = DEFAULT_MODEL):
    return client.post(
        "/predict",
        params={"model": model},
        files={"file": ("x.jpg", make_image_bytes(color=color), "image/jpeg")},
    ).json()


class TestJudge:
    def test_a_leaf_is_in_scope_with_its_crop(self):
        [result] = judge(_reference(), _axis("folha_cafe")[None])

        assert result.in_scope is True
        assert result.scene == "folha_cafe"
        assert result.crop == "coffee"

    def test_an_aerial_photo_is_out_of_scope(self):
        [result] = judge(_reference(), _axis("aereo")[None])

        assert result.in_scope is False
        assert result.score < 0.01
        assert result.scene == "aereo"

    def test_the_threshold_decides_the_borderline(self):
        between = (_axis("folha") + _axis("colmo")) / 2  # meio a meio

        [strict] = judge(_reference(threshold=0.6), between[None])
        [lenient] = judge(_reference(threshold=0.4), between[None])

        assert strict.in_scope is False and lenient.in_scope is True

    def test_probabilities_sum_to_one(self):
        rng = np.random.default_rng(0)
        probabilities = _reference().scene_probabilities(rng.normal(size=(5, len(SCENES))))

        assert probabilities.sum(axis=1) == pytest.approx(np.ones(5))

    def test_scenes_cover_the_documented_failure_modes(self):
        """Aérea, colmo e slide são os casos medidos que passavam como diagnóstico."""
        out_of_scope = {scene.key for scene in SCENES if not scene.in_scope}
        assert {"aereo", "colmo", "texto", "inseto", "folha_outra"} <= out_of_scope


class TestPersistence:
    def test_survives_a_round_trip_without_pickle(self, tmp_path):
        path = tmp_path / "framing.npz"
        _reference().save(path)
        loaded = FramingReference.load(path)

        assert loaded.keys == KEYS
        assert loaded.threshold == pytest.approx(0.5)
        assert np.array_equal(loaded.in_scope, _reference().in_scope)
        np.testing.assert_allclose(loaded.text_embeddings, np.eye(len(SCENES)))

    def test_reports_an_unreadable_file(self, tmp_path):
        path = tmp_path / "framing.npz"
        path.write_bytes(b"nada")
        with pytest.raises(ValueError, match="ilegível"):
            FramingReference.load(path)


class TestFramingModel:
    def test_is_silent_without_a_reference(self, tmp_path):
        model = FramingModel(tmp_path / "ausente.npz")
        model.load()

        assert model.is_loaded is False
        assert model.assess([Image.new("RGB", (8, 8))]) == [None]

    def test_encodes_in_blocks_of_max_forward_batch(self, monkeypatch):
        """Um quadro do rover tem centenas de recortes; a torre de visão não pode
        recebê-los de uma vez (~13,5 MB de RAM por recorte)."""
        from app.config import settings

        monkeypatch.setattr(settings, "max_forward_batch", 16)
        sizes: list[int] = []

        def counting(images):
            sizes.append(len(images))
            return brightness_encoder(images)

        model = FramingModel()
        model.attach(counting, _reference())
        results = model.assess([Image.new("RGB", (8, 8))] * 70)

        assert len(results) == 70
        assert max(sizes) == 16
        assert sum(sizes) == 70

    def test_a_failing_encoder_does_not_break_the_prediction(self):
        def broken(images):
            raise RuntimeError("GPU sem memória")

        model = FramingModel()
        model.attach(broken, _reference())
        assert model.assess([Image.new("RGB", (8, 8))]) == [None]


class TestInThePrediction:
    def test_passes_a_leaf_of_the_right_crop(self, client, gate):
        attach_fake(DEFAULT_MODEL)
        framing = _post(client, (20, 60, 20))["framing"]

        assert framing["in_scope"] is True
        assert framing["crop"] == "corn"
        assert framing["crop_matches"] is True

    def test_refuses_an_aerial_photo(self, client, gate):
        attach_fake(DEFAULT_MODEL)
        framing = _post(client, (230, 230, 230))["framing"]

        assert framing["in_scope"] is False
        assert framing["scene"] == "aereo"

    def test_flags_a_leaf_sent_to_the_model_of_another_crop(self, client, gate):
        """Folha de milho no modelo de café: responderia uma doença de café."""
        attach_fake("coffee-foliar")
        framing = _post(client, (20, 60, 20), model="coffee-foliar")["framing"]

        assert framing["crop"] == "corn"
        assert framing["crop_matches"] is False

    def test_does_not_apply_to_models_that_expect_another_framing(self, client, gate):
        attach_fake("rover-understory")
        assert _post(client, (20, 60, 20), model="rover-understory")["framing"] is None

    def test_is_null_without_the_gate(self, client, fake_model):
        assert _post(client, (20, 60, 20))["framing"] is None

    def test_counts_refused_images_in_a_batch(self, client, gate):
        attach_fake(DEFAULT_MODEL)
        files = [
            ("files", ("aerea.jpg", make_image_bytes(color=(230, 230, 230)), "image/jpeg")),
            ("files", ("folha.jpg", make_image_bytes(color=(20, 60, 20)), "image/jpeg")),
        ]
        body = client.post("/predict/batch", files=files).json()

        assert body["out_of_scope_count"] == 1


class TestGatePerChosenModel:
    """Com o modelo escolhido, a folha disputa só com a folha da cultura dele."""

    def mixed(self, weights: dict[str, float]) -> np.ndarray:
        """Embedding que mistura as cenas dadas (é normalizado antes da comparação)."""
        vector = np.zeros(len(SCENES))
        for key, weight in weights.items():
            vector[KEYS.index(key)] = weight
        return vector[None, :]

    def test_leaf_of_another_crop_leaves_the_competition(self):
        """Café escolhido: 'folha de citros' não entra na conta."""
        reference = _reference()
        allowed = reference.allowed_scenes("coffee")

        assert allowed[KEYS.index("folha_cafe")]
        assert not allowed[KEYS.index("folha_citros")]
        assert allowed[KEYS.index("aereo")], "o que não é folha continua valendo"

    def test_a_coffee_leaf_that_looks_like_citrus_still_reads_as_coffee(self):
        # O embedding é normalizado: 1,00 contra 0,99 dá ~2x a favor do citros.
        embedding = self.mixed({"folha_citros": 1.0, "folha_cafe": 0.99})

        (unrestricted,) = judge(_reference(), embedding)
        (chosen,) = judge(_reference(), embedding, crop="coffee")

        assert unrestricted.scene == "folha_citros"
        assert chosen.scene == "folha_cafe"
        assert chosen.in_scope
        assert chosen.crop_matches, "folga pequena não é cultura trocada"

    def test_a_leaf_clearly_of_another_crop_is_flagged(self):
        (result,) = judge(_reference(), _axis("folha_citros")[None, :], crop="coffee")
        assert result.crop_matches is False

    def test_uses_the_threshold_of_the_chosen_crop(self):
        import dataclasses

        reference = dataclasses.replace(_reference(threshold=0.99), thresholds={"banana": 0.01})
        embedding = self.mixed({"folha_banana": 0.02, "aereo": 0.02})

        (general,) = judge(reference, embedding)
        (banana,) = judge(reference, embedding, crop="banana")

        assert not general.in_scope
        assert banana.in_scope

    def test_crop_thresholds_survive_a_round_trip(self, tmp_path):
        import dataclasses

        reference = dataclasses.replace(_reference(), thresholds={"citrus": 0.2, "banana": 0.1})
        reference.save(tmp_path / "f.npz")

        assert FramingReference.load(tmp_path / "f.npz").thresholds == {
            "banana": 0.1,
            "citrus": 0.2,
        }

    def test_an_old_reference_without_crop_thresholds_still_loads(self, tmp_path):
        _reference().save(tmp_path / "f.npz")
        with np.load(tmp_path / "f.npz") as raw:
            old = {k: raw[k] for k in raw.files if not k.startswith("threshold_")}
        np.savez(tmp_path / "old.npz", **old)

        assert FramingReference.load(tmp_path / "old.npz").thresholds == {}


def _mistura(principal: str, folha: str = "folha_cafe", peso: float = 0.97) -> np.ndarray:
    """A cena ``principal`` vence, mas ``folha`` leva massa suficiente para passar no limiar."""
    return _axis(principal) + peso * _axis(folha)


def test_cena_vencedora_tem_de_ser_folha_so_no_cafe(tmp_path) -> None:
    import dataclasses

    base = _reference(threshold=0.05)
    estrita = dataclasses.replace(base, leaf_winner_crops=("coffee",))
    gente = np.array([_mistura("pessoa")])
    # O limiar sozinho deixa passar a pessoa na frente do cafeeiro...
    assert judge(base, gente, crop="coffee")[0].in_scope
    veredito = judge(estrita, gente, crop="coffee")[0]
    assert not veredito.in_scope and veredito.scene == "pessoa"
    # ...e a regra vale só para as culturas listadas.
    assert judge(estrita, gente)[0].in_scope
    # Folha de outra planta vencendo continua passando: é folha (custo medido: 13,6% → 3,4%).
    assert judge(estrita, np.array([_mistura("folha_outra")]), crop="coffee")[0].in_scope
    assert judge(estrita, np.array([_axis("folha_cafe")]), crop="coffee")[0].in_scope

    estrita.save(tmp_path / "f.npz")
    assert FramingReference.load(tmp_path / "f.npz").leaf_winner_crops == ("coffee",)
    base.save(tmp_path / "antiga.npz")
    with np.load(tmp_path / "antiga.npz") as raw:  # um framing.npz de antes desta regra
        np.savez(tmp_path / "sem_chave.npz", **{k: raw[k] for k in raw.files
                                                 if k != "leaf_winner_crops"})
    assert FramingReference.load(tmp_path / "sem_chave.npz").leaf_winner_crops == ()
