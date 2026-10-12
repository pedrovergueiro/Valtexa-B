"""Testes das cabeças sobre o embedding do portão (:mod:`app.inference.heads`)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.framing import SCENES, FramingModel, framing_model
from app.inference.heads import DiagnosisHead, EmbeddingMLP, SpeciesCheck, softmax
from app.registry import get_spec
from tests.conftest import attach_fake, make_image_bytes
from tests.test_framing import _axis, _reference

WIDTH = len(SCENES)
"""O dublê da torre de visão devolve um eixo por cena (ver tests/test_framing.py)."""


def _mlp(
    rng: np.random.Generator, outputs: int, hidden: tuple[int, ...] = (6,), **extra
) -> EmbeddingMLP:
    sizes = [WIDTH, *hidden]
    return EmbeddingMLP(
        mean=rng.normal(size=WIDTH) * 0.01,
        std=1.0 + rng.random(WIDTH),
        weights=tuple(rng.normal(size=(sizes[i + 1], sizes[i])) for i in range(len(hidden))),
        biases=tuple(rng.normal(size=sizes[i + 1]) for i in range(len(hidden))),
        activations=extra.pop("activations", ("relu",) * len(hidden)),
        out_weight=rng.normal(size=(outputs, sizes[-1])),
        out_bias=rng.normal(size=outputs),
        **extra,
    )


def _constant(outputs: int, bias: list[float]) -> EmbeddingMLP:
    """Sem camada oculta e sem pesos: a resposta é só o viés, qualquer que seja a foto."""
    return EmbeddingMLP(
        mean=np.zeros(WIDTH),
        std=np.ones(WIDTH),
        weights=(),
        biases=(),
        activations=(),
        out_weight=np.zeros((outputs, WIDTH)),
        out_bias=np.array(bias, dtype=float),
    )


class TestEmbeddingMLP:
    def test_matches_the_torch_network_it_came_from(self):
        """BatchNorm dobrado, GELU, salto identidade e salto com projeção, como o dNATY."""
        torch = pytest.importorskip("torch")
        nn = torch.nn

        class Dynamic(nn.Module):
            """Réplica da ``dnaty.core.arch.DynamicMLP`` (Linear -> BN -> ativação, saltos)."""

            def __init__(self):
                super().__init__()
                self.layer_sizes = [WIDTH, 8, 8, 5]
                self.activations = ["gelu", "relu", "tanh"]
                acts = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}
                layers = []
                for i in range(3):
                    layers += [
                        nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]),
                        nn.BatchNorm1d(self.layer_sizes[i + 1]),
                        acts[self.activations[i]](),
                    ]
                layers.append(nn.Linear(5, 3))
                self.net = nn.Sequential(*layers)
                self.skip_projs = nn.ModuleList([nn.Linear(WIDTH, 5)])
                self.skip_connections = [(1, 2, None), (0, 3, 0)]

            def forward(self, x):
                outputs, idx = [x], 0
                for i in range(len(self.layer_sizes) - 1):
                    out = self.net[idx + 2](self.net[idx + 1](self.net[idx](outputs[-1])))
                    for src, dst, proj in self.skip_connections:
                        if dst == i + 1 and src < len(outputs):
                            skip = (
                                outputs[src]
                                if proj is None
                                else self.skip_projs[proj](outputs[src])
                            )
                            if skip.shape == out.shape:
                                out = out + skip
                    outputs.append(out)
                    idx += 3
                return self.net[idx](outputs[-1])

        torch.manual_seed(0)
        model = Dynamic()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.uniform_(-1, 1)
                module.running_var.uniform_(0.5, 2)
                module.weight.data.uniform_(0.5, 1.5)
                module.bias.data.uniform_(-0.5, 0.5)
        model.eval()
        rng = np.random.default_rng(0)
        mean, std = rng.normal(size=WIDTH) * 0.01, 1.0 + rng.random(WIDTH)
        raw = rng.normal(size=(7, WIDTH)) * 5
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        with torch.no_grad():
            expected = model(torch.tensor((unit - mean) / std, dtype=torch.float32)).numpy()

        mlp = EmbeddingMLP.from_dnaty(model, mean, std)

        np.testing.assert_allclose(mlp.logits(raw), expected, atol=1e-5)
        assert mlp.layer_sizes == [WIDTH, 8, 8, 5, 3]

    def test_reads_a_raw_embedding_of_any_scale(self):
        mlp = _mlp(np.random.default_rng(1), 3)
        x = np.random.default_rng(2).normal(size=(4, WIDTH))

        np.testing.assert_allclose(mlp.logits(x), mlp.logits(10 * x))

    def test_refuses_an_embedding_of_another_width(self):
        with pytest.raises(ValueError, match="espera"):
            _mlp(np.random.default_rng(1), 3).logits(np.ones((2, WIDTH + 1)))

    def test_survives_a_round_trip(self, tmp_path):
        rng = np.random.default_rng(3)
        mlp = _mlp(rng, 4, hidden=(5, 5), activations=("gelu", "sigmoid"), skips=((1, 2, -1),))
        np.savez(tmp_path / "mlp.npz", **mlp.to_arrays())
        with np.load(tmp_path / "mlp.npz") as raw:
            loaded = EmbeddingMLP.from_arrays(raw)
        x = rng.normal(size=(3, WIDTH))

        np.testing.assert_allclose(loaded.probabilities(x), mlp.probabilities(x))

    def test_rejects_an_unknown_activation(self, tmp_path):
        arrays = _mlp(np.random.default_rng(4), 2).to_arrays()
        arrays["activations"] = np.array(["swish"])
        with pytest.raises(ValueError, match="swish"):
            EmbeddingMLP.from_arrays(arrays)


def _head(**changes) -> DiagnosisHead:
    names = get_spec("coffee-foliar").class_names
    fields = {
        "class_names": tuple(names),
        "mlp": _constant(len(names), [8.0] + [0.0] * (len(names) - 1)),
        "encoder": "dublê",
        "mix": {"cnn": 1.0, "head": 1.0},
        "trained_on": 10,
        "license": "CC BY 4.0",
    }
    fields.update(changes)
    return DiagnosisHead(**fields)


class TestDiagnosisHead:
    def test_mixes_by_weighted_geometric_mean(self):
        head = _head()
        cnn = np.array([[0.1, 0.1, 0.1, 0.1, 0.6]])
        own = head.mlp.probabilities(np.ones((1, WIDTH)))

        expected = softmax(np.log(cnn) / 2 + np.log(own) / 2)

        np.testing.assert_allclose(head.combine(cnn, np.ones((1, WIDTH))), expected)

    def test_a_reading_with_zero_weight_is_left_out(self):
        head = _head(mix={"cnn": 0.0, "head": 1.0})
        cnn = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])

        np.testing.assert_allclose(
            head.combine(cnn, np.ones((1, WIDTH))), head.mlp.probabilities(np.ones((1, WIDTH)))
        )

    def test_the_zero_shot_reading_follows_the_phrases(self):
        names = get_spec("coffee-foliar").class_names
        texts = np.eye(len(names), WIDTH)  # a frase da classe k aponta para o eixo k
        head = _head(
            mlp=_constant(len(names), [0.0] * len(names)),
            mix={"head": 1.0, "zero_shot": 1.0},
            text_embeddings=texts,
            logit_scale=50.0,
        )
        x = np.zeros((1, WIDTH))
        x[0, 2] = 1.0

        assert head.readings(x)["zero_shot"].argmax() == 2
        assert head.combine(np.full((1, len(names)), 0.2), x).argmax() == 2

    def test_survives_a_round_trip(self, tmp_path):
        names = get_spec("coffee-foliar").class_names
        head = _head(text_embeddings=np.eye(len(names), WIDTH), logit_scale=12.5)
        head.save(tmp_path / "x.head.npz")
        loaded = DiagnosisHead.load(tmp_path / "x.head.npz")
        x = np.random.default_rng(5).normal(size=(2, WIDTH))
        cnn = np.full((2, len(names)), 0.2)

        assert loaded.class_names == head.class_names and loaded.mix == head.mix
        np.testing.assert_allclose(loaded.combine(cnn, x), head.combine(cnn, x))

    def test_reports_an_unreadable_file(self, tmp_path):
        (tmp_path / "x.head.npz").write_bytes(b"nada")
        with pytest.raises(ValueError, match="ilegível"):
            DiagnosisHead.load(tmp_path / "x.head.npz")

    def test_refuses_outputs_that_do_not_match_the_classes(self, tmp_path):
        _head(mlp=_constant(3, [0.0, 0.0, 0.0])).save(tmp_path / "x.head.npz")
        with pytest.raises(ValueError, match="saídas"):
            DiagnosisHead.load(tmp_path / "x.head.npz")


class TestFitHeadsHelpers:
    def test_the_four_views_are_the_ones_of_the_tta(self):
        from training.fit_heads import four_views

        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        views = [np.asarray(v) for v in four_views(Image.fromarray(pixels), 4)]

        np.testing.assert_array_equal(views[0], pixels)
        np.testing.assert_array_equal(views[1], pixels[:, ::-1])
        np.testing.assert_array_equal(views[2], pixels[::-1])
        np.testing.assert_array_equal(views[3], pixels[::-1, ::-1])

    def test_mix_defaults_to_equal_weights_and_refuses_unknown_readings(self):
        from training.fit_heads import parse_mix

        assert parse_mix(None, ["cnn", "head"]) == {"cnn": 1.0, "head": 1.0}
        assert parse_mix("cnn=0,head=1", ["cnn", "head"]) == {"cnn": 0.0, "head": 1.0}
        with pytest.raises(SystemExit, match="zero_shot"):
            parse_mix("zero_shot=1", ["cnn", "head"])


class TestSpeciesCheck:
    def test_the_threshold_decides(self):
        check = SpeciesCheck(
            crop="citrus", mlp=_constant(2, [0.0, 1.0]), threshold=0.6, encoder="dublê"
        )
        score = check.scores(np.ones((1, WIDTH)))[0]

        assert score == pytest.approx(1 / (1 + np.exp(-1.0)))
        assert check.matches(np.ones((1, WIDTH)))[0] == (score >= 0.6)

    def test_survives_a_round_trip(self, tmp_path):
        check = SpeciesCheck(
            crop="citrus", mlp=_mlp(np.random.default_rng(6), 2), threshold=0.3, encoder="dublê"
        )
        check.save(tmp_path / "x.species.npz")
        loaded = SpeciesCheck.load(tmp_path / "x.species.npz")
        x = np.random.default_rng(7).normal(size=(3, WIDTH))

        assert loaded.crop == "citrus" and loaded.threshold == pytest.approx(0.3)
        np.testing.assert_allclose(loaded.scores(x), check.scores(x))

    def test_needs_two_outputs(self, tmp_path):
        SpeciesCheck(crop="citrus", mlp=_constant(3, [0, 0, 0]), threshold=0.5, encoder="d").save(
            tmp_path / "x.species.npz"
        )
        with pytest.raises(ValueError, match="2 saídas"):
            SpeciesCheck.load(tmp_path / "x.species.npz")


def leaf_encoder(images: list[Image.Image]) -> np.ndarray:
    """Dublê da torre: foto escura é folha de citros; clara, vista aérea."""
    rows = []
    for image in images:
        dark = np.asarray(image.convert("L")).mean() <= 128
        rows.append(_axis("folha_citros") if dark else _axis("aereo"))
    return np.array(rows)


@pytest.fixture
def models_dir(tmp_path, monkeypatch) -> Iterator:
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    yield tmp_path


@pytest.fixture
def gate() -> Iterator[FramingModel]:
    framing_model.attach(leaf_encoder, _reference())
    yield framing_model
    framing_model.unload()


def _post(client, model: str, color=(20, 60, 20)) -> dict:
    return client.post(
        "/predict",
        params={"model": model},
        files={"file": ("x.jpg", make_image_bytes(color=color), "image/jpeg")},
    ).json()


class TestInThePrediction:
    def test_the_head_changes_the_answer(self, client, models_dir, gate):
        """A CNN dublê põe 0,6 na última classe; a cabeça tem certeza da primeira."""
        _head().save(get_spec("coffee-foliar").head_path)
        attach_fake("coffee-foliar", confidence=0.6)

        body = _post(client, "coffee-foliar")
        catalogue = {entry["key"]: entry for entry in client.get("/models").json()}

        assert body["predicted_class"] == get_spec("coffee-foliar").class_names[0]
        assert sum(item["probability"] for item in body["probabilities"]) == pytest.approx(1.0)
        assert catalogue["coffee-foliar"]["safeguards"]["diagnosis_head"] is True

    def test_without_the_gate_the_cnn_answers_alone(self, client, models_dir):
        _head().save(get_spec("coffee-foliar").head_path)
        attach_fake("coffee-foliar", confidence=0.6)

        body = _post(client, "coffee-foliar")

        assert body["predicted_class"] == get_spec("coffee-foliar").class_names[-1]

    def test_a_head_of_another_vision_tower_is_ignored(self, client, models_dir, gate):
        _head(encoder="outra-torre").save(get_spec("coffee-foliar").head_path)
        attach_fake("coffee-foliar", confidence=0.6)

        body = _post(client, "coffee-foliar")
        catalogue = {entry["key"]: entry for entry in client.get("/models").json()}

        assert body["predicted_class"] == get_spec("coffee-foliar").class_names[-1]
        assert catalogue["coffee-foliar"]["safeguards"]["diagnosis_head"] is False

    def test_a_head_for_other_classes_is_ignored(self, client, models_dir, gate):
        _head(class_names=tuple(reversed(get_spec("coffee-foliar").class_names))).save(
            get_spec("coffee-foliar").head_path
        )
        attach_fake("coffee-foliar", confidence=0.6)

        assert (
            _post(client, "coffee-foliar")["predicted_class"]
            == get_spec("coffee-foliar").class_names[-1]
        )

    def test_a_leaf_of_another_plant_leaves_the_scope(self, client, models_dir, gate):
        SpeciesCheck(
            crop="citrus", mlp=_constant(2, [3.0, 0.0]), threshold=0.5, encoder="dublê"
        ).save(get_spec("citrus-foliar").species_path)
        attach_fake("citrus-foliar")

        framing = _post(client, "citrus-foliar")["framing"]

        assert framing["in_scope"] is False
        assert framing["scene"] == "folha_outra"
        assert framing["crop_matches"] is False
        assert framing["species_score"] == pytest.approx(1 / (1 + np.exp(3.0)), abs=1e-4)

    def test_a_citrus_leaf_keeps_its_verdict(self, client, models_dir, gate):
        SpeciesCheck(
            crop="citrus", mlp=_constant(2, [0.0, 3.0]), threshold=0.5, encoder="dublê"
        ).save(get_spec("citrus-foliar").species_path)
        attach_fake("citrus-foliar")

        framing = _post(client, "citrus-foliar")["framing"]

        assert framing["in_scope"] is True
        assert framing["scene"] == "folha_citros"
        assert framing["species_score"] > 0.9

    def test_an_aerial_photo_keeps_its_own_scene(self, client, models_dir, gate):
        """Fora do escopo pelo portão, a cena continua a do portão, não folha_outra."""
        SpeciesCheck(
            crop="citrus", mlp=_constant(2, [3.0, 0.0]), threshold=0.5, encoder="dublê"
        ).save(get_spec("citrus-foliar").species_path)
        attach_fake("citrus-foliar")

        framing = _post(client, "citrus-foliar", color=(230, 230, 230))["framing"]

        assert framing["in_scope"] is False
        assert framing["scene"] == "aereo"

    def test_the_rover_skips_it_unless_configured(self, client, models_dir, gate, monkeypatch):
        """No talhão, ela recusou 49% dos recortes do rover: lá, só se for pedida."""
        from app.registry import registry
        from app.rover_api import _predict_in_chunks

        SpeciesCheck(
            crop="citrus", mlp=_constant(2, [3.0, 0.0]), threshold=0.5, encoder="dublê"
        ).save(get_spec("citrus-foliar").species_path)
        attach_fake("citrus-foliar")
        model = registry.get("citrus-foliar")
        crops = [Image.new("RGB", (64, 64), (20, 60, 20))]

        [tile] = _predict_in_chunks(model, ["r.jpg"], crops, False)
        assert tile.framing.in_scope is True
        assert tile.framing.species_score is None

        monkeypatch.setattr(settings, "rover_species_check", True)
        [tile] = _predict_in_chunks(model, ["r.jpg"], crops, False)
        assert tile.framing.in_scope is False
        assert tile.framing.scene == "folha_outra"

    def test_a_check_of_another_crop_is_ignored(self, client, models_dir, gate):
        SpeciesCheck(
            crop="banana", mlp=_constant(2, [3.0, 0.0]), threshold=0.5, encoder="dublê"
        ).save(get_spec("citrus-foliar").species_path)
        attach_fake("citrus-foliar")

        framing = _post(client, "citrus-foliar")["framing"]

        assert framing["in_scope"] is True
        assert framing["species_score"] is None
