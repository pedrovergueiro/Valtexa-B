"""Testes da inferência pelo ONNX Runtime.

Constroem um grafo ONNX minúsculo na hora — média de cor por canal vira o
embedding, e uma matriz fixa vira as probabilidades — para exercitar o caminho
inteiro sem pesos de verdade.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

from app.config import settings  # noqa: E402
from app.onnx_runtime import OnnxModel, onnx_path  # noqa: E402
from app.registry import get_spec, registry  # noqa: E402
from app.severity import SeverityModel  # noqa: E402
from tests.conftest import DEFAULT_MODEL, make_image_bytes  # noqa: E402

SIZE = 8
WEIGHTS = np.array(
    [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]], dtype=np.float32
)
"""Canal vermelho puxa a classe 0, verde a 1, azul a 2; a 3 fica sem voto."""


def classifier_graph(path: Path, with_embedding: bool = True) -> Path:
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", SIZE, SIZE, 3])
    probs = helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", 4])
    embedding = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 3])
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["embedding"], axes=[1, 2], keepdims=0),
        helper.make_node("MatMul", ["embedding", "weights"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "classificador",
        [image],
        [probs, embedding] if with_embedding else [probs],
        initializer=[helper.make_tensor("weights", TensorProto.FLOAT, [3, 4], WEIGHTS.ravel())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.save(model, path)
    return path


def segmentation_graph(path: Path) -> Path:
    """Máscara = sigmoid(10 x média dos canais): pixel claro é "lesão"."""
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", SIZE, SIZE, 3])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["batch", SIZE, SIZE, 1])
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[3], keepdims=1),
        helper.make_node("Mul", ["mean", "gain"], ["scaled"]),
        helper.make_node("Sigmoid", ["scaled"], ["mask"]),
    ]
    graph = helper.make_graph(
        nodes,
        "segmentador",
        [image],
        [mask],
        initializer=[helper.make_tensor("gain", TensorProto.FLOAT, [], [10.0])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), path
    )
    return path


@pytest.fixture
def onnx_models_dir(client, tmp_path, monkeypatch) -> Path:
    """Um models/ só com o grafo e os rótulos — o que vai para a ground station."""
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "inference_runtime", "onnx")
    spec = get_spec(DEFAULT_MODEL)
    classifier_graph(onnx_path(spec.weights_path))
    spec.labels_path.write_text(json.dumps(list(spec.class_names)), encoding="utf-8")
    return tmp_path


class TestOnnxModel:
    def test_reads_shapes_like_a_keras_model(self, tmp_path):
        model = OnnxModel(classifier_graph(tmp_path / "m.onnx"))

        assert model.input_shape == (None, SIZE, SIZE, 3)
        assert model.output_shape == (None, 4)
        assert model.has_embedding is True

    def test_gives_probabilities_and_embedding_in_one_pass(self, tmp_path):
        model = OnnxModel(classifier_graph(tmp_path / "m.onnx"))
        batch = np.zeros((2, SIZE, SIZE, 3), np.float32)
        batch[0, ..., 1] = 1.0  # verde

        probs, embedding = model.dual_forward(batch)

        assert probs.shape == (2, 4) and embedding.shape == (2, 3)
        assert probs[0].argmax() == 1
        np.testing.assert_allclose(embedding[0], [0.0, 1.0, 0.0], atol=1e-6)

    def test_a_graph_without_embedding_still_classifies(self, tmp_path):
        model = OnnxModel(classifier_graph(tmp_path / "m.onnx", with_embedding=False))
        probs, embedding = model.dual_forward(np.zeros((1, SIZE, SIZE, 3), np.float32))

        assert probs.shape == (1, 4)
        assert embedding is None


class TestServingThroughOnnx:
    def test_serves_without_the_keras_weights(self, client, onnx_models_dir):
        """Só o .onnx e os rótulos: é o pacote da borda."""
        registry.get(DEFAULT_MODEL).load()

        response = client.post(
            "/predict",
            files={"file": ("x.jpg", make_image_bytes(color=(20, 220, 20)), "image/jpeg")},
        )

        assert response.status_code == 200
        assert response.json()["predicted_class"] == get_spec(DEFAULT_MODEL).class_names[1]

    def test_tta_runs_through_the_graph_too(self, client, onnx_models_dir):
        registry.get(DEFAULT_MODEL).load()

        body = client.post(
            "/predict",
            params={"tta": True},
            files={"file": ("x.jpg", make_image_bytes(color=(220, 20, 20)), "image/jpeg")},
        ).json()

        assert body["view_agreement"] == 1.0
        assert body["predicted_class"] == get_spec(DEFAULT_MODEL).class_names[0]

    def test_falls_back_to_keras_when_there_is_no_graph(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        monkeypatch.setattr(settings, "inference_runtime", "onnx")

        registry.get(DEFAULT_MODEL).load()  # sem .onnx e sem .keras: só avisa

        assert registry.get(DEFAULT_MODEL).is_loaded is False

    def test_without_keras_a_model_without_graph_is_off_not_a_crash(
        self, client, tmp_path, monkeypatch
    ):
        """O pacote de borda não traz o Keras. Com a pasta models/ copiada inteira, o
        .keras de um modelo sem .onnx derrubava a partida da API toda."""
        import sys

        monkeypatch.setattr(settings, "models_dir", tmp_path)
        monkeypatch.setattr(settings, "inference_runtime", "onnx")
        monkeypatch.setitem(sys.modules, "keras", None)  # import keras -> ImportError
        get_spec(DEFAULT_MODEL).weights_path.write_bytes(b"pesos")
        (tmp_path / "leaf_severity.keras").write_bytes(b"pesos")

        registry.get(DEFAULT_MODEL).load()
        severity = SeverityModel(tmp_path / "leaf_severity.keras")
        severity.load()

        assert registry.get(DEFAULT_MODEL).is_loaded is False
        assert severity.is_loaded is False

    def test_keras_runtime_ignores_the_graph(self, client, onnx_models_dir, monkeypatch):
        monkeypatch.setattr(settings, "inference_runtime", "keras")

        registry.get(DEFAULT_MODEL).load()  # há .onnx, mas não há .keras

        assert registry.get(DEFAULT_MODEL).is_loaded is False


class TestSeverityThroughOnnx:
    def test_measures_the_lesion_fraction(self, tmp_path, monkeypatch):
        from PIL import Image

        monkeypatch.setattr(settings, "inference_runtime", "onnx")
        segmentation_graph(tmp_path / "leaf_severity.onnx")
        model = SeverityModel(tmp_path / "leaf_severity.keras")
        model.load()

        half = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
        half.paste((255, 255, 255), (0, 0, SIZE, SIZE // 2))
        [severity] = model.estimate([half])

        assert model.is_loaded
        assert severity == pytest.approx(0.5)


class TestFramingThroughOnnx:
    def test_the_gate_runs_without_torch(self, tmp_path, monkeypatch):
        """Torre de visão em ONNX, alimentada pelo pré-processamento da própria API."""
        from PIL import Image

        from app.framing import SCENES, VISION_ONNX_NAME, FramingModel, FramingReference

        pixels = helper.make_tensor_value_info(
            "pixel_values", TensorProto.FLOAT, ["batch", 3, 224, 224]
        )
        embedding = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 3])
        node = helper.make_node(
            "ReduceMean", ["pixel_values"], ["embedding"], axes=[2, 3], keepdims=0
        )
        graph = helper.make_graph([node], "torre", [pixels], [embedding])
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8),
            tmp_path / VISION_ONNX_NAME,
        )

        # Folha de milho aponta para o escuro; vista aérea, para o claro.
        keys = tuple(scene.key for scene in SCENES)
        text = np.tile([1.0, -1.0, 0.0], (len(SCENES), 1)) / np.sqrt(2)
        text[keys.index("folha_milho")] = -np.ones(3) / np.sqrt(3)
        text[keys.index("aereo")] = np.ones(3) / np.sqrt(3)
        FramingReference(
            model_id="não-usado",
            keys=keys,
            prompts=tuple(scene.prompt for scene in SCENES),
            in_scope=np.array([scene.in_scope for scene in SCENES]),
            crops=tuple(scene.crop or "" for scene in SCENES),
            text_embeddings=text,
            logit_scale=100.0,
            threshold=0.5,
            calibrated_on=1,
        ).save(tmp_path / "framing.npz")

        monkeypatch.setattr(settings, "inference_runtime", "onnx")
        gate = FramingModel(tmp_path / "framing.npz")
        gate.load()

        dark, bright = gate.assess(
            [Image.new("RGB", (300, 200), (10, 10, 10)), Image.new("RGB", (300, 200), (250,) * 3)]
        )
        assert gate.is_loaded
        assert dark.in_scope is True and dark.scene == "folha_milho"
        assert bright.in_scope is False and bright.scene == "aereo"
