"""Testes da lógica de treino que não depende do TensorFlow.

O treino em si roda na GPU, fora da suíte. O que é coberto aqui é justamente o
que costuma falhar tarde demais: resolução de caminhos por modelo, validação dos
argumentos e a gravação dos rótulos que a API vai consumir depois.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from app.registry import get_spec
from training.train import MONITOR, _best_metric, _validate_args, _write_labels, resolve_paths

LEAF = get_spec("leaf-foliar")


def _args(tmp_path, **overrides) -> argparse.Namespace:
    defaults = {"data_dir": tmp_path, "validation_split": 0.2, "epochs": 15}
    return argparse.Namespace(**{**defaults, **overrides})


class TestPathResolution:
    def test_fills_paths_from_the_selected_model(self):
        args = argparse.Namespace(model="rover-understory", data_dir=None, output=None)
        spec = resolve_paths(args)

        assert spec.key == "rover-understory"
        assert args.data_dir.name == "rover-understory"
        assert args.output.name == "rover_understory.keras"

    def test_keeps_paths_given_explicitly(self, tmp_path):
        args = argparse.Namespace(
            model="leaf-foliar", data_dir=tmp_path, output=tmp_path / "meu.keras"
        )
        resolve_paths(args)

        assert args.data_dir == tmp_path
        assert args.output == tmp_path / "meu.keras"

    def test_each_model_writes_to_its_own_file(self):
        outputs = set()
        for key in ("leaf-foliar", "rover-understory", "aerial-canopy"):
            args = argparse.Namespace(model=key, data_dir=None, output=None)
            resolve_paths(args)
            outputs.add(args.output)
        assert len(outputs) == 3


class TestArgValidation:
    def test_accepts_sane_arguments(self, tmp_path):
        _validate_args(_args(tmp_path))

    def test_rejects_missing_data_dir(self, tmp_path):
        with pytest.raises(SystemExit, match="não encontrado"):
            _validate_args(_args(tmp_path / "ausente"))

    @pytest.mark.parametrize("split", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_validation_split_outside_range(self, tmp_path, split):
        with pytest.raises(SystemExit, match="validation-split"):
            _validate_args(_args(tmp_path, validation_split=split))

    def test_rejects_zero_epochs(self, tmp_path):
        with pytest.raises(SystemExit, match="epochs"):
            _validate_args(_args(tmp_path, epochs=0))


class TestLabelsFile:
    def test_writes_labels_next_to_the_weights(self, tmp_path):
        output = tmp_path / "models" / "meu_modelo.keras"
        labels_path = _write_labels(LEAF, output, list(LEAF.class_names))

        assert labels_path == output.parent / "meu_modelo.labels.json"
        assert json.loads(labels_path.read_text(encoding="utf-8")) == list(LEAF.class_names)

    def test_creates_the_output_directory(self, tmp_path):
        output = tmp_path / "novo" / "nivel" / "model.keras"
        _write_labels(LEAF, output, ["A", "B"])
        assert output.parent.is_dir()

    def test_preserves_the_class_order_from_the_dataset(self, tmp_path):
        """A ordem é o contrato com a API — nunca deve ser reordenada."""
        desordenado = ["Healthy", "Blight", "Gray_Leaf_Spot", "Common_Rust"]
        labels_path = _write_labels(LEAF, tmp_path / "model.keras", desordenado)
        assert json.loads(labels_path.read_text(encoding="utf-8")) == desordenado

    def test_uses_the_catalogue_path_for_the_default_output(self):
        labels_path = _write_labels(LEAF, LEAF.weights_path, list(LEAF.class_names))
        assert labels_path == LEAF.labels_path
        labels_path.unlink()


class TestBestMetric:
    def test_returns_the_best_epoch(self):
        history = SimpleNamespace(history={MONITOR: [0.70, 0.91, 0.85]})
        assert _best_metric(history) == pytest.approx(0.91)

    def test_fails_when_the_metric_is_absent(self):
        history = SimpleNamespace(history={"accuracy": [0.9]})
        with pytest.raises(SystemExit, match=MONITOR):
            _best_metric(history)
