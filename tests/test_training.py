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

from app.config import settings
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

    def test_uses_the_catalogue_path_for_the_default_output(self, tmp_path, monkeypatch):
        """Quando a saída é a do catálogo, os rótulos vão para o par dela.

        O ``models_dir`` é redirecionado para um diretório temporário porque
        ``weights_path`` e ``labels_path`` são resolvidos contra ele em tempo de
        acesso. Sem isso este teste escreveria — e apagaria — o
        ``.labels.json`` de um modelo realmente treinado, destruindo o contrato
        de ordem de classes que a API lê em produção.
        """
        monkeypatch.setattr(settings, "models_dir", tmp_path)

        labels_path = _write_labels(LEAF, LEAF.weights_path, list(LEAF.class_names))

        assert labels_path == LEAF.labels_path
        assert labels_path.parent == tmp_path
        assert json.loads(labels_path.read_text(encoding="utf-8")) == list(LEAF.class_names)


class TestSplitIndices:
    """O corte treino/validação/teste é o contrato que sustenta toda métrica.

    Se ele vazar, nenhum número do projeto significa o que diz — e o vazamento
    não gera erro, só uma acurácia alta e um piloto decepcionante.
    """

    SPLIT = pytest.importorskip("training.dataset", reason="exige TensorFlow instalado")

    def test_sizes_follow_the_requested_fractions(self):
        train, val, test = self.SPLIT.split_indices(1000, 42, 0.2, 0.15)
        assert (len(train), len(val), len(test)) == (650, 200, 150)

    def test_the_three_sets_never_overlap(self):
        train, val, test = self.SPLIT.split_indices(1000, 42, 0.2, 0.15)
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()

    def test_every_sample_lands_in_exactly_one_set(self):
        train, val, test = self.SPLIT.split_indices(1000, 42, 0.2, 0.15)
        assert sorted([*train, *val, *test]) == list(range(1000))

    def test_the_same_seed_gives_the_same_split(self):
        """É o que permite comparar dois modelos e o que o evaluate reproduz."""
        first = self.SPLIT.split_indices(500, 7, 0.2, 0.1)
        second = self.SPLIT.split_indices(500, 7, 0.2, 0.1)
        for a, b in zip(first, second, strict=True):
            assert list(a) == list(b)

    def test_a_different_seed_gives_a_different_split(self):
        _, _, test_a = self.SPLIT.split_indices(500, 7, 0.2, 0.1)
        _, _, test_b = self.SPLIT.split_indices(500, 8, 0.2, 0.1)
        assert list(test_a) != list(test_b)

    def test_zero_test_split_leaves_the_test_set_empty(self):
        train, val, test = self.SPLIT.split_indices(1000, 42, 0.2, 0.0)
        assert len(test) == 0
        assert len(train) + len(val) == 1000

    @pytest.mark.parametrize(
        ("validation", "test"),
        [
            (-0.1, 0.2),  # soma 0,1 e passaria pelo teste de intervalo
            (0.2, -0.1),
            (0.9, 0.2),  # não sobra treino
            (0.0, 0.0),  # não sobra validação
        ],
    )
    def test_rejects_fractions_that_would_corrupt_the_split(self, validation, test):
        """Fração negativa fatia ao contrário e sobrepõe treino e teste."""
        with pytest.raises(ValueError):
            self.SPLIT.split_indices(1000, 42, validation, test)


class TestBestMetric:
    def test_returns_the_best_epoch(self):
        history = SimpleNamespace(history={MONITOR: [0.70, 0.91, 0.85]})
        assert _best_metric(history) == pytest.approx(0.91)

    def test_fails_when_the_metric_is_absent(self):
        history = SimpleNamespace(history={"accuracy": [0.9]})
        with pytest.raises(SystemExit, match=MONITOR):
            _best_metric(history)
