"""Teste externo — pasta de outra fonte, com só parte das classes do modelo."""

from __future__ import annotations

import numpy as np
import pytest

from training.evaluate import external_labels
from training.metrics import format_report, macro_f1, per_class_report

MODEL = ["Cordana", "Fusarium_Wilt", "Healthy", "Pestalotiopsis", "Sigatoka"]


class TestExternalLabels:
    def test_translates_folder_order_to_model_order(self):
        """Na pasta, Sigatoka é a classe 2; no modelo, 2 é Healthy."""
        folder = ["Cordana", "Fusarium_Wilt", "Sigatoka"]
        assert external_labels(folder, [0, 1, 2, 2], MODEL) == [0, 1, 4, 4]

    def test_refuses_a_class_the_model_does_not_know(self):
        with pytest.raises(ValueError, match="Moko"):
            external_labels(["Moko", "Sigatoka"], [0, 1], MODEL)


class TestAbsentClasses:
    """Uma classe sem nenhum exemplo não pode entrar na média nem no pior recall."""

    def matrix(self):
        # Só Fusarium_Wilt e Sigatoka no conjunto; uma Sigatoka prevista como Healthy.
        m = np.zeros((5, 5), dtype=np.int64)
        m[1, 1] = 10
        m[4, 4] = 9
        m[4, 2] = 1
        return m

    def test_macro_f1_ignores_classes_without_examples(self):
        report = per_class_report(self.matrix(), MODEL)
        expected = np.mean([row["f1"] for row in report if row["support"] > 0])
        assert macro_f1(report) == pytest.approx(expected)
        assert macro_f1(report) > 0.9

    def test_worst_recall_is_a_class_that_was_tested(self):
        text = format_report(self.matrix(), MODEL)
        assert "Menor recall: Sigatoka (0.900)" in text
