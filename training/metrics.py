"""Métricas por classe — o que a acurácia global esconde.

Uma acurácia de 92% num dataset onde 30% das folhas são saudáveis pode ser um
modelo excelente ou um modelo que aprendeu a chutar "Healthy". Os dois números
são iguais; as consequências em lavoura não são.

Para este produto o que importa é o **recall por classe**: a fração das folhas
realmente doentes que o modelo pegou. Um falso negativo é uma doença que se
espalha sem ninguém olhar — caro e irreversível. Um falso positivo é uma ida a
campo à toa, que custa uma tarde. As duas coisas não se equivalem, e uma métrica
única as trata como se equivalessem.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Matriz ``[real, previsto]`` com a contagem de cada combinação."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def per_class_report(matrix: np.ndarray, class_names: Sequence[str]) -> list[dict[str, float]]:
    """Precisão, recall, F1 e suporte de cada classe.

    Divisões por zero viram zero em vez de ``nan``: uma classe que o modelo nunca
    previu tem precisão indefinida na matemática, mas na prática o que se quer
    ver no relatório é que ela não está funcionando — e ``nan`` polui a média.
    """
    report = []
    for index, name in enumerate(class_names):
        true_positive = float(matrix[index, index])
        predicted = float(matrix[:, index].sum())
        actual = float(matrix[index, :].sum())

        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        report.append(
            {
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(actual),
            }
        )
    return report


def macro_f1(report: Sequence[dict[str, float]]) -> float:
    """F1 médio sem ponderar pelo tamanho da classe.

    É a métrica que denuncia o modelo que vai bem só nas classes abundantes: uma
    classe rara pesa aqui tanto quanto a maior de todas.

    Classe sem nenhum exemplo no conjunto fica de fora: num teste externo que só
    traz folhas doentes, "Healthy" teria F1 zero por construção, e a média
    passaria a medir a composição do conjunto em vez do modelo.
    """
    present = [row["f1"] for row in report if row["support"] > 0]
    return float(np.mean(present)) if present else 0.0


def format_report(
    matrix: np.ndarray, class_names: Sequence[str], *, title: str = "Desempenho por classe"
) -> str:
    """Monta o relatório completo em texto, pronto para imprimir."""
    report = per_class_report(matrix, class_names)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    width = max(len(name) for name in class_names) if class_names else 10

    lines = [
        f"\n=== {title} ===",
        f"{'classe':<{width}}  {'precisão':>9}  {'recall':>7}  {'F1':>6}  {'n':>6}",
        "-" * (width + 34),
    ]
    for row in report:
        lines.append(
            f"{row['class_name']:<{width}}  {row['precision']:>9.3f}  "
            f"{row['recall']:>7.3f}  {row['f1']:>6.3f}  {row['support']:>6}"
        )

    lines.append("-" * (width + 34))
    accuracy = correct / total if total else 0.0
    lines.append(f"acurácia: {accuracy:.4f}   macro-F1: {macro_f1(report):.4f}")

    lines.append("\n=== Matriz de confusão (linha = real, coluna = previsto) ===")
    header = " " * (width + 2) + "".join(f"{name[:7]:>8}" for name in class_names)
    lines.append(header)
    for index, name in enumerate(class_names):
        row = "".join(f"{int(value):>8}" for value in matrix[index])
        lines.append(f"{name:<{width}}  {row}")

    present = [row for row in report if row["support"] > 0]
    worst = min(present, key=lambda row: row["recall"]) if present else None
    if worst is not None:
        lines.append(
            f"\nMenor recall: {worst['class_name']} ({worst['recall']:.3f}) — de cada 100 "
            f"folhas dessa classe, {round((1 - worst['recall']) * 100)} passam batido."
        )
    return "\n".join(lines)


def evaluate_dataset(model, dataset, num_classes: int, *, tta: bool = False) -> np.ndarray:
    """Roda o dataset pelo modelo e devolve a matriz de confusão.

    Converte cada lote para ``numpy`` porque o dataset é do ``tf.data`` e o
    modelo pode estar em qualquer backend do Keras — sem isso, o backend torch
    recebe tensor do TensorFlow e falha.

    Args:
        tta: se ``True``, prevê cada imagem em quatro orientações e soma as
            probabilidades. Uma folha espelhada continua tendo a mesma doença,
            então as quatro previsões deveriam concordar; onde discordam, o
            modelo estava inseguro, e a média acerta mais do que qualquer uma
            das quatro isolada.
    """
    from app.keras_backend import no_grad, to_numpy

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in dataset:
        batch = np.asarray(images)
        # Sem isto o backend torch grava o grafo de gradientes a cada lote — ver
        # app.keras_backend.no_grad.
        with no_grad():
            probabilities = to_numpy(model(batch, training=False))

            if tta:
                for transformada in (batch[:, :, ::-1], batch[:, ::-1], batch[:, ::-1, ::-1]):
                    probabilities = probabilities + to_numpy(
                        model(np.ascontiguousarray(transformada), training=False)
                    )

        predictions = probabilities.argmax(axis=1)
        matrix += confusion_matrix(np.asarray(labels), predictions, num_classes)
    return matrix
