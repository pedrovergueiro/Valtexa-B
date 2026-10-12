"""Mede um modelo já treinado, classe por classe.

    python -m training.evaluate --model leaf-foliar --tta

Serve para responder "esse modelo está bom?" sem retreinar nada — e a resposta
útil nunca é um número só. Um modelo com 92% de acurácia que erra 40% das
manchas de cercosporiose é pior, para esta aplicação, do que um de 88% que
distribui o erro: a lavoura não liga para a média, liga para a doença que
passou batido.

O conjunto avaliado é o mesmo *split* de teste que o ``train`` separou, obtido
com a mesma ``--seed``. Reproduzi-lo aqui é o que permite comparar dois modelos
de forma justa: mesmas imagens, mesma ordem.

Esse teste, porém, sai das mesmas fontes do treino — mesma câmera, mesmo fundo,
mesma luz. Com ``--external`` o modelo é medido numa pasta de **outra fonte**
(por exemplo, fotos de campo do PlantSeg)::

    python -m training.evaluate --model banana-foliar --external campo/banana

A pasta segue ``<Classe>/*.jpg`` com nomes de classe do modelo, e pode trazer só
parte delas: um teste de campo raramente tem folha sadia fotografada.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relatório por classe de um modelo treinado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Pesos a avaliar. Padrão: os do catálogo para este modelo.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tta", action="store_true", help="Média de 4 orientações por imagem.")
    parser.add_argument(
        "--on",
        choices=("test", "val", "train"),
        default="test",
        help="Qual split avaliar. 'train' serve para medir o quanto o modelo decorou.",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=None,
        help=(
            "Pasta de teste de outra fonte (<pasta>/<Classe>/). Avalia todas as "
            "imagens dela em vez de um split de data/<modelo>/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    weights = args.weights or spec.weights_path
    data_dir = args.data_dir or spec.data_dir

    if not weights.exists():
        raise SystemExit(f"Pesos não encontrados: {weights}")
    if args.external is not None:
        data_dir = args.external
    if not data_dir.exists():
        raise SystemExit(f"Dados não encontrados: {data_dir}")

    import keras

    from training.dataset import load_datasets
    from training.metrics import evaluate_dataset, format_report

    model = keras.models.load_model(weights)
    img_size = _input_size(model) or spec.img_size

    if args.external is not None:
        class_names = _model_classes(weights, spec.class_names)
        dataset = external_dataset(args.external, class_names, img_size, args.batch_size)
        print(f"\n=== {spec.title} ({spec.key}) — teste externo ===")
        print(f"Pesos:  {weights}")
        print(f"Pasta:  {args.external}   TTA: {'sim' if args.tta else 'não'}")
        matrix = evaluate_dataset(model, dataset, len(class_names), tta=args.tta)
        titulo = "Teste externo" + (" com TTA" if args.tta else "")
        print(format_report(matrix, class_names, title=titulo))
        return

    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir,
        img_size,
        args.batch_size,
        args.validation_split,
        args.seed,
        test_split=args.test_split,
    )
    dataset = {"test": test_ds, "val": val_ds, "train": train_ds}[args.on]
    if dataset is None:
        raise SystemExit(f"O split {args.on!r} está vazio — verifique --test-split.")

    print(f"\n=== {spec.title} ({spec.key}) ===")
    print(f"Pesos:  {weights}")
    print(
        f"Entrada: {img_size}x{img_size}   split: {args.on}   TTA: {'sim' if args.tta else 'não'}"
    )

    matrix = evaluate_dataset(model, dataset, len(class_names), tta=args.tta)
    titulo = f"Split '{args.on}'" + (" com TTA" if args.tta else "")
    print(format_report(matrix, class_names, title=titulo))


def external_labels(
    folder_classes: Sequence[str], folder_labels: Sequence[int], model_classes: Sequence[str]
) -> list[int]:
    """Traduz o índice de classe da pasta externa para o índice do modelo.

    A pasta externa pode ter só parte das classes, e então a ordem alfabética
    dela não bate com a do modelo — sem esta tradução, "Sigatoka" (índice 2
    numa pasta com três classes) seria cobrada como "Healthy" (índice 2 no
    modelo) e o relatório sairia coerente e completamente errado.

    Raises:
        ValueError: se a pasta tiver uma classe que o modelo não conhece.
    """
    unknown = sorted(set(folder_classes) - set(model_classes))
    if unknown:
        raise ValueError(
            f"Classes da pasta que o modelo não conhece: {', '.join(unknown)}. "
            f"As do modelo são: {', '.join(model_classes)}."
        )
    return [list(model_classes).index(folder_classes[label]) for label in folder_labels]


def external_dataset(folder: Path, model_classes: Sequence[str], img_size: int, batch_size: int):
    """Todas as imagens de ``folder``, com o mesmo pré-processamento do treino."""
    import tensorflow as tf

    from app.inference.preprocess import scale_to_unit_range
    from training.dataset import _decode, index_by_class

    paths, labels, folder_classes = index_by_class(folder)
    if not paths:
        raise SystemExit(f"Nenhuma imagem em {folder}.")
    try:
        labels = external_labels(folder_classes, labels, model_classes)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda path, label: (_decode(path, img_size), label), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).map(lambda x, y: (scale_to_unit_range(x), y)).prefetch(2)


def _model_classes(weights: Path, fallback: Sequence[str]) -> list[str]:
    """As classes na ordem gravada junto dos pesos, ou as do catálogo."""
    labels = weights.with_suffix(".labels.json")
    if labels.exists():
        return list(json.loads(labels.read_text(encoding="utf-8")))
    return list(fallback)


def _input_size(model) -> int | None:
    """Lado da entrada declarado pelos pesos, para não depender do catálogo."""
    shape = getattr(model, "input_shape", None)
    if not isinstance(shape, tuple) or len(shape) != 4:
        return None
    height, width = shape[1], shape[2]
    return int(height) if height is not None and height == width else None


if __name__ == "__main__":
    main()
