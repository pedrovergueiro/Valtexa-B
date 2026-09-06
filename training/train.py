"""Treina um dos modelos do catálogo (ver ``MODELS.md``).

Execute a partir da raiz do repositório, como módulo::

    python -m training.train --model leaf-foliar --epochs 15 --fine-tune-epochs 5

O diretório de dados precisa ter uma subpasta por classe. Por padrão é
``data/<modelo>/``, e o ``prepare_dataset`` monta essa estrutura a partir dos
datasets públicos::

    data/leaf-foliar/
      Blight/*.jpg
      Common_Rust/*.jpg
      Gray_Leaf_Spot/*.jpg
      Healthy/*.jpg

Ao final ficam em ``models/`` os pesos da melhor época e o ``.labels.json`` com a
ordem das classes — que é o que a API lê para traduzir índices em nomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from app.registry import MODEL_SPECS, ModelSpec, get_spec
from training import DEFAULT_SEED

MONITOR = "val_accuracy"
"""Métrica que decide qual época é a melhor, tanto no checkpoint quanto no
early stopping."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino dos modelos de diagnóstico em milho",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        default=settings.default_model,
        help="Modelo do catálogo a treinar.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Padrão: data/<modelo>/.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Épocas da fase 1 (cabeça).")
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=5, help="Épocas da fase 2. Use 0 para pular."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4, help="Paciência do early stopping.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="LR da fase 1.")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5, help="LR da fase 2.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=None, help="Padrão: o caminho de pesos do modelo."
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ModelSpec:
    """Preenche os caminhos que dependem do modelo escolhido."""
    spec = get_spec(args.model)
    if args.data_dir is None:
        args.data_dir = spec.data_dir
    if args.output is None:
        args.output = spec.weights_path
    return spec


def main() -> None:
    args = parse_args()
    spec = resolve_paths(args)
    _validate_args(args)

    print(f"\n=== {spec.title} ({spec.key}) ===")
    print(f"{spec.summary}\n")
    if spec.needs_field_data:
        print(
            "Atenção: este modelo depende de coleta própria em campo para as "
            "classes que não têm dataset público. Ver MODELS.md.\n"
        )

    # Imports tardios: carregar o TensorFlow custa alguns segundos, e assim
    # ``--help`` e a validação dos argumentos respondem na hora.
    from training.dataset import load_datasets
    from training.model_def import build_model, unfreeze_base

    train_ds, val_ds, class_names = load_datasets(
        args.data_dir, spec.img_size, args.batch_size, args.validation_split, args.seed
    )
    _report_class_order(spec, class_names)

    # Gravado antes do treino: se a execução for interrompida, o checkpoint já
    # salvo continua acompanhado da ordem de classes correta.
    labels_path = _write_labels(spec, args.output, class_names)

    import tensorflow as tf

    model = build_model(spec.img_size, num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n== Fase 1: treinando a cabeça de classificação (base congelada) ==")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=_build_callbacks(args, args.output, best_so_far=None),
    )
    best_accuracy = _best_metric(history)

    if args.fine_tune_epochs > 0:
        print("\n== Fase 2: fine-tuning da base MobileNetV2 ==")
        unfreeze_base(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # ``best_so_far`` impede que a fase 2 sobrescreva um checkpoint melhor
        # da fase 1 — sem isso, o limiar do checkpoint reinicia e a primeira
        # época do fine-tuning grava por cima mesmo sendo pior.
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=_build_callbacks(args, args.output, best_so_far=best_accuracy),
        )
        best_accuracy = max(best_accuracy, _best_metric(history))

    print(f"\nMelhor {MONITOR}: {best_accuracy:.4f}")
    print(f"Pesos salvos em:  {args.output}")
    print(f"Labels salvos em: {labels_path}")
    print(
        "\nLembre-se: val_accuracy medida sobre dataset público não prevê o "
        "desempenho em campo. Valide num conjunto de teste capturado pela sua "
        "própria plataforma antes de prometer número para cliente."
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise SystemExit(
            f"Diretório de dados não encontrado: {args.data_dir}\n"
            "Monte o dataset com 'python -m training.prepare_dataset --help' "
            "ou organize as imagens em subpastas por classe."
        )
    if not 0.0 < args.validation_split < 1.0:
        raise SystemExit("--validation-split precisa estar entre 0 e 1 (exclusivo).")
    if args.epochs < 1:
        raise SystemExit("--epochs precisa ser pelo menos 1.")


def _build_callbacks(args: argparse.Namespace, output: Path, best_so_far: float | None) -> list:
    """Callbacks de uma fase de treino.

    São recriados por fase de propósito: o early stopping precisa reiniciar sua
    paciência, enquanto o checkpoint precisa herdar o melhor resultado já obtido
    via ``best_so_far``.
    """
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR,
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output),
            monitor=MONITOR,
            mode="max",
            save_best_only=True,
            initial_value_threshold=best_so_far,
            verbose=1,
        ),
    ]


def _best_metric(history) -> float:
    values = history.history.get(MONITOR)
    if not values:
        raise SystemExit(f"Métrica {MONITOR!r} ausente no histórico — verifique o dataset.")
    return float(max(values))


def _report_class_order(spec: ModelSpec, class_names: list[str]) -> None:
    print(f"\nClasses encontradas ({len(class_names)}): {', '.join(class_names)}")
    if tuple(class_names) != tuple(spec.class_names):
        print(
            f"Aviso: a ordem difere do catálogo para {spec.key} "
            f"({', '.join(spec.class_names)}).\n"
            "Não é um problema: a API lê o .labels.json gerado aqui, que tem "
            "precedência. Ajuste o catálogo apenas se quiser manter os dois iguais."
        )


def _write_labels(spec: ModelSpec, output: Path, class_names: list[str]) -> Path:
    """Grava a ordem das classes ao lado dos pesos."""
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_path = (
        spec.labels_path if output == spec.weights_path else output.with_suffix(".labels.json")
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    return labels_path


if __name__ == "__main__":
    main()
