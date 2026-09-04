"""
Treina o classificador de doenças de folha de milho.

Uso:
    python -m training.train --data-dir data --epochs 15

Espera um diretório com uma subpasta por classe, ex:
    data/
      Blight/*.jpg
      Common_Rust/*.jpg
      Gray_Leaf_Spot/*.jpg
      Healthy/*.jpg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import CLASS_NAMES, IMG_SIZE, MODEL_PATH
from training.dataset import load_datasets
from training.model_def import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino do modelo de doenças de milho")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise SystemExit(
            f"Diretório de dados não encontrado: {args.data_dir}\n"
            "Baixe o dataset (ex.: 'Corn or Maize Leaf Disease Dataset' no Kaggle) "
            "e organize em subpastas por classe dentro dele."
        )

    train_ds, val_ds, class_names = load_datasets(
        args.data_dir, IMG_SIZE, args.batch_size, args.validation_split
    )

    if class_names != CLASS_NAMES:
        print(
            f"Aviso: classes encontradas no diretório ({class_names}) "
            f"diferem da ordem esperada em app/config.py ({CLASS_NAMES}). "
            "Ajuste CLASS_NAMES para bater exatamente com esta ordem."
        )

    import tensorflow as tf

    model = build_model(IMG_SIZE, num_classes=len(class_names), fine_tune_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(args.output), save_best_only=True),
    ]

    print("== Fase 1: treinando cabeça de classificação (base congelada) ==")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune_epochs > 0:
        print("== Fase 2: fine-tuning da base MobileNetV2 ==")
        model.layers[1].trainable = True  # base_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )

    model.save(args.output)

    labels_path = args.output.parent / "labels.json"
    labels_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Modelo salvo em: {args.output}")
    print(f"Labels salvos em: {labels_path}")


if __name__ == "__main__":
    main()
