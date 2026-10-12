"""Treina a segmentação de lesão que dá a severidade.

    python -m training.train_severity --plantseg ~/valtexa-raw/plantseg/plantseg

Severidade aqui é **área lesionada no quadro**: pixels marcados como lesão sobre
os pixels da imagem. Aproxima a fração da folha quando a folha preenche o
quadro — a mesma premissa do ``leaf-foliar``, ao lado do qual ela roda. Numa foto
com solo ou céu, o número sai menor do que a severidade real da folha.

Dados:

* **PlantSeg, milho** — as únicas máscaras de lesão públicas e *in-the-wild* que
  temos: ferrugem, cercosporiose e helmintosporiose. ``corn_smut`` fica de fora:
  carvão ataca a espiga, e aprender a segmentá-lo ensinaria o modelo a marcar
  espiga como lesão foliar.
* **Folhas saudáveis do leaf-foliar, com máscara vazia.** Sem elas o modelo só
  vê imagens em que há lesão e aprende a sempre achar alguma. Uma folha sadia
  tem, por definição, área lesionada zero.

O teste é o ``test`` do próprio PlantSeg mais uma fração das saudáveis, nenhum
dos dois usado para escolher época.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.inference.preprocess import scale_to_unit_range
from app.severity import LESION_THRESHOLD, WEIGHTS_NAME, lesion_fraction
from training import DEFAULT_SEED
from training.prepare_dataset import is_image

LESION_PREFIXES = ("corn_rust", "corn_gray_leaf_spot", "corn_northern_leaf_blight")
"""Prefixos de arquivo do PlantSeg que viram exemplo de lesão foliar de milho."""

SKIP_LAYERS = (
    "block_1_expand_relu",  # 1/2
    "block_3_expand_relu",  # 1/4
    "block_6_expand_relu",  # 1/8
    "block_13_expand_relu",  # 1/16
)
BOTTLENECK = "block_16_project"  # 1/32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmentação de lesão foliar de milho -> severidade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plantseg", type=Path, required=True, help="Raiz do PlantSeg extraído.")
    parser.add_argument(
        "--healthy",
        type=Path,
        default=settings.data_dir / "leaf-foliar" / "Healthy",
        help="Folhas saudáveis, usadas com máscara vazia.",
    )
    parser.add_argument("--healthy-count", type=int, default=400)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20, help="Épocas com o encoder congelado.")
    parser.add_argument("--fine-tune-epochs", type=int, default=25)
    parser.add_argument("--fine-tune-layers", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=settings.models_dir / WEIGHTS_NAME)
    return parser.parse_args()


# --- dados -------------------------------------------------------------------


def _load_pair(
    image_path: Path, mask_path: Path | None, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Imagem e máscara redimensionadas juntas, em ``uint8``.

    Sem ``exif_transpose`` de propósito: a máscara do PlantSeg foi desenhada
    sobre os pixels como estão no arquivo, e girar só a imagem desalinharia as
    duas. A imagem vai bilinear, como no :mod:`app.inference.preprocess`; a
    máscara vai por vizinho mais próximo, para continuar binária.
    """
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if mask_path is None:
        mask = np.zeros((size, size), dtype=np.uint8)
    else:
        with Image.open(mask_path) as source:
            if source.size != image.size:
                raise ValueError(f"{image_path.name}: imagem {image.size} e máscara {source.size}")
            mask = np.asarray(source.convert("L").resize((size, size), Image.Resampling.NEAREST))
            mask = (mask > 0).astype(np.uint8)
    pixels = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    return pixels, mask


def plantseg_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    """Pares (imagem, máscara) de lesão foliar de milho num split do PlantSeg."""
    pairs = []
    for mask in sorted((root / "annotations" / split).glob("*.png")):
        if not mask.name.startswith(LESION_PREFIXES):
            continue
        image = next(
            (p for p in (root / "images" / split).glob(mask.stem + ".*") if is_image(p)), None
        )
        if image is not None:
            pairs.append((image, mask))
    return pairs


def load_arrays(
    pairs: list[tuple[Path, Path | None]], size: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images, masks, names, skipped = [], [], [], 0
    for image_path, mask_path in pairs:
        try:
            pixels, mask = _load_pair(image_path, mask_path, size)
        except (OSError, ValueError) as exc:
            skipped += 1
            print(f"  ignorada: {exc}")
            continue
        images.append(pixels)
        masks.append(mask)
        names.append(image_path.name)
    if skipped:
        print(f"  {skipped} par(es) ignorado(s)")
    return np.stack(images), np.stack(masks)[..., None], names


def _augment(images: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
    """Espelhamentos e giros de 90°: a lesão não tem lado certo."""
    out_images, out_masks = images.copy(), masks.copy()
    for index in range(len(images)):
        turns = int(rng.integers(4))
        image, mask = np.rot90(images[index], turns), np.rot90(masks[index], turns)
        if rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        out_images[index], out_masks[index] = image, mask
    return out_images, out_masks


# --- modelo ------------------------------------------------------------------


def _keras():
    from app.keras_backend import configure

    configure()
    import keras

    return keras


def build_unet(img_size: int):
    """U-Net com a MobileNetV2 da ImageNet como encoder.

    Mesma base e mesma escala de entrada (``[-1, 1]``) do classificador: o
    pré-processamento da API serve às duas redes sem ramificação.
    """
    keras = _keras()
    layers = keras.layers
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    encoder = keras.Model(
        base.input,
        [base.get_layer(name).output for name in (*SKIP_LAYERS, BOTTLENECK)],
        name="encoder",
    )
    encoder.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    *skips, x = encoder(inputs, training=False)
    for filters, skip in zip((256, 128, 64, 32), reversed(skips), strict=True):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="lesion")(x)
    return keras.Model(inputs, outputs, name="leaf_severity_unet")


def _loss():
    """BCE + Dice.

    Só BCE, com lesão ocupando uma fração pequena de muitos quadros, empurra o
    modelo para "nada é lesão". O Dice por imagem pune isso — e numa folha sadia,
    com máscara vazia, ele só é mínimo se a previsão também for vazia.
    """
    keras = _keras()
    ops = keras.ops
    bce = keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        intersection = ops.sum(y_true * y_pred, axis=(1, 2, 3))
        total = ops.sum(y_true, axis=(1, 2, 3)) + ops.sum(y_pred, axis=(1, 2, 3))
        dice = (2.0 * intersection + 1.0) / (total + 1.0)
        return bce(y_true, y_pred) + (1.0 - ops.mean(dice))

    return loss


def _batches(images, masks, batch_size: int, seed: int, augment: bool):
    keras = _keras()

    class Batches(keras.utils.PyDataset):
        def __init__(self) -> None:
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(len(images))

        def __len__(self) -> int:
            return int(np.ceil(len(images) / batch_size))

        def __getitem__(self, index: int):
            chunk = self.order[index * batch_size : (index + 1) * batch_size]
            batch_images, batch_masks = images[chunk], masks[chunk]
            if augment:
                batch_images, batch_masks = _augment(batch_images, batch_masks, self.rng)
            return (
                scale_to_unit_range(batch_images.astype(np.float32)),
                batch_masks.astype(np.float32),
            )

        def on_epoch_end(self) -> None:
            if augment:
                self.rng.shuffle(self.order)

    return Batches()


def predict_masks(model, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    from app.keras_backend import no_grad, to_numpy

    outputs = []
    for start in range(0, len(images), batch_size):
        batch = scale_to_unit_range(images[start : start + batch_size].astype(np.float32))
        with no_grad():  # sem isto o torch grava o grafo de gradientes a cada lote
            outputs.append(to_numpy(model(batch, training=False)))
    return np.concatenate(outputs)


# --- avaliação ---------------------------------------------------------------


def evaluate(probabilities: np.ndarray, masks: np.ndarray, names: list[str]) -> dict:
    """Mede o que importa para severidade, não só a sobreposição de pixels."""
    predicted = probabilities[..., 0] > LESION_THRESHOLD
    truth = masks[..., 0] > 0
    lesion = truth.reshape(len(truth), -1).any(axis=1)

    intersection = (predicted & truth).sum(axis=(1, 2))
    union = (predicted | truth).sum(axis=(1, 2))
    sizes = predicted.sum(axis=(1, 2)) + truth.sum(axis=(1, 2))
    true_fraction = truth.mean(axis=(1, 2))
    estimated = np.array(lesion_fraction(probabilities))
    error = np.abs(estimated - true_fraction)

    def summary(selection: np.ndarray) -> dict:
        if not selection.any():
            return {}
        return {
            "n": int(selection.sum()),
            "dice_medio": round(
                float((2 * intersection / np.maximum(sizes, 1))[selection].mean()), 4
            ),
            "iou_medio": round(float((intersection / np.maximum(union, 1))[selection].mean()), 4),
            "fracao_real_media": round(float(true_fraction[selection].mean()), 4),
            "fracao_estimada_media": round(float(estimated[selection].mean()), 4),
            "erro_absoluto_medio": round(float(error[selection].mean()), 4),
            "erro_absoluto_p90": round(float(np.percentile(error[selection], 90)), 4),
            "correlacao": (
                round(float(np.corrcoef(estimated[selection], true_fraction[selection])[0, 1]), 4)
                if selection.sum() > 2 and true_fraction[selection].std() > 0
                else None
            ),
        }

    result = {"com_lesao": summary(lesion)}
    for prefix in LESION_PREFIXES:
        chosen = np.array([name.startswith(prefix) for name in names]) & lesion
        result[prefix] = summary(chosen)
    healthy = ~lesion
    if healthy.any():
        result["saudaveis"] = {
            "n": int(healthy.sum()),
            "lesao_falsa_media": round(float(estimated[healthy].mean()), 4),
            "lesao_falsa_p95": round(float(np.percentile(estimated[healthy], 95)), 4),
            "com_mais_de_5pct": round(float((estimated[healthy] > 0.05).mean()), 4),
        }
    return result


def _print(result: dict) -> None:
    print("\n=== Severidade no teste (nunca visto) ===")
    for group, values in result.items():
        if values:
            print(f"{group:28} {json.dumps(values, ensure_ascii=False)}")
    print(
        "\nLembre-se: é área lesionada no quadro, medida contra máscaras de um "
        "dataset público. Não é a escala diagramática de um fitopatologista."
    )


# --- principal ---------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if not args.plantseg.exists():
        raise SystemExit(f"PlantSeg não encontrado: {args.plantseg}")
    rng = np.random.default_rng(args.seed)

    train_pairs = plantseg_pairs(args.plantseg, "train")
    test_pairs = plantseg_pairs(args.plantseg, "test")
    healthy = (
        sorted(p for p in args.healthy.iterdir() if is_image(p)) if args.healthy.exists() else []
    )
    healthy = [healthy[i] for i in rng.permutation(len(healthy))[: args.healthy_count]]
    cut = int(len(healthy) * 0.8)
    train_healthy, test_healthy = healthy[:cut], healthy[cut:]
    print(
        f"PlantSeg: {len(train_pairs)} treino, {len(test_pairs)} teste · "
        f"saudáveis: {len(train_healthy)} treino, {len(test_healthy)} teste"
    )
    if not train_pairs:
        raise SystemExit("Nenhuma máscara de lesão foliar de milho encontrada.")

    images, masks, _ = load_arrays(
        [*train_pairs, *((path, None) for path in train_healthy)], args.img_size
    )
    order = rng.permutation(len(images))
    n_val = int(len(images) * args.validation_split)
    val_idx, train_idx = order[:n_val], order[n_val:]

    keras = _keras()
    model = build_unet(args.img_size)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
    ]
    training = _batches(images[train_idx], masks[train_idx], args.batch_size, args.seed, True)
    validation = _batches(images[val_idx], masks[val_idx], args.batch_size, args.seed, False)

    print("\n== Fase 1: decoder, com o encoder congelado ==")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=_loss())
    model.fit(training, validation_data=validation, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune_epochs:
        print(f"\n== Fase 2: fine-tuning das {args.fine_tune_layers} camadas finais do encoder ==")
        encoder = model.get_layer("encoder")
        encoder.trainable = True
        for layer in encoder.layers[: -args.fine_tune_layers]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=_loss())
        model.fit(
            training, validation_data=validation, epochs=args.fine_tune_epochs, callbacks=callbacks
        )

    test_images, test_masks, test_names = load_arrays(
        [*test_pairs, *((path, None) for path in test_healthy)], args.img_size
    )
    result = evaluate(predict_masks(model, test_images), test_masks, test_names)
    _print(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Recompila com uma perda padrão só para gravar: a perda de treino é uma
    # função local, e o arquivo não deve depender dela para ser aberto. Os
    # pesos não mudam; a API carrega com ``compile=False``.
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.save(args.output)
    metrics = args.output.with_suffix(".metrics.json")
    metrics.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPesos:    {args.output}\nMétricas: {metrics}")


if __name__ == "__main__":
    main()
