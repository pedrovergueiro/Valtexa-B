"""Carregamento e augmentation do dataset de treino."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from app.inference.preprocess import scale_to_unit_range
from training import DEFAULT_SEED


def build_augmentation() -> tf.keras.Sequential:
    """Augmentations leves, coerentes com fotos aéreas de folhas.

    Só transformações que preservam o diagnóstico: espelhar, girar e aproximar
    um pouco não mudam a doença presente na folha. Alterações fortes de cor
    seriam contraproducentes, já que a cor é justamente o sinal que separa
    ferrugem, mancha e folha saudável.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def load_labeled_dataset(
    data_dir: Path, img_size: int, batch_size: int
) -> tuple[tf.data.Dataset, list[str]]:
    """Carrega o diretório inteiro, sem split e **sem augmentation**.

    É o que a calibração de novidade precisa: a referência tem de descrever as
    imagens como elas realmente são, não versões giradas e distorcidas delas.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    class_names = list(dataset.class_names)
    prepared = dataset.map(
        lambda images, labels: (scale_to_unit_range(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return prepared.prefetch(tf.data.AUTOTUNE), class_names


def load_datasets(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    validation_split: float,
    seed: int = DEFAULT_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Monta os datasets de treino e validação a partir de pastas por classe.

    O redimensionamento aqui é bilinear e não preserva o aspect ratio (padrão do
    ``image_dataset_from_directory``); a API reproduz exatamente isso em
    :func:`app.inference.preprocess.image_to_array`. A normalização vem da mesma
    função usada na inferência, de modo que treino e produção não podem divergir.

    Returns:
        Os datasets de treino e validação já normalizados, e os nomes das
        classes na ordem em que o Keras as indexou (ordem alfabética das pastas).
    """
    common = {
        "validation_split": validation_split,
        "seed": seed,
        "image_size": (img_size, img_size),
        "batch_size": batch_size,
        "label_mode": "int",  # pareia com sparse_categorical_crossentropy
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, subset="validation", **common)

    class_names = list(train_ds.class_names)
    augmentation = build_augmentation()

    def prepare_train(images, labels):
        images = augmentation(images, training=True)
        return scale_to_unit_range(images), labels

    def prepare_val(images, labels):
        return scale_to_unit_range(images), labels

    train_ds = train_ds.map(prepare_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prepare_val, num_parallel_calls=tf.data.AUTOTUNE)

    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        class_names,
    )
