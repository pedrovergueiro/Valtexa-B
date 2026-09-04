from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SEED = 42


def load_datasets(data_dir: Path, img_size: int, batch_size: int, validation_split: float):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    def prep_train(images, labels):
        images = augmentation(images, training=True)
        images = preprocess_input(images)
        return images, labels

    def prep_val(images, labels):
        images = preprocess_input(images)
        return images, labels

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
