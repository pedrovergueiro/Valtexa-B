from tensorflow import keras
from tensorflow.keras import layers


def build_model(img_size: int, num_classes: int, fine_tune_base: bool = False) -> keras.Model:
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune_base

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=fine_tune_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="corn_disease_mobilenetv2")
    return model
