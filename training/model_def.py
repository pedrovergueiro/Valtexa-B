"""Arquitetura do classificador: MobileNetV2 pré-treinada + cabeça densa."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

MODEL_NAME = "corn_disease_mobilenetv2"

EMBEDDING_LAYER = "embedding"
"""Nome da camada densa que vira a "assinatura visual" da imagem.

É de onde a detecção de novidade lê o vetor de 128 dimensões — nomear a camada
evita depender de índice posicional, que quebra em silêncio se a arquitetura
mudar. Ver :mod:`app.inference.novelty`.
"""


def build_model(img_size: int, num_classes: int, dropout: float = 0.3) -> keras.Model:
    """Monta o modelo com a base convolucional congelada.

    A base nasce congelada de propósito: treinar a cabeça primeiro evita que os
    gradientes enormes de uma cabeça aleatória destruam os pesos da ImageNet.
    O descongelamento da segunda fase é feito por :func:`unfreeze_base`.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    # ``training=False`` fixa as camadas de BatchNorm em modo de inferência, de
    # forma permanente. É intencional: durante o fine-tuning, deixar a BN
    # atualizar suas estatísticas com lotes pequenos degrada a base pré-treinada.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", name=EMBEDDING_LAYER)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name=MODEL_NAME)


def find_base_model(model: keras.Model) -> keras.Model:
    """Devolve a MobileNetV2 aninhada dentro do modelo.

    Procura pela camada que é ela própria um ``Model``, em vez de indexar por
    posição: o nome que o Keras dá à base varia conforme ``alpha`` e o tamanho
    de entrada, e um índice fixo quebra em silêncio se a arquitetura mudar.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError(f"Nenhuma base convolucional encontrada em {model.name!r}.")


def unfreeze_base(model: keras.Model) -> keras.Model:
    """Libera os pesos da base para o fine-tuning.

    O modelo precisa ser recompilado depois desta chamada para que o Keras
    registre o novo conjunto de variáveis treináveis.
    """
    base_model = find_base_model(model)
    base_model.trainable = True
    return base_model
