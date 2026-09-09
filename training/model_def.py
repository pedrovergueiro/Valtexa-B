"""Arquitetura do classificador: MobileNetV2 pré-treinada + cabeça densa."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.keras_backend import configure

if TYPE_CHECKING:  # só para as anotações — em tempo de execução o Keras é tardio
    import keras

MODEL_NAME = "corn_disease_mobilenetv2"

EMBEDDING_LAYER = "embedding"
"""Nome da camada densa que vira a "assinatura visual" da imagem.

É de onde a detecção de novidade lê o vetor de 128 dimensões — nomear a camada
evita depender de índice posicional, que quebra em silêncio se a arquitetura
mudar. Ver :mod:`app.inference.novelty`.
"""

DEFAULT_ARCH = "mobilenetv2"

ARCHITECTURES = {
    # nome no keras.applications, parâmetros aproximados, top-1 no ImageNet
    "mobilenetv2": ("MobileNetV2", "2,3M", 0.713),
    "efficientnetv2b0": ("EfficientNetV2B0", "7,2M", 0.787),
    "efficientnetv2s": ("EfficientNetV2S", "21,6M", 0.838),
}
"""Bases convolucionais disponíveis, da mais leve para a mais precisa.

O construtor é guardado pelo **nome**, e não pela função: assim esta tabela pode
ser lida — pelo ``--help`` do treino e pela suíte de testes — sem carregar o
Keras, que custa segundos e que a suíte não exige.

A ``mobilenetv2`` continua sendo o padrão porque o destino do projeto é embarcar
na ground station — mas quando o que se quer é acurácia, a base é o parâmetro
que mais move o número: são 12 pontos de top-1 no ImageNet entre a primeira e a
última desta tabela, e essa diferença se transfere em boa parte para o
diagnóstico foliar.

**Todas são instanciadas na mesma escala de entrada, ``[-1, 1]``.** Nas
EfficientNetV2 isso exige ``include_preprocessing=False``: com o padrão
``True`` elas embutem uma camada que espera ``[0, 255]`` e reescala sozinha, o
que aplicaria a normalização duas vezes sobre o que o
:mod:`app.inference.preprocess` já entregou normalizado. Não daria erro — só
derrubaria a acurácia em silêncio, que é o modo de falha que este projeto mais
teme.
"""


def _load_keras():
    """Importa o Keras, com o backend já escolhido.

    Tardio de propósito: carregar o Keras custa segundos, e assim o ``--help``
    do treino e a suíte de testes — que exercitam caminhos, argumentos e
    rótulos, não a rede — rodam sem precisar dele instalado.
    """
    configure()
    import keras
    from keras import layers

    return keras, layers


def build_model(
    img_size: int, num_classes: int, dropout: float = 0.3, arch: str = DEFAULT_ARCH
) -> keras.Model:
    """Monta o modelo com a base convolucional congelada.

    A base nasce congelada de propósito: treinar a cabeça primeiro evita que os
    gradientes enormes de uma cabeça aleatória destruam os pesos da ImageNet.
    O descongelamento da segunda fase é feito por :func:`unfreeze_base`.
    """
    if arch not in ARCHITECTURES:
        disponiveis = ", ".join(ARCHITECTURES)
        raise ValueError(f"Arquitetura desconhecida: {arch!r}. Disponíveis: {disponiveis}.")

    keras, layers = _load_keras()

    constructor = getattr(keras.applications, ARCHITECTURES[arch][0])
    kwargs = {
        "input_shape": (img_size, img_size, 3),
        "include_top": False,
        "weights": "imagenet",
    }
    if arch.startswith("efficientnetv2"):
        kwargs["include_preprocessing"] = False

    base_model = constructor(**kwargs)
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

    return keras.Model(inputs, outputs, name=f"corn_disease_{arch}")


def find_base_model(model: keras.Model) -> keras.Model:
    """Devolve a MobileNetV2 aninhada dentro do modelo.

    Procura pela camada que é ela própria um ``Model``, em vez de indexar por
    posição: o nome que o Keras dá à base varia conforme ``alpha`` e o tamanho
    de entrada, e um índice fixo quebra em silêncio se a arquitetura mudar.
    """
    keras, _ = _load_keras()

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError(f"Nenhuma base convolucional encontrada em {model.name!r}.")


def unfreeze_base(model: keras.Model, trainable_layers: int | None = None) -> keras.Model:
    """Libera os pesos da base para o fine-tuning.

    Args:
        trainable_layers: quantas camadas finais liberar. ``None`` libera a base
            inteira.

    Liberar só o topo é o padrão por um motivo prático: as camadas iniciais de
    uma rede pré-treinada codificam borda, textura e cor — que valem tanto para
    o ImageNet quanto para uma folha de milho —, enquanto as finais codificam as
    formas específicas das mil classes do ImageNet, que é o que não serve aqui.
    Retreinar tudo com um dataset pequeno gasta épocas mexendo no que já estava
    certo e ainda arrisca desfazê-lo.

    O modelo precisa ser recompilado depois desta chamada para que o Keras
    registre o novo conjunto de variáveis treináveis.
    """
    base_model = find_base_model(model)
    base_model.trainable = True

    if trainable_layers is not None:
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    return base_model
