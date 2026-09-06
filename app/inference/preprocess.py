"""Decodificação e pré-processamento de imagens para inferência.

Este módulo é a **fonte única de verdade** do pré-processamento: o pipeline de
treino (:mod:`training.dataset`) importa daqui as mesmas operações. Qualquer
divergência entre treino e inferência não gera erro — ela só derruba a acurácia
em produção, silenciosamente —, por isso as duas pontas compartilham este código.

O pipeline reproduz o que o Keras faz no treino:

1. redimensiona para ``img_size`` x ``img_size`` com interpolação **bilinear e
   sem preservar o aspect ratio**, igual a
   ``keras.utils.image_dataset_from_directory(image_size=...)``;
2. escala os pixels de ``[0, 255]`` para ``[-1, 1]``, igual a
   ``keras.applications.mobilenet_v2.preprocess_input``.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 80_000_000
"""Teto de pixels por imagem, como defesa contra *decompression bombs*.

Cabe folgadamente uma foto de drone de 80 MP. O valor também é aplicado ao
limite global do Pillow como segunda camada de proteção.
"""

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class InvalidImageError(ValueError):
    """A imagem enviada não pôde ser decodificada ou é grande demais."""


def scale_to_unit_range(pixels):
    """Escala pixels de ``[0, 255]`` para ``[-1, 1]``.

    Equivalente a ``keras.applications.mobilenet_v2.preprocess_input``. É escrito
    com aritmética simples de propósito: funciona tanto em ``numpy.ndarray``
    (inferência) quanto em tensores do TensorFlow (pipeline ``tf.data`` do
    treino), sem obrigar a API a importar o Keras só para normalizar.
    """
    return pixels / 127.5 - 1.0


def load_image_from_bytes(data: bytes, filename: str = "upload") -> Image.Image:
    """Decodifica os bytes de um upload em uma imagem RGB.

    Corrige a orientação a partir do EXIF — fotos de drone e de celular
    costumam vir rotacionadas apenas por metadado.

    Raises:
        InvalidImageError: bytes vazios, formato não reconhecido ou imagem
            acima de :data:`MAX_IMAGE_PIXELS`.
    """
    if not data:
        raise InvalidImageError(f"{filename}: arquivo vazio.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            # image.size vem do cabeçalho, antes de decodificar os pixels — é o
            # que permite recusar uma bomba sem alocar memória por ela.
            _reject_oversized(image, filename)
            oriented = ImageOps.exif_transpose(image)
            return oriented.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidImageError(f"{filename}: não foi possível ler a imagem ({exc}).") from exc


def image_to_array(image: Image.Image, img_size: int) -> np.ndarray:
    """Converte uma imagem RGB no tensor ``(img_size, img_size, 3)`` do modelo."""
    resized = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    return scale_to_unit_range(pixels)


def preprocess_batch(images: Iterable[Image.Image], img_size: int) -> np.ndarray:
    """Empilha várias imagens em um único lote ``(N, img_size, img_size, 3)``.

    Uma passada só pelo modelo é bem mais barata que N passadas de tamanho 1.
    """
    arrays = [image_to_array(image, img_size) for image in images]
    if not arrays:
        raise InvalidImageError("Nenhuma imagem para processar.")
    return np.stack(arrays, axis=0)


def preprocess_image(image: Image.Image, img_size: int) -> np.ndarray:
    """Converte uma imagem no lote unitário ``(1, img_size, img_size, 3)``."""
    return preprocess_batch([image], img_size)


def _reject_oversized(image: Image.Image, filename: str) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise InvalidImageError(f"{filename}: imagem com dimensões inválidas ({width}x{height}).")
    if width * height > MAX_IMAGE_PIXELS:
        raise InvalidImageError(
            f"{filename}: imagem de {width}x{height} excede o limite de "
            f"{MAX_IMAGE_PIXELS:,} pixels."
        )
