"""Testes do pré-processamento — a camada onde treino e inferência precisam bater."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.inference import preprocess
from app.inference.preprocess import (
    InvalidImageError,
    image_to_array,
    load_image_from_bytes,
    preprocess_batch,
    preprocess_image,
    resize_like_tensorflow,
    scale_to_unit_range,
)
from tests.conftest import make_image_bytes

IMG_SIZE = 224


def test_scale_maps_pixel_range_to_minus_one_one():
    scaled = scale_to_unit_range(np.array([0.0, 127.5, 255.0], dtype=np.float32))
    np.testing.assert_allclose(scaled, [-1.0, 0.0, 1.0])


def test_matches_keras_mobilenet_preprocessing():
    """Garante que a versão em NumPy não divergiu da do Keras.

    É a proteção contra o pior modo de falha do projeto: um pré-processamento
    diferente entre treino e API não gera erro, só derruba a acurácia.
    """
    keras = pytest.importorskip("tensorflow.keras", reason="TensorFlow não instalado")
    pixels = np.linspace(0, 255, 48, dtype=np.float32).reshape(1, 4, 4, 3)
    expected = keras.applications.mobilenet_v2.preprocess_input(pixels.copy())
    np.testing.assert_allclose(scale_to_unit_range(pixels), expected, rtol=1e-6)


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_decodes_supported_formats(image_format):
    image = load_image_from_bytes(make_image_bytes(image_format=image_format))
    assert image.mode == "RGB"


def test_converts_to_model_input_shape(image_bytes):
    array = preprocess_image(load_image_from_bytes(image_bytes), IMG_SIZE)
    assert array.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert array.dtype == np.float32
    assert array.min() >= -1.0 and array.max() <= 1.0


def test_stacks_batch_in_a_single_array():
    images = [load_image_from_bytes(make_image_bytes()) for _ in range(3)]
    assert preprocess_batch(images, IMG_SIZE).shape == (3, IMG_SIZE, IMG_SIZE, 3)


def test_resize_ignores_aspect_ratio_like_keras():
    """O treino distorce a imagem para o quadrado; a API precisa fazer o mesmo."""
    wide = load_image_from_bytes(make_image_bytes(size=(800, 200)))
    assert image_to_array(wide, IMG_SIZE).shape == (IMG_SIZE, IMG_SIZE, 3)


def _rotated_by_exif() -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotacionar 90°
    return make_image_bytes(size=(320, 240), exif=exif)


def test_keeps_the_pixels_as_stored_like_training():
    """O treino decodifica sem girar pelo EXIF; girar aqui custou 0,4 a 0,5 ponto.

    Um terço das fotos de milho traz rotação no EXIF. Girada, a foto é espremida
    para o quadrado na outra direção — e o modelo nunca viu esse achatamento.
    """
    assert load_image_from_bytes(_rotated_by_exif()).size == (320, 240)


def test_applies_exif_orientation_when_asked():
    """Para mostrar a foto a uma pessoa, a rotação do EXIF é a certa."""
    assert load_image_from_bytes(_rotated_by_exif(), apply_exif=True).size == (240, 320)


def test_resize_matches_tensorflow_bilinear_by_hand():
    """Centros de pixel, sem antialias: a conta do tf.image.resize, feita à mão."""
    pixels = np.array([[0, 10], [20, 30]], dtype=np.float32)[..., None].repeat(3, axis=2)

    resized = resize_like_tensorflow(pixels, 4)[..., 0]

    np.testing.assert_allclose(resized[0], [0.0, 2.5, 7.5, 10.0])
    np.testing.assert_allclose(resized[:, 0], [0.0, 5.0, 15.0, 20.0])


@pytest.mark.parametrize("shape", [(128, 128), (300, 240), (1000, 750), (97, 413)])
def test_resize_is_identical_to_tensorflow(shape):
    """A garantia que importa: a API redimensiona como o treino redimensionou."""
    tf = pytest.importorskip("tensorflow", reason="TensorFlow não instalado")
    pixels = np.random.default_rng(0).integers(0, 256, (*shape, 3)).astype(np.uint8)

    expected = tf.image.resize(tf.cast(pixels, tf.float32), (224, 224), method="bilinear").numpy()

    np.testing.assert_allclose(resize_like_tensorflow(pixels, 224), expected, atol=1e-3)


def test_severity_keeps_the_pillow_resize_it_was_trained_with():
    image = load_image_from_bytes(make_image_bytes(size=(320, 240)))
    pillow = image_to_array(image, IMG_SIZE, resampling="pillow")
    expected = scale_to_unit_range(
        np.asarray(image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR), np.float32)
    )
    np.testing.assert_allclose(pillow, expected)


def test_rejects_empty_upload():
    with pytest.raises(InvalidImageError, match="vazio"):
        load_image_from_bytes(b"", "vazio.jpg")


def test_rejects_bytes_that_are_not_an_image():
    with pytest.raises(InvalidImageError, match=r"quebrado\.jpg"):
        load_image_from_bytes(b"isto nao e uma imagem", "quebrado.jpg")


def test_rejects_oversized_image(monkeypatch):
    monkeypatch.setattr(preprocess, "MAX_IMAGE_PIXELS", 1_000)
    with pytest.raises(InvalidImageError, match="excede o limite"):
        load_image_from_bytes(make_image_bytes(size=(320, 240)), "bomba.jpg")


def test_rejects_empty_batch():
    with pytest.raises(InvalidImageError):
        preprocess_batch([], IMG_SIZE)
