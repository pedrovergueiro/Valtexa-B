"""Testes da troca de fundo — a folha sai do papel e vai para o pomar."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from training.background import (
    composite,
    leaf_mask,
    load_backgrounds,
    mask_or_empty,
    otsu_threshold,
    swap_backgrounds,
    swap_with_masks,
)

SIZE = 224


def leaf_on_paper(paper=(215, 212, 205), lesion=(218, 214, 200)) -> np.ndarray:
    """Folha verde sobre papel claro, com uma lesão da cor do papel no meio."""
    image = Image.new("RGB", (SIZE, SIZE), paper)
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 60, 184, 164), fill=(40, 110, 40))
    draw.ellipse((100, 100, 124, 124), fill=lesion)
    return np.asarray(image, dtype=np.float32)


def field_photo(seed: int = 0) -> np.ndarray:
    """Textura de folhagem sem fundo liso — como as fotos de campo."""
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 200, (28, 28, 3)).astype(np.uint8)
    return np.asarray(Image.fromarray(base).resize((SIZE, SIZE)), dtype=np.float32)


class TestLeafMask:
    def test_separates_the_leaf_from_the_paper(self):
        mask = leaf_mask(leaf_on_paper())

        assert mask is not None
        assert mask[112, 60] > 0.9, "dentro da folha"
        assert mask[10, 10] < 0.1, "no papel"

    def test_a_pale_lesion_inside_the_leaf_stays_leaf(self):
        """Pela cor, a lesão parece papel — mas não encosta na borda."""
        mask = leaf_mask(leaf_on_paper())
        assert mask[112, 112] > 0.9

    def test_a_field_photo_is_left_alone(self):
        assert leaf_mask(field_photo()) is None

    def test_a_leaf_filling_the_frame_is_left_alone(self):
        full = np.full((SIZE, SIZE, 3), (40, 110, 40), dtype=np.float32)
        assert leaf_mask(full) is None

    def test_otsu_splits_two_groups(self):
        values = np.concatenate([np.full(500, 10.0), np.full(500, 90.0)])
        assert 10 < otsu_threshold(values) < 90


class TestComposite:
    def test_puts_the_leaf_over_the_orchard(self):
        image = leaf_on_paper()
        background = np.zeros_like(image)
        background[..., 1] = 255  # "pomar" verde puro

        out = composite(image, leaf_mask(image), background)

        assert tuple(out[10, 10].round()) == (0, 255, 0), "o papel virou pomar"
        np.testing.assert_allclose(out[112, 60], image[112, 60], atol=1), "a folha ficou"

    def test_swaps_only_a_fraction_of_the_batch(self):
        batch = np.stack([leaf_on_paper() for _ in range(40)])
        backgrounds = np.zeros((3, SIZE, SIZE, 3), dtype=np.uint8)

        out = swap_backgrounds(batch, backgrounds, 0.5, np.random.default_rng(0))

        swapped = int((out[:, 10, 10].sum(axis=1) < 10).sum())
        assert 10 < swapped < 30

    def test_field_photos_pass_through_untouched(self):
        batch = np.stack([field_photo(seed) for seed in range(4)])
        out = swap_backgrounds(batch, np.zeros((2, SIZE, SIZE, 3)), 1.0, np.random.default_rng(0))
        np.testing.assert_array_equal(out, batch)


class TestCachedMasks:
    """A máscara vai para o cache: calculada uma vez, usada em toda época."""

    def test_is_uint8_and_empty_for_a_field_photo(self):
        assert mask_or_empty(leaf_on_paper()).dtype == np.uint8
        assert not mask_or_empty(field_photo()).any()

    def test_swapping_with_cached_masks_matches_computing_them(self):
        batch = np.stack([leaf_on_paper(), field_photo()])
        backgrounds = np.random.default_rng(3).integers(0, 255, (4, SIZE, SIZE, 3)).astype(np.uint8)
        masks = np.stack([mask_or_empty(image) for image in batch])

        cached = swap_with_masks(batch, masks, backgrounds, 1.0, np.random.default_rng(7))
        fresh = swap_backgrounds(batch, backgrounds, 1.0, np.random.default_rng(7))

        np.testing.assert_array_equal(cached, fresh)
        np.testing.assert_array_equal(cached[1], batch[1]), "foto de campo intacta"


def test_loads_square_background_crops(tmp_path):
    for i in range(2):
        Image.fromarray(field_photo(i).astype(np.uint8)).resize((640, 360)).save(
            tmp_path / f"{i}.jpg"
        )

    crops = load_backgrounds(tmp_path, SIZE, count=5, seed=1)

    assert crops.shape == (5, SIZE, SIZE, 3) and crops.dtype == np.uint8


def test_refuses_an_empty_background_folder(tmp_path):
    with pytest.raises(ValueError, match="fundo"):
        load_backgrounds(tmp_path, SIZE, count=1)
