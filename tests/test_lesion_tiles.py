"""Recortes sobre a lesão — o que o rover veria de uma foto de campo."""

from __future__ import annotations

import numpy as np

from app.rover import MIN_TILE_PX
from training.lesion_tiles import classify, grid, iou, lesion_tiles


def test_classifies_by_filename_prefix():
    assert classify("banana_black_leaf_streak_Bing_0003.jpg") == ("banana", "Sigatoka")
    assert classify("citrus_greening_disease_google_12.jpg") == ("citrus", "Greening")
    assert classify("banana_bunchy_top_Bing_0001.jpg") is None


def test_grid_reaches_both_edges():
    boxes = grid(1000, 600, 300)
    assert max(b[2] for b in boxes) == 1000
    assert max(b[3] for b in boxes) == 600
    assert all(b[2] - b[0] == 300 and b[3] - b[1] == 300 for b in boxes)


def test_keeps_only_tiles_with_enough_lesion():
    lesion = np.zeros((600, 900), dtype=bool)
    lesion[40:140, 40:140] = True  # uma lesão no canto superior esquerdo

    tiles = lesion_tiles(lesion)

    assert tiles, "a lesão tem de render ao menos um recorte"
    for (x0, y0, _x1, _y1), fraction in tiles:
        assert fraction >= 0.05
        assert x0 < 140 and y0 < 140, "todo recorte encosta na lesão"


def test_a_photo_without_lesion_gives_nothing():
    assert lesion_tiles(np.zeros((600, 900), dtype=bool)) == []


def test_chosen_tiles_do_not_repeat_each_other():
    lesion = np.ones((900, 900), dtype=bool)

    tiles = lesion_tiles(lesion, max_per_image=6)

    assert len(tiles) == 6
    for i, (a, _) in enumerate(tiles):
        for b, _ in tiles[i + 1 :]:
            assert iou(a, b) <= 0.5


def test_skips_scales_smaller_than_the_rover_tile():
    """Uma foto pequena não vira recorte abaixo do mínimo que o rover classifica."""
    lesion = np.ones((200, 200), dtype=bool)
    for (x0, _y0, x1, _y1), _ in lesion_tiles(lesion):
        assert x1 - x0 >= MIN_TILE_PX
