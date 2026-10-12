"""Testes da geometria de captura do rover."""

from __future__ import annotations

import pytest

from app.rover import (
    FRAME_PX,
    HFOV_DEG,
    MIN_TILE_PX,
    SIDE_CAMERA_HEIGHTS_MM,
    gsd_mm_per_px,
    plan_tiles,
    vertical_coverage_mm,
    zones_at,
)


class TestMatchesTheRoverDesign:
    """Os números têm de bater com o projeto rev E (VALTEXA-SAAS/hardware)."""

    def test_resolution_at_the_design_distance(self):
        """O documento do rover: GSD ≈ 0,26 mm/px a 270 mm."""
        assert gsd_mm_per_px(270, HFOV_DEG, FRAME_PX[0]) == pytest.approx(0.267, abs=0.002)

    @pytest.mark.parametrize(
        ("camera", "expected"),
        [(450.0, (64, 836)), (1000.0, (614, 1386)), (1540.0, (1154, 1926))],
    )
    def test_each_tier_covers_what_the_design_says(self, camera, expected):
        low, high = vertical_coverage_mm(camera, 270)
        assert (round(low), round(high)) == pytest.approx(expected, abs=1)

    def test_the_three_tiers_overlap(self):
        tiers = [vertical_coverage_mm(z, 270) for z in SIDE_CAMERA_HEIGHTS_MM]
        assert all(tiers[i][1] > tiers[i + 1][0] for i in range(len(tiers) - 1))


class TestPlanTiles:
    def test_tiles_have_the_physical_size_of_a_training_photo(self):
        near = plan_tiles(FRAME_PX, 270, 450)
        far = plan_tiles(FRAME_PX, 1000, 450)

        near_side = near[0].box[2] - near[0].box[0]
        far_side = far[0].box[2] - far[0].box[0]
        assert near_side == pytest.approx(562, abs=2), "150 mm a 270 mm"
        assert far_side == pytest.approx(152, abs=2), "150 mm a 1 m"

    def test_tiles_cover_the_whole_frame(self):
        tiles = plan_tiles(FRAME_PX, 270, 1000)

        assert min(t.box[0] for t in tiles) == 0 and min(t.box[1] for t in tiles) == 0
        assert max(t.box[2] for t in tiles) == FRAME_PX[0]
        assert max(t.box[3] for t in tiles) == FRAME_PX[1]

    def test_upper_tiles_are_higher_off_the_ground(self):
        tiles = plan_tiles(FRAME_PX, 270, 1000)
        top = min(tiles, key=lambda t: t.box[1])
        bottom = max(tiles, key=lambda t: t.box[1])

        assert top.center_height_mm > 1000 > bottom.center_height_mm

    def test_tiles_below_the_ground_are_dropped(self):
        """Câmera a 450 mm, planta a 1 m: a metade de baixo do quadro é chão."""
        tiles = plan_tiles(FRAME_PX, 1000, 450)
        assert all(t.center_height_mm >= 0 for t in tiles)
        assert max(t.box[3] for t in tiles) < FRAME_PX[1]

    def test_banana_tiles_know_pseudostem_from_lower_leaves(self):
        tiles = plan_tiles(FRAME_PX, 1000, 450, crop="banana")
        low = min(tiles, key=lambda t: t.center_height_mm)
        high = max(tiles, key=lambda t: t.center_height_mm)

        assert low.zones == ("pseudocaule",)
        assert "folhas_baixas" in high.zones

    def test_refuses_tiles_too_small_for_the_model(self):
        """A 3 m, 150 mm de planta cabem em ~50 px: o modelo veria borrão ampliado."""
        with pytest.raises(ValueError, match=str(MIN_TILE_PX)):
            plan_tiles(FRAME_PX, 3000, 450)

    @pytest.mark.parametrize("overlap", [-0.1, 1.0])
    def test_refuses_a_meaningless_overlap(self, overlap):
        with pytest.raises(ValueError):
            plan_tiles(FRAME_PX, 270, 450, overlap=overlap)


def test_zones_overlap_at_the_border_on_purpose():
    assert zones_at("banana", 1000.0) == ("pseudocaule", "folhas_baixas")
    with pytest.raises(KeyError):
        zones_at("milho", 100.0)
