"""Máscaras de segmentação não podem entrar no dataset como fotos.

Aconteceu: 388 máscaras do PlantSeg — mesmo nome da foto, extensão .png —
viraram "fotos" de doença no leaf-foliar, 55 delas no teste isolado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from training.prepare_dataset import execute_plan, looks_like_label_mask


def _mask(path: Path, mode: str = "L") -> Path:
    """Fundo 0 e dois polígonos com o índice da classe — como o PlantSeg publica."""
    image = Image.new(mode, (400, 300), 0)
    draw = ImageDraw.Draw(image)
    fill = 46 if mode == "L" else (46, 46, 46)
    draw.polygon([(50, 50), (200, 60), (120, 90)], fill=fill)
    draw.polygon([(220, 200), (380, 190), (300, 240)], fill=fill)
    image.save(path)
    return path


def _photo(path: Path, gray: bool = False) -> Path:
    rng = np.random.default_rng(0)
    pixels = np.clip(rng.normal(110, 40, (300, 400, 3)), 0, 255).astype(np.uint8)
    if gray:
        Image.fromarray(pixels[..., 0]).save(path)
    else:
        Image.fromarray(pixels).save(path)
    return path


class TestLooksLikeLabelMask:
    def test_a_grayscale_label_map_is_a_mask(self, tmp_path):
        assert looks_like_label_mask(_mask(tmp_path / "corn_rust_1.png")) is True

    def test_an_rgb_label_map_is_a_mask_too(self, tmp_path):
        assert looks_like_label_mask(_mask(tmp_path / "corn_rust_1.png", mode="RGB")) is True

    def test_a_color_photo_is_not(self, tmp_path):
        assert looks_like_label_mask(_photo(tmp_path / "corn_rust_1.jpg")) is False

    def test_a_black_and_white_photo_is_not(self, tmp_path):
        """Cinza, sim, mas com centenas de tons — é foto."""
        assert looks_like_label_mask(_photo(tmp_path / "folha.png", gray=True)) is False

    def test_an_unreadable_file_is_left_to_the_other_checks(self, tmp_path):
        broken = tmp_path / "quebrada.png"
        broken.write_bytes(b"nada")
        assert looks_like_label_mask(broken) is False


class TestTheImportersRefuseMasks:
    def test_prepare_dataset_copies_the_photo_and_skips_its_mask(self, tmp_path, capsys):
        source = tmp_path / "Common_Rust"
        source.mkdir()
        _photo(source / "corn_rust_1.jpg")
        _mask(source / "corn_rust_1.png")

        counts = execute_plan({"Common_Rust": [source]}, tmp_path / "saida")

        assert counts == {"Common_Rust": 1}
        assert [p.suffix for p in (tmp_path / "saida" / "Common_Rust").iterdir()] == [".jpg"]
        assert "máscaras de segmentação recusadas" in capsys.readouterr().out

    def test_import_plantseg_skips_masks(self, tmp_path, monkeypatch, capsys):
        from training import import_plantseg

        source = tmp_path / "plantseg"
        (source / "images").mkdir(parents=True)
        (source / "annotations").mkdir()
        _photo(source / "images" / "corn_rust_7.jpg")
        _mask(source / "annotations" / "corn_rust_7.png")
        output = tmp_path / "classes"
        monkeypatch.setattr(
            sys, "argv", ["import_plantseg", "--source", str(source), "--output", str(output)]
        )

        import_plantseg.main()

        assert [p.name for p in (output / "Common_Rust").iterdir()] == ["plantseg__corn_rust_7.jpg"]
        assert "máscaras de segmentação" in capsys.readouterr().out
