"""A pasta de fotos da demonstração do rover no SaaS."""

from __future__ import annotations

import pytest
from PIL import Image

from training import demo_photos


def _photo(path, size=(120, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 120, 40)).save(path)


def test_disease_from_the_field_test_and_healthy_from_the_dataset(tmp_path):
    for i in range(3):
        _photo(tmp_path / "tiles" / "banana" / "Sigatoka" / f"s{i}.jpg")
    _photo(tmp_path / "data" / "banana-foliar" / "Healthy" / "h.jpg", size=(512, 512))

    demo_photos.main(
        [
            "--tiles", str(tmp_path / "tiles"),
            "--data-dir", str(tmp_path / "data"),
            "--output", str(tmp_path / "out"),
            "--per-class", "2",
        ]
    )  # fmt: skip

    sigatoka = sorted((tmp_path / "out" / "banana" / "Sigatoka").iterdir())
    assert len(sigatoka) == 2
    assert len(list((tmp_path / "out" / "banana" / "Healthy").iterdir())) == 1
    with Image.open(sigatoka[0]) as image:
        # O SaaS não redimensiona: a foto já sai com pixel para o corte de 150 mm.
        assert image.size == (320, 320)


def test_refuses_a_size_too_small_for_the_tile(tmp_path):
    with pytest.raises(SystemExit):
        demo_photos.parse_args(
            ["--tiles", "t", "--data-dir", "d", "--output", "o", "--size", "150"]
        )
