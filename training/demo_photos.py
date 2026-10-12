"""Fotos para a demonstração do rover no SaaS (``python -m simulator.pomar`` lá).

    python -m training.demo_photos \\
        --tiles ~/valtexa-raw/_plantseg-recortes/test \\
        --data-dir data \\
        --output ~/valtexa-raw/_demo-pomar

A doença vem do teste de campo: recortes do PlantSeg na escala do rover, os
mesmos da acurácia de campo do MODELS.md, que o modelo nunca viu. A folha sadia
vem do dataset de treino, porque o PlantSeg só tem folha doente — então a
demonstração **não é medida**: a sadia o modelo já viu.

Cada foto sai quadrada com ``--size`` pixels. O SaaS repassa o quadro sem
redimensionar, e o corte de 150 mm do ``/predict/rover-frame`` precisa de pelo
menos :data:`~app.rover.MIN_TILE_PX` pixels; os recortes do teste têm ~120.

Saída: ``<output>/<cultura>/<Classe>/*.jpg``, a pasta que o simulador espera.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageOps

from app.rover import MIN_TILE_PX

CROP_TO_MODEL = {"banana": "banana-foliar", "citrus": "citrus-foliar"}
"""A pasta do dataset de onde sai a folha sadia de cada cultura."""

HEALTHY = "Healthy"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tiles", type=Path, required=True, help="Recortes do teste: <tiles>/<cultura>/<Classe>/."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Datasets: <data-dir>/<modelo>/Healthy/."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    # A 75 mm (a distância que o simulador usa) o recorte de 150 mm é metade da foto.
    if args.size < 2 * MIN_TILE_PX:
        parser.error(f"--size precisa de pelo menos {2 * MIN_TILE_PX} px.")
    return args


def photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES)


def square(path: Path, size: int) -> Image.Image:
    """O centro da foto, quadrado, no tamanho pedido."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, (size, size), Image.Resampling.LANCZOS)


def sources(tiles: Path, data_dir: Path, crop: str) -> dict[str, list[Path]]:
    """As fotos de cada classe de uma cultura: doenças do teste, sadia do dataset."""
    out = {
        folder.name: photos(folder)
        for folder in sorted((tiles / crop).iterdir())
        if folder.is_dir()
    }
    healthy = data_dir / CROP_TO_MODEL[crop] / HEALTHY
    if healthy.is_dir():
        out[HEALTHY] = photos(healthy)
    return {name: files for name, files in out.items() if files}


def build(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    rng = random.Random(args.seed)
    written: dict[str, dict[str, int]] = {}
    for crop in CROP_TO_MODEL:
        if not (args.tiles / crop).is_dir():
            continue
        written[crop] = {}
        for name, files in sources(args.tiles, args.data_dir, crop).items():
            chosen = rng.sample(files, min(args.per_class, len(files)))
            target = args.output / crop / name
            target.mkdir(parents=True, exist_ok=True)
            for index, path in enumerate(sorted(chosen)):
                square(path, args.size).save(target / f"{index:03d}.jpg", quality=90)
            written[crop][name] = len(chosen)
    return written


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for crop, counts in build(args).items():
        print(crop, ", ".join(f"{name}: {n}" for name, n in counts.items()))
        if HEALTHY not in counts:
            print(f"  sem {HEALTHY}: as plantas sadias da demonstração ficarão sem foto")


if __name__ == "__main__":
    main()
