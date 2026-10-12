"""Recortes na escala do rover, cortados sobre as lesões marcadas no PlantSeg.

    python -m training.lesion_tiles \\
        --source ~/valtexa-raw/plantseg/plantseg \\
        --output ~/valtexa-raw/_plantseg-recortes

O rover não classifica a planta inteira: ele fatia o quadro em recortes de
~150 mm (ver :mod:`app.rover`) e classifica cada um — um pedaço de limbo que
ocupa o recorte todo. Os datasets de treino são fotos nessa escala. As fotos do
PlantSeg não são: há bananal inteiro, planta murcha de corpo inteiro, uma mão
segurando a folha. Medir o modelo nelas mede outra tarefa — e ele erra, com toda
a razão (a Fusariose de corpo inteiro saiu como Sigatoka, a Sigatoka vista de
longe saiu como Fusariose).

Mas o PlantSeg traz a **máscara da lesão** de cada foto. Aqui cada foto vira uma
grade de recortes quadrados, em duas escalas, e fica só o recorte que tem lesão
de verdade dentro. O resultado é o que o rover veria se passasse por aquela
planta: pedaços de folha doente, com o fundo de campo que as fotos de treino
não têm.

Os recortes saem separados pelo split oficial do PlantSeg::

    <saida>/train/<cultura>/<Classe>/   -> entram no treino
    <saida>/test/<cultura>/<Classe>/    -> teste de campo (``evaluate --external``)

Recortes de uma mesma foto ficam sempre do mesmo lado — o split oficial é por
foto, então nenhuma lesão aparece no treino e no teste.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from app.rover import MIN_TILE_PX

PREFIX_TO_CLASS = {
    "banana_black_leaf_streak": ("banana", "Sigatoka"),
    "banana_panama_disease": ("banana", "Fusarium_Wilt"),
    "banana_cordana_leaf_spot": ("banana", "Cordana"),
    "citrus_canker": ("citrus", "Canker"),
    "citrus_greening_disease": ("citrus", "Greening"),
}
"""Prefixos de arquivo do PlantSeg com classe equivalente no catálogo.

Ficam de fora os da banana sem classe no modelo: antracnose (é do fruto),
*bunchy top* (vírus, sem dataset de treino público nesta escala) e *cigar end
rot* (ponta do fruto).
"""

SCALES = (1 / 2, 1 / 3)
"""Lado do recorte, como fração do lado menor da foto.

A foto não tem escala física (não há profundidade), então não dá para cortar
150 mm exatos. Duas escalas cobrem a foto de perto (recorte grande, lesão
nítida) e a de longe (recorte pequeno, que isola um pedaço de folha).
"""

MIN_LESION = 0.05
"""Fração mínima do recorte coberta por lesão para ele herdar o rótulo da foto.

Abaixo disso o recorte é, na maior parte, folha sã ou fundo — rotulá-lo com a
doença ensinaria o modelo a ver doença em tecido sadio.
"""

MAX_PER_IMAGE = 6
"""Teto de recortes por foto: sem ele, uma foto grande e muito lesionada viraria
dezenas de exemplos quase iguais e pesaria mais que as outras."""

MAX_OVERLAP = 0.5
"""IoU máximo entre dois recortes escolhidos da mesma foto."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recortes com lesão do PlantSeg, na escala do rover",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Pasta com images/ e annotations/."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-lesion", type=float, default=MIN_LESION)
    parser.add_argument("--max-per-image", type=int, default=MAX_PER_IMAGE)
    return parser.parse_args()


def classify(filename: str) -> tuple[str, str] | None:
    """``(cultura, Classe)`` pelo prefixo do nome, ou ``None`` se não interessa."""
    name = filename.lower()
    for prefix in sorted(PREFIX_TO_CLASS, key=len, reverse=True):
        if name.startswith(prefix):
            return PREFIX_TO_CLASS[prefix]
    return None


def grid(
    width: int, height: int, side: int, overlap: float = 0.25
) -> list[tuple[int, int, int, int]]:
    """Quadrados de ``side`` cobrindo a foto, com sobreposição, até as bordas."""
    step = max(1, round(side * (1 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= side:
            return [0]
        values = list(range(0, length - side + 1, step))
        if values[-1] != length - side:
            values.append(length - side)
        return values

    return [(x, y, x + side, y + side) for y in starts(height) for x in starts(width)]


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


def lesion_tiles(
    lesion: np.ndarray,
    *,
    min_lesion: float = MIN_LESION,
    max_per_image: int = MAX_PER_IMAGE,
    scales: tuple[float, ...] = SCALES,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Recortes com lesão suficiente, os mais lesionados primeiro e sem se repetir.

    Args:
        lesion: máscara booleana ``(H, W)`` da lesão.

    Returns:
        ``(caixa, fração de lesão)`` de cada recorte escolhido.
    """
    height, width = lesion.shape
    # Imagem integral: a soma de qualquer retângulo sai em O(1).
    integral = np.pad(lesion.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    candidates = []
    for scale in scales:
        side = round(min(width, height) * scale)
        if side < MIN_TILE_PX:
            continue
        for box in grid(width, height, side):
            x0, y0, x1, y1 = box
            total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            fraction = float(total) / (side * side)
            if fraction >= min_lesion:
                candidates.append((box, fraction))

    chosen: list[tuple[tuple[int, int, int, int], float]] = []
    for box, fraction in sorted(candidates, key=lambda item: -item[1]):
        if all(iou(box, other) <= MAX_OVERLAP for other, _ in chosen):
            chosen.append((box, fraction))
        if len(chosen) == max_per_image:
            break
    return chosen


def main() -> None:
    args = parse_args()
    counts: Counter = Counter()
    photos: Counter = Counter()

    for split in ("train", "test"):
        for image_path in sorted((args.source / "images" / split).glob("*.jpg")):
            target = classify(image_path.name)
            if target is None:
                continue
            mask_path = args.source / "annotations" / split / f"{image_path.stem}.png"
            if not mask_path.exists():
                continue
            crop, label = target
            with Image.open(mask_path) as mask_image:
                lesion = np.asarray(mask_image.convert("L")) > 0
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            if image.size != (lesion.shape[1], lesion.shape[0]):
                continue  # máscara de outra resolução: não dá para confiar no recorte

            tiles = lesion_tiles(
                lesion, min_lesion=args.min_lesion, max_per_image=args.max_per_image
            )
            folder = args.output / split / crop / label
            folder.mkdir(parents=True, exist_ok=True)
            for index, (box, _) in enumerate(tiles):
                image.crop(box).save(folder / f"{image_path.stem}_r{index}.jpg", quality=92)
            counts[(split, crop, label)] += len(tiles)
            photos[(split, crop, label)] += 1 if tiles else 0

    print(f"{'split':6s} {'cultura':8s} {'classe':15s} {'fotos':>6s} {'recortes':>9s}")
    for key in sorted(counts):
        split, crop, label = key
        print(f"{split:6s} {crop:8s} {label:15s} {photos[key]:>6d} {counts[key]:>9d}")


if __name__ == "__main__":
    main()
