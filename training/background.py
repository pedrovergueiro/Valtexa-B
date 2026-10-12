"""Troca de fundo: a folha fotografada sobre papel, colada sobre um pomar de verdade.

Os três datasets públicos de citros (limão e laranja de Bangladesh) fotografam a
folha **destacada, sobre papel liso**. O rover a vê **na árvore**, com outras
folhas, galhos e céu atrás. Um modelo treinado só com papel aprende também o
papel — e pior: o papel muda de cor por classe (azulado numa, cinza noutra), o
que é um atalho pronto.

Aqui, a cada época e com probabilidade ``p``, a folha é recortada do papel e
colada sobre um recorte de fundo real — quadros de vídeos de pomar filmados do
chão. O modelo passa a ver a mesma lesão sobre fundos diferentes e deixa de ter
onde se apoiar que não seja a folha.

## Como a folha é separada do papel

O papel é liso e ocupa a borda da foto. A cor dele sai da mediana da borda; a
distância de cada pixel a essa cor é binarizada pelo limiar de Otsu; o que está
ligado à borda e parece papel é fundo, e **todo o resto é folha** — inclusive
uma lesão clara no meio dela, que por cor pareceria papel mas não toca a borda.

Foto sem fundo liso (as de campo, como as da banana) não tem uma cor de borda
dominante: a separação sai absurda (folha ocupando quase nada ou quase tudo) e a
imagem é devolvida como está.

Não depende do TensorFlow: roda no ``tf.data`` por ``tf.numpy_function`` e nos
testes com NumPy puro.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BORDER = 6
"""Espessura, em pixels (a 224), da faixa de borda que estima a cor do papel."""

MIN_LEAF, MAX_LEAF = 0.08, 0.92
"""Fração do quadro que a folha pode ocupar para a separação valer."""

MAX_BORDER_SPREAD = 28.0
"""Desvio máximo das cores da borda (0-255). Acima disso a borda não é papel liso."""


def paper_color(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Cor mediana da borda e o quanto ela varia — papel liso varia pouco."""
    ring = np.concatenate(
        [
            image[:BORDER].reshape(-1, 3),
            image[-BORDER:].reshape(-1, 3),
            image[:, :BORDER].reshape(-1, 3),
            image[:, -BORDER:].reshape(-1, 3),
        ]
    ).astype(np.float32)
    median = np.median(ring, axis=0)
    spread = float(np.median(np.abs(ring - median).max(axis=1)))
    return median, spread


def otsu_threshold(values: np.ndarray) -> float:
    """Limiar que melhor separa dois grupos num histograma (Otsu, 1979)."""
    histogram, edges = np.histogram(values, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(histogram)
    total = weight[-1]
    mean = np.cumsum(histogram * centers)
    between = (mean[-1] * weight - mean * total) ** 2 / np.maximum(weight * (total - weight), 1)
    return float(centers[int(np.argmax(between))])


def leaf_mask(image: np.ndarray) -> np.ndarray | None:
    """Máscara ``float32`` em ``[0, 1]`` da folha, ou ``None`` se o fundo não for papel.

    Args:
        image: ``(H, W, 3)`` em ``[0, 255]``.
    """
    image = np.asarray(image, dtype=np.float32)
    color, spread = paper_color(image)
    if spread > MAX_BORDER_SPREAD:
        return None

    distance = np.sqrt(((image - color) ** 2).sum(axis=2))
    paper_like = distance < max(otsu_threshold(distance), 12.0)
    background = _connected_to_border(paper_like)
    leaf = ~background
    leaf = _open_close(leaf)

    fraction = float(leaf.mean())
    if not MIN_LEAF <= fraction <= MAX_LEAF:
        return None
    return _feather(leaf.astype(np.float32))


def composite(image: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    """A folha de ``image`` sobre ``background``, pela máscara (mesmo tamanho)."""
    alpha = mask[..., None]
    return image * alpha + background * (1.0 - alpha)


def mask_or_empty(image: np.ndarray) -> np.ndarray:
    """A máscara da folha em ``uint8`` (0-255), ou toda zero se não há papel a trocar.

    É a forma que vai para o cache do ``tf.data``: a máscara só depende da
    imagem, então é calculada **uma vez** por foto, e não a cada época — era ela
    que triplicava o tempo do passo de treino (394 ms contra 114 ms, medido no
    citros). Em ``uint8`` ocupa um quarto do ``float32``.
    """
    mask = leaf_mask(image)
    if mask is None:
        return np.zeros(np.shape(image)[:2], dtype=np.uint8)
    return np.round(mask * 255).astype(np.uint8)


def swap_with_masks(
    batch: np.ndarray,
    masks: np.ndarray,
    backgrounds: np.ndarray,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Troca o fundo de parte de um lote, com as máscaras já calculadas.

    Args:
        batch: ``(N, H, W, 3)`` em ``[0, 255]``.
        masks: ``(N, H, W)`` de :func:`mask_or_empty`; toda zero = não trocar.
    """
    out = np.array(batch, dtype=np.float32, copy=True)
    for i in range(len(out)):
        if rng.random() >= probability or not masks[i].any():
            continue
        background = backgrounds[rng.integers(len(backgrounds))]
        if rng.random() < 0.5:
            background = background[:, ::-1]
        alpha = masks[i].astype(np.float32) / 255.0
        out[i] = composite(out[i], alpha, background.astype(np.float32))
    return out


def swap_backgrounds(
    batch: np.ndarray, backgrounds: np.ndarray, probability: float, rng: np.random.Generator
) -> np.ndarray:
    """Troca o fundo de parte das imagens de um lote ``(N, H, W, 3)`` em ``[0, 255]``."""
    masks = np.stack([mask_or_empty(image) for image in batch])
    return swap_with_masks(batch, masks, backgrounds, probability, rng)


def load_backgrounds(folder: Path, size: int, count: int, seed: int = 0) -> np.ndarray:
    """``count`` recortes quadrados de fundo, sorteados das imagens de ``folder``.

    Recortes de 30% a 70% do lado menor da imagem, reduzidos para ``size``: a
    escala de uma copa vista de perto, que é o que fica atrás de uma folha.

    Raises:
        ValueError: se a pasta não tiver imagens.
    """
    files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise ValueError(f"Nenhuma imagem de fundo em {folder}.")
    rng = np.random.default_rng(seed)
    crops = np.empty((count, size, size, 3), dtype=np.uint8)
    for n in range(count):
        with Image.open(files[rng.integers(len(files))]) as source:
            image = source.convert("RGB")
        side = int(min(image.size) * rng.uniform(0.3, 0.7))
        x = int(rng.integers(0, image.width - side + 1))
        y = int(rng.integers(0, image.height - side + 1))
        crop = image.crop((x, y, x + side, y + side)).resize(
            (size, size), Image.Resampling.BILINEAR
        )
        crops[n] = np.asarray(crop)
    return crops


def _connected_to_border(candidate: np.ndarray) -> np.ndarray:
    """Pixels de ``candidate`` ligados à borda por outros pixels de ``candidate``.

    Preenchimento por dilatação repetida — sem SciPy, que a API não carrega.
    """
    reached = np.zeros_like(candidate)
    reached[0, :], reached[-1, :] = candidate[0, :], candidate[-1, :]
    reached[:, 0], reached[:, -1] = candidate[:, 0], candidate[:, -1]
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= candidate
        if (grown == reached).all():
            return reached
        reached = grown


def _shift_all(mask: np.ndarray, combine) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    views = [
        padded[1 + dy : padded.shape[0] - 1 + dy, 1 + dx : padded.shape[1] - 1 + dx]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ]
    return combine.reduce(views)


def _open_close(mask: np.ndarray) -> np.ndarray:
    """Tira pontos soltos (abertura) e fecha frestas (fechamento), 3x3."""
    opened = _shift_all(_shift_all(mask, np.logical_and), np.logical_or)
    return _shift_all(_shift_all(opened, np.logical_or), np.logical_and)


def _feather(mask: np.ndarray, passes: int = 2) -> np.ndarray:
    """Borda suave: sem ela, o recorte vira uma quina que o modelo aprenderia."""
    for _ in range(passes):
        padded = np.pad(mask, 1, mode="edge")
        mask = (
            sum(
                padded[1 + dy : padded.shape[0] - 1 + dy, 1 + dx : padded.shape[1] - 1 + dx]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            )
            / 9.0
        )
    return mask
