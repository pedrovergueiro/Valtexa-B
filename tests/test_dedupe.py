"""Deduplicação — cópia de verdade sai; folha diferente sobre o mesmo papel fica."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from training.dedupe_dataset import dhash, group_duplicates, thumbnail_confirmation


def leaf(path: Path, color, lesion=None, paper=(225, 225, 228)) -> Path:
    """Folha elíptica no centro de um papel claro — a mesma silhueta sempre."""
    image = Image.new("RGB", (240, 240), paper)
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 30, 180, 210), fill=color)
    if lesion:
        draw.ellipse((90, 80, 150, 160), fill=lesion)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)
    return path


def signatures(paths):
    return {p: dhash(p) for p in paths}


def test_different_leaves_on_the_same_paper_share_a_dhash(tmp_path):
    """O defeito que motivou a confirmação: o dHash sozinho não as distingue."""
    a = leaf(tmp_path / "Healthy" / "a.jpg", (40, 120, 40))
    b = leaf(tmp_path / "Canker" / "b.jpg", (150, 150, 60), lesion=(120, 80, 40))

    assert group_duplicates(signatures([a, b]), 4) == [sorted([a, b])]


def test_the_thumbnail_keeps_different_leaves_apart(tmp_path):
    a = leaf(tmp_path / "Healthy" / "a.jpg", (40, 120, 40))
    b = leaf(tmp_path / "Canker" / "b.jpg", (150, 150, 60), lesion=(120, 80, 40))

    assert group_duplicates(signatures([a, b]), 4, thumbnail_confirmation(8.0)) == []


def test_a_recompressed_copy_is_still_a_copy(tmp_path):
    original = leaf(tmp_path / "Healthy" / "a.jpg", (40, 120, 40))
    copy = tmp_path / "Healthy" / "a_copy.jpg"
    with Image.open(original) as image:
        image.resize((180, 180)).save(copy, quality=70)

    groups = group_duplicates(signatures([original, copy]), 4, thumbnail_confirmation(8.0))

    assert groups == [sorted([original, copy])]


def test_an_unreadable_file_never_confirms(tmp_path):
    good = leaf(tmp_path / "Healthy" / "a.jpg", (40, 120, 40))
    broken = tmp_path / "Healthy" / "b.jpg"
    broken.write_bytes(b"nao e imagem")

    assert not thumbnail_confirmation(8.0)(good, broken)


def test_a_near_identical_shot_of_the_same_leaf_is_a_copy(tmp_path):
    """Duas fotos em sequência da mesma folha: cor quase igual, mesma posição."""
    a = leaf(tmp_path / "x" / "a.jpg", (40, 120, 40))
    b = leaf(tmp_path / "x" / "b.jpg", (42, 118, 41))
    assert thumbnail_confirmation(8.0)(a, b)


def tiny_leaf(path: Path, color, spots=()) -> Path:
    """Folha pequena (~4% do quadro) sobre papel cinza — o caso do limão."""
    image = Image.new("RGB", (400, 300), (190, 190, 186))
    draw = ImageDraw.Draw(image)
    draw.ellipse((180, 130, 230, 175), fill=color)
    for x, y in spots:
        draw.ellipse((x, y, x + 8, y + 8), fill=(200, 170, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)
    return path


def test_the_paper_does_not_decide_for_a_tiny_leaf(tmp_path):
    """Com o quadro inteiro, o papel ocupa 96% e as duas folhas empatariam."""
    a = tiny_leaf(tmp_path / "Healthy" / "a.jpg", (60, 110, 50))
    b = tiny_leaf(tmp_path / "Canker" / "b.jpg", (120, 130, 60), spots=((190, 140), (210, 155)))

    assert not thumbnail_confirmation(8.0)(a, b)


def test_the_same_tiny_leaf_twice_is_a_copy(tmp_path):
    a = tiny_leaf(tmp_path / "Healthy" / "a.jpg", (60, 110, 50))
    b = tmp_path / "Healthy" / "b.jpg"
    with Image.open(a) as image:
        image.save(b, quality=60)

    assert thumbnail_confirmation(8.0)(a, b)
