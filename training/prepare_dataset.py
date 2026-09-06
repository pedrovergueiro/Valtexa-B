"""Organiza datasets públicos no layout que o treino espera.

Cada dataset nomeia as suas pastas de um jeito — ``Corn_(maize)___Common_rust_``
no PlantVillage, ``Corn rust leaf`` no PlantDoc, ``Common_Rust`` no dataset do
Kaggle. Este script traduz todos para as classes canônicas do catálogo e
consolida em ``data/<modelo>/<Classe>/``.

Uso::

    python -m training.prepare_dataset --model leaf-foliar \\
        --source "C:/datasets/PlantVillage" \\
        --source "C:/datasets/corn-or-maize-leaf-disease"

Pastas cujo nome não estiver no mapa são **reportadas, nunca adivinhadas**: um
palpite errado aqui contamina o dataset inteiro em silêncio e só aparece como
acurácia baixa muito depois.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from app.registry import (
    MODEL_SPECS,
    SOURCE_CLASS_ALIASES,
    ModelSpec,
    get_spec,
    normalize_class_name,
)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida datasets públicos no layout de treino",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Raiz de um dataset extraído. Pode repetir para consolidar vários.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move em vez de copiar. Mais rápido, mas consome a origem.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Só mostra o plano, sem escrever nada."
    )
    return parser.parse_args()


def find_class_dirs(root: Path) -> list[Path]:
    """Lista os diretórios que contêm imagens diretamente.

    Percorre a árvore inteira porque cada dataset aninha de um jeito — alguns
    trazem um nível ``train/``/``test/``, outros não.
    """
    class_dirs = [
        directory for directory in root.rglob("*") if directory.is_dir() and _has_images(directory)
    ]
    return sorted(class_dirs)


def plan_dataset(sources: list[Path], spec: ModelSpec) -> tuple[dict[str, list[Path]], list[Path]]:
    """Decide o destino de cada pasta de origem.

    Returns:
        O plano ``{classe canônica: [pastas de origem]}`` e a lista de pastas
        que não puderam ser mapeadas para uma classe deste modelo.
    """
    plan: dict[str, list[Path]] = defaultdict(list)
    unmapped: list[Path] = []
    valid_classes = set(spec.class_names)

    for source in sources:
        for directory in find_class_dirs(source):
            canonical = SOURCE_CLASS_ALIASES.get(normalize_class_name(directory.name))
            if canonical is None or canonical not in valid_classes:
                unmapped.append(directory)
                continue
            plan[canonical].append(directory)

    return dict(plan), unmapped


def execute_plan(
    plan: dict[str, list[Path]], output: Path, *, move: bool = False
) -> dict[str, int]:
    """Copia (ou move) as imagens para ``output/<Classe>/``.

    Os arquivos são renomeados com o nome da pasta de origem como prefixo, para
    que dois datasets com ``001.jpg`` não se sobrescrevam.
    """
    counts: dict[str, int] = {}
    transfer = shutil.move if move else shutil.copy2

    for canonical, directories in sorted(plan.items()):
        destination = output / canonical
        destination.mkdir(parents=True, exist_ok=True)
        total = 0

        for directory in directories:
            prefix = _slug(directory.name)
            for image in sorted(directory.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
                    continue
                transfer(str(image), str(destination / f"{prefix}__{image.name}"))
                total += 1

        counts[canonical] = total

    return counts


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    output = args.output or spec.data_dir

    missing = [source for source in args.source if not source.exists()]
    if missing:
        raise SystemExit("Origem não encontrada: " + ", ".join(str(path) for path in missing))

    plan, unmapped = plan_dataset(args.source, spec)

    print(f"\n=== {spec.title} ({spec.key}) ===")
    print(f"Destino: {output}\n")

    if not plan:
        raise SystemExit(
            "Nenhuma pasta reconhecida. Confira se apontou para a raiz extraída "
            "do dataset e veja SOURCE_CLASS_ALIASES em app/registry.py."
        )

    for canonical, directories in sorted(plan.items()):
        print(f"{canonical}:")
        for directory in directories:
            print(f"    {_count_images(directory):>6} imagens  <- {directory}")

    if unmapped:
        print("\nPastas ignoradas (sem mapeamento para este modelo):")
        for directory in unmapped:
            print(f"    {directory.name}")
        print(
            "\nSe alguma delas deveria virar uma classe, acrescente o nome em "
            "SOURCE_CLASS_ALIASES (app/registry.py) e rode de novo."
        )

    faltando = sorted(set(spec.class_names) - set(plan))
    if faltando:
        print(f"\nClasses sem nenhuma imagem: {', '.join(faltando)}")
        print("Elas exigem coleta própria em campo — ver MODELS.md.")

    if args.dry_run:
        print("\n--dry-run: nada foi escrito.")
        return

    counts = execute_plan(plan, output, move=args.move)
    total = sum(counts.values())
    print(f"\n{total} imagens organizadas:")
    for canonical, count in sorted(counts.items()):
        print(f"    {count:>6}  {canonical}")
    print(f"\nPronto. Treine com: python -m training.train --model {spec.key}")


def _has_images(directory: Path) -> bool:
    return any(
        item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in directory.iterdir()
    )


def _count_images(directory: Path) -> int:
    return sum(
        1
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def _slug(name: str) -> str:
    return normalize_class_name(name).replace(" ", "-") or "src"


if __name__ == "__main__":
    main()
