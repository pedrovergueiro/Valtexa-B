"""Organiza o recorte de milho e café do PlantSeg em pastas por classe.

    python -m training.import_plantseg \\
        --source ~/valtexa-raw/plantseg \\
        --output ~/valtexa-raw/_plantseg-classes

O PlantSeg entrega as imagens num diretório plano, com a espécie e a doença
codificadas no **nome do arquivo** (``corn_northern_leaf_blight_google_0175.jpg``),
não em subpastas — então o ``prepare_dataset``, que lê pastas, não o enxerga.

Ele vale o script porque é o único dos datasets baixados capturado
*in-the-wild*: fundo complexo, luz variável, oclusão. É o contraponto ao
PlantVillage, cujas fotos de folha destacada sobre fundo uniforme produzem
modelos com 99% de acurácia que desabam em lavoura.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from training.prepare_dataset import is_image, looks_like_label_mask

PREFIX_TO_CLASS = {
    # Milho -> leaf-foliar
    "corn_northern_leaf_blight": "Blight",
    "corn_gray_leaf_spot": "Gray_Leaf_Spot",
    "corn_rust": "Common_Rust",
    "corn_healthy": "Healthy",
    # Café -> coffee-foliar
    "coffee_leaf_rust": "Rust",
    "coffee_cercospora_leaf_spot": "Cercospora_Leaf_Spot",
    "coffee_phoma_leaf_spot": "Phoma_Leaf_Spot",
    "coffee_leaf_miner": "Leaf_Miner_Damage",
    "coffee_healthy": "Healthy",
}
"""Prefixos de nome de arquivo que correspondem a uma classe do catálogo.

Ficam de fora os prefixos sem equivalente — ``corn_smut`` (carvão, que ataca a
espiga e não a folha) e ``coffee_berry_blotch`` (fruto, não folha). Encaixá-los
numa classe existente seria inventar rótulo.

A ordem importa na hora de casar: os prefixos são testados do mais longo para o
mais curto, para que ``corn_gray_leaf_spot`` não seja capturado por um prefixo
mais genérico que porventura seja acrescentado aqui depois.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorte de milho e café do PlantSeg -> pastas por classe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="Raiz do PlantSeg extraído.")
    parser.add_argument("--output", type=Path, required=True, help="Onde criar <Classe>/.")
    parser.add_argument("--move", action="store_true", help="Move em vez de copiar.")
    parser.add_argument("--dry-run", action="store_true", help="Só mostra o plano.")
    return parser.parse_args()


def classify(filename: str) -> str | None:
    """Classe canônica de uma imagem do PlantSeg, pelo nome do arquivo."""
    name = filename.lower()
    for prefix in sorted(PREFIX_TO_CLASS, key=len, reverse=True):
        if name.startswith(prefix):
            return PREFIX_TO_CLASS[prefix]
    return None


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Origem não encontrada: {args.source}")

    plan: list[tuple[Path, str]] = []
    ignorados: Counter[str] = Counter()

    for image in sorted(args.source.rglob("*")):
        if not is_image(image):
            continue
        # O PlantSeg publica a máscara de cada foto com o mesmo nome, em .png.
        # Foi assim que 388 máscaras viraram "fotos" de doença no leaf-foliar.
        if looks_like_label_mask(image):
            ignorados["máscaras de segmentação"] += 1
            continue
        canonical = classify(image.name)
        if canonical is None:
            # Agrupa pelo prefixo até o terceiro "_", só para o relatório mostrar
            # quais espécies foram deixadas de lado em vez de um total opaco.
            ignorados["_".join(image.name.split("_")[:3])] += 1
            continue
        plan.append((image, canonical))

    counts = Counter(canonical for _, canonical in plan)
    print("\n=== PlantSeg: recorte de milho e café ===")
    print(f"Origem: {args.source}\n")
    print(f"{len(plan)} imagens aproveitáveis:")
    for canonical, count in counts.most_common():
        print(f"    {count:>6}  {canonical}")

    print(f"\n{sum(ignorados.values())} imagens de outras culturas/doenças ignoradas.")
    for prefix, count in ignorados.most_common(8):
        print(f"    {count:>6}  {prefix}")

    if not plan:
        raise SystemExit(
            "Nenhuma imagem reconhecida — confira se apontou para a raiz do "
            "PlantSeg extraído e veja PREFIX_TO_CLASS neste módulo."
        )

    if args.dry_run:
        print("\n--dry-run: nada foi escrito.")
        return

    transfer = shutil.move if args.move else shutil.copy2
    for image, canonical in plan:
        destination = args.output / canonical
        destination.mkdir(parents=True, exist_ok=True)
        transfer(str(image), str(destination / f"plantseg__{image.name}"))

    print(f"\nPronto em {args.output}")


if __name__ == "__main__":
    main()
