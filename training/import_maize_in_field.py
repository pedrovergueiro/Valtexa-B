"""Converte o Maize_in_Field (rótulo em CSV) para pastas por classe.

    python -m training.import_maize_in_field \\
        --source ~/valtexa-raw/"maize-in-field/Kaggle Dataset" \\
        --output ~/valtexa-raw/_maize-in-field-por-classe

Depois é só apontar o ``prepare_dataset`` para a saída, como em qualquer outro
dataset.

Por que ele merece um script só seu: é o único dataset de milho do catálogo
capturado **em campo, em lavoura real**, com anotação de fitopatologista — o
antídoto para o viés de laboratório do PlantVillage, onde a folha aparece
destacada sobre fundo uniforme. O preço é que o rótulo vem num CSV de colunas
binárias, uma por doença, e não em pastas.

**Uma imagem pode ter mais de uma doença ao mesmo tempo**, e é o caso de cerca
de 38% delas. Um classificador ``softmax`` responde uma classe só, então essas
ficam de fora: rotular uma folha com ferrugem *e* mancha como se tivesse apenas
uma ensina o modelo a ignorar a que foi descartada. Quando o projeto tiver um
modelo multirrótulo, elas voltam a valer — o descarte é uma limitação do
classificador atual, não do dado.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

COLUMN_TO_CLASS = {
    "GLS": "Gray_Leaf_Spot",  # Gray Leaf Spot (Cercospora zeae-maydis)
    "NCLB": "Blight",  # Northern Corn Leaf Blight (Exserohilum turcicum)
    "CR": "Common_Rust",  # Common Rust (Puccinia sorghi)
    "NoFoliarSymptoms": "Healthy",
}
"""Colunas do CSV que correspondem a uma classe do ``leaf-foliar``.

Ficam de fora, de propósito, as colunas sem classe equivalente no catálogo:
``PLS`` (Phaeosphaeria Leaf Spot), ``SR`` (Southern Rust, outro patógeno que a
ferrugem comum), ``Other`` e ``UnidentifiedDisease``. Mapear qualquer uma delas
para uma classe existente seria inventar rótulo.
"""

IMAGE_DIRNAME = "leaf_images"
CSV_NAME = "Database.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Maize_in_Field (CSV multirrótulo) -> pastas por classe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, required=True, help=f"Pasta com {CSV_NAME} e {IMAGE_DIRNAME}/."
    )
    parser.add_argument("--output", type=Path, required=True, help="Onde criar <Classe>/.")
    parser.add_argument("--move", action="store_true", help="Move em vez de copiar.")
    parser.add_argument("--dry-run", action="store_true", help="Só mostra o plano.")
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    # utf-8-sig porque o arquivo vem com BOM, e o BOM gruda no nome da primeira
    # coluna — o que faria a busca por "imgID_id" falhar sem erro nenhum.
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def single_label(row: dict[str, str], label_columns: list[str]) -> str | None:
    """A classe da imagem, ou ``None`` se o rótulo for ambíguo.

    Devolve ``None`` tanto para a imagem com duas doenças quanto para a que só
    tem colunas fora do catálogo — nos dois casos não há uma resposta única e
    correta para um ``softmax``.
    """
    positives = [column for column in label_columns if row.get(column, "").strip() == "1"]
    if len(positives) != 1:
        return None
    return COLUMN_TO_CLASS.get(positives[0])


def plan_rows(
    rows: list[dict[str, str]], images_dir: Path, label_columns: list[str]
) -> tuple[list[tuple[Path, str]], Counter]:
    """Decide o destino de cada linha do CSV.

    Returns:
        As imagens aproveitáveis com a classe de cada uma, e a contagem dos
        motivos de descarte — que é o que revela se o dataset encolheu por
        ambiguidade de rótulo ou por arquivo faltando.
    """
    plan: list[tuple[Path, str]] = []
    descartes: Counter = Counter()

    for row in rows:
        name = (row.get("filePath") or "").strip()
        if not name:
            descartes["linha sem arquivo"] += 1
            continue

        # O CSV às vezes traz o caminho completo do site original; só o nome importa.
        image = images_dir / Path(name.replace("\\", "/")).name
        if not image.exists():
            descartes["arquivo ausente"] += 1
            continue

        canonical = single_label(row, label_columns)
        if canonical is None:
            positives = [c for c in label_columns if row.get(c, "").strip() == "1"]
            descartes["multirrótulo" if len(positives) > 1 else "classe fora do catálogo"] += 1
            continue

        plan.append((image, canonical))

    return plan, descartes


def main() -> None:
    args = parse_args()
    csv_path = args.source / CSV_NAME
    images_dir = args.source / IMAGE_DIRNAME

    for path in (csv_path, images_dir):
        if not path.exists():
            raise SystemExit(f"Não encontrado: {path}")

    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit(f"{csv_path} está vazio.")

    label_columns = [c for c in rows[0] if c not in ("imgID_id", "filePath")]
    print("\n=== Maize_in_Field ===")
    print(f"Origem: {args.source}")
    print(f"{len(rows)} linhas, colunas de rótulo: {', '.join(label_columns)}\n")

    plan, descartes = plan_rows(rows, images_dir, label_columns)
    counts = Counter(canonical for _, canonical in plan)
    print(f"{len(plan)} imagens aproveitáveis:")
    for canonical, count in counts.most_common():
        print(f"    {count:>6}  {canonical}")

    print("\nDescartadas:")
    for motivo, count in descartes.most_common():
        print(f"    {count:>6}  {motivo}")

    if args.dry_run:
        print("\n--dry-run: nada foi escrito.")
        return

    transfer = shutil.move if args.move else shutil.copy2
    for image, canonical in plan:
        destination = args.output / canonical
        destination.mkdir(parents=True, exist_ok=True)
        transfer(str(image), str(destination / f"maizeinfield__{image.name}"))

    print(f"\nPronto em {args.output}")
    print(
        "Consolide com: python -m training.prepare_dataset --model leaf-foliar "
        f"--source {args.output}"
    )


if __name__ == "__main__":
    main()
