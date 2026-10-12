"""Organiza as folhas do BRACOL em pastas por classe do ``coffee-foliar``.

    python -m training.download_public mendeley yy2k5y8mxg --out ~/valtexa-raw/bracol
    python -m training.import_bracol \\
        --zip ~/valtexa-raw/bracol/yy2k5y8mxg-v1.zip --output data/coffee-bracol

O BRACOL (Krohling, Esgario & Ventura, Mendeley Data, CC BY 4.0) fotografa folhas de
café arábica de Santa Maria de Marechal Floreano, Espírito Santo, **inteiras e sobre
fundo branco**, com vários celulares. É o estilo de foto em que o ``coffee-foliar``
desabava (6,5% de acerto em fotos assim, de outra fazenda brasileira). Não traz
subpastas por classe: o rótulo está no ``leaf/dataset.csv`` (``predominant_stress``).

## O zip do Mendeley vem truncado

Medido em 01/10/2026: o arquivo hospedado (164.516.964 bytes) termina no meio de uma
imagem e não tem diretório central — ``unzip`` e ``zipfile`` recusam. Os tamanhos estão
nos cabeçalhos locais, então ``--zip`` lê as entradas em sequência e recupera as
completas: 1.401 das 1.747 folhas, com o CSV. Com o dataset já extraído, use
``--source`` apontando para a pasta ``leaf``.

## Cópias que já estavam no dataset do café

73 folhas de phoma do BRACOL (2048x1024) **já estavam** em ``data/coffee-foliar`` sob
``phoma__N.jpg``, vindas de algum reempacotamento — e eram as únicas folhas inteiras
sobre fundo branco do dataset. É daí que saía o atalho "folha inteira em fundo branco
= phoma". O ``fit_heads --extra`` descarta do BRACOL o que é parecido demais com o
dataset principal antes de juntar.
"""

from __future__ import annotations

import argparse
import csv
import io
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import Path

STRESS_TO_CLASS = {
    0: "Healthy",
    1: "Leaf_Miner_Damage",
    2: "Rust",
    3: "Phoma_Leaf_Spot",
    4: "Cercospora_Leaf_Spot",
}
"""``predominant_stress`` do ``dataset.csv`` -> classe do catálogo.

A ordem é a das colunas do próprio CSV (``miner, rust, phoma, cercospora``), conferida
nos dados: cada código tem a sua coluna marcada em 100% das linhas. O código 5 (folha
com mais de um estresse e nenhum predominante) fica de fora — encaixá-lo numa classe
seria inventar rótulo.
"""

LEAF_PREFIX = "leaf/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BRACOL -> pastas por classe do coffee-foliar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="O zip baixado do Mendeley (pode estar truncado).")
    source.add_argument("--source", type=Path, help="A pasta 'leaf' já extraída.")
    parser.add_argument("--output", type=Path, required=True, help="Onde criar <Classe>/.")
    return parser.parse_args()


def salvage_zip(data: bytes) -> dict[str, bytes]:
    """As entradas completas de um zip, lidas pelos cabeçalhos locais.

    Para na primeira entrada incompleta. Confere o CRC de cada uma: um arquivo
    recuperado pela metade não passa.
    """
    entries: dict[str, bytes] = {}
    position = 0
    while data[position : position + 4] == b"PK\x03\x04":
        header = struct.unpack("<IHHHHHIIIHH", data[position : position + 30])
        flags, method, crc, compressed = header[2], header[3], header[6], header[7]
        name_size, extra_size = header[9], header[10]
        name = data[position + 30 : position + 30 + name_size].decode("utf-8", "replace")
        start = position + 30 + name_size + extra_size
        end = start + compressed
        if flags & 0x08 or end > len(data):
            break  # tamanho só no descritor, ou entrada cortada: daqui em diante não há como ler
        if not name.endswith("/"):
            raw = data[start:end]
            content = raw if method == 0 else zlib.decompress(raw, -15)
            if zlib.crc32(content) == crc:
                entries[name] = content
        position = end
    return entries


def read_zip(data: bytes) -> dict[str, bytes]:
    """Pelo diretório central quando ele existe; senão, pelos cabeçalhos locais."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    except zipfile.BadZipFile:
        return salvage_zip(data)


def leaf_files_from_zip(path: Path) -> dict[str, bytes]:
    """``leaf/...`` -> bytes, abrindo o zip interno do BRACOL se ele vier embrulhado."""
    entries = read_zip(path.read_bytes())
    inner = [name for name in entries if name.lower().endswith(".zip")]
    for name in inner:
        entries.update(read_zip(entries.pop(name)))
    leaf = {}
    for name, content in entries.items():
        marker = name.find(LEAF_PREFIX)
        if marker >= 0:
            leaf[name[marker + len(LEAF_PREFIX) :]] = content
    return leaf


def main() -> None:
    args = parse_args()
    if args.zip is not None:
        files = leaf_files_from_zip(args.zip)
        table = files.get("dataset.csv")
        images = {
            name.split("/")[-1]: content
            for name, content in files.items()
            if name.startswith("images/")
        }
    else:
        table = (args.source / "dataset.csv").read_bytes()
        images = {p.name: p.read_bytes() for p in (args.source / "images").iterdir() if p.is_file()}
    if table is None:
        raise SystemExit("dataset.csv não encontrado no BRACOL.")

    counts: Counter[str] = Counter()
    skipped = Counter()
    for row in csv.DictReader(io.StringIO(table.decode("utf-8"))):
        stress = int(row["predominant_stress"])
        content = images.get(f"{row['id']}.jpg")
        if content is None:
            skipped["sem imagem (zip truncado)"] += 1
            continue
        if stress not in STRESS_TO_CLASS:
            skipped["sem estresse predominante"] += 1
            continue
        target = args.output / STRESS_TO_CLASS[stress] / f"bracol_{row['id']}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        counts[STRESS_TO_CLASS[stress]] += 1

    print(f"BRACOL -> {args.output}")
    for name, count in sorted(counts.items()):
        print(f"  {name:22s} {count}")
    for reason, count in skipped.items():
        print(f"  fora: {count} {reason}")


if __name__ == "__main__":
    main()
