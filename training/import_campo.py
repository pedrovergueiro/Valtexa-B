"""Fotos do Rover no campo, rotuladas pelo agrônomo, viram dado de treino.

    # 1) na plataforma: a ficha do agrônomo (CSV) e os originais da missão
    #    GET /v1/fields/<talhão>/validacao.csv            -> rotulos.csv
    #    python scripts/exporta_originais.py --mission <id> --dest originais/
    # 2) aqui: os recortes, a partir do registro do Rover na IA (FIELDS_TILE_LOG=true)
    python -m training.import_campo --labels rotulos.csv \\
        --registro fields/plantas/registro/T01.jsonl --originais originais/ \\
        --out data/coffee-campo

O rótulo do agrônomo é **por planta**, e o modelo aprende **por recorte**. Numa planta
com ferrugem, a maior parte dos recortes mostra folha sadia; chamar todos de "Rust"
ensinaria o modelo a ver ferrugem em folha sadia — o erro que mais custa no café (alarme
falso). Então:

* planta **sadia** → todos os recortes dela vão para ``Healthy/``: negativos confiáveis,
  do cafezal de verdade, que os acervos públicos não têm;
* planta **doente** → só os recortes em que o modelo já vê a doença (probabilidade dela ≥
  ``--limiar``) vão para ``_revisar/<classe>/``, com ``_revisar/manifesto.csv``. Alguém
  confirma recorte a recorte e move para ``<classe>/``; só então entram no treino.

Cada recorte sai do original que o Rover processou (lente já corrigida), pela caixa que o
registro guarda. O nome do arquivo diz talhão, passada, planta e quadro — rastreável até a
foto e removível se o cliente retirar o consentimento.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from training.fit_fusion import read_labels

HEALTHY = "Healthy"


def _seguro(texto: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(texto))


def importa(labels: dict, registros: list[Path], originais: Path, out: Path,
            limiar: float = 0.5) -> dict[str, int]:
    contagem: Counter[str] = Counter()
    revisar = out / "_revisar"
    manifesto_linhas = []
    abertas: dict[str, Image.Image | None] = {}
    for caminho in registros:
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            rec = json.loads(linha)
            field_id, pass_id, planta = rec["field_id"], rec.get("pass_id"), str(rec["plant"])
            rotulo = labels.get((field_id, planta, pass_id)) or labels.get((field_id, planta, None))
            if rotulo is None:
                contagem["sem_rotulo"] += 1
                continue
            quadro = str(rec["frame"])
            if quadro not in abertas:
                foto = originais / f"{quadro}.jpg"
                abertas.clear()  # um quadro por vez: os de 12 MP pesam
                abertas[quadro] = Image.open(foto).convert("RGB") if foto.exists() else None
            imagem = abertas[quadro]
            if imagem is None:
                contagem["sem_original"] += 1
                continue
            for i, t in enumerate(rec.get("tiles") or []):
                caixa = tuple(int(round(v)) for v in t["box"])
                nome = (f"campo_{_seguro(field_id)}_{_seguro(pass_id)}_{_seguro(planta)}_"
                        f"{_seguro(quadro)}_{i:03d}.jpg")
                if rotulo == HEALTHY:
                    destino = out / HEALTHY
                    destino.mkdir(parents=True, exist_ok=True)
                    imagem.crop(caixa).save(destino / nome, quality=95)
                    contagem[HEALTHY] += 1
                    continue
                p = float((t.get("probabilities") or {}).get(rotulo, 0.0))
                if p < limiar:
                    contagem["doente_descartado"] += 1
                    continue
                destino = revisar / _seguro(rotulo)
                destino.mkdir(parents=True, exist_ok=True)
                imagem.crop(caixa).save(destino / nome, quality=95)
                manifesto_linhas.append([f"{_seguro(rotulo)}/{nome}", field_id, pass_id, planta,
                                         quadro, rotulo, f"{p:.4f}"])
                contagem[f"revisar:{rotulo}"] += 1
    if manifesto_linhas:
        revisar.mkdir(parents=True, exist_ok=True)
        with (revisar / "manifesto.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["arquivo", "field_id", "pass_id", "plant_id", "frame", "rotulo_planta",
                        "prob_modelo"])
            w.writerows(sorted(manifesto_linhas, key=lambda x: -float(x[-1])))
    return dict(contagem)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Recortes de campo rotulados → dataset de treino.")
    p.add_argument("--labels", type=Path, required=True, help="CSV da validação de campo.")
    p.add_argument("--registro", type=Path, nargs="+", required=True,
                   help="Registro do Rover na IA (fields/plantas/registro/<talhão>.jsonl).")
    p.add_argument("--originais", type=Path, required=True,
                   help="<record_id>.jpg (scripts/exporta_originais.py da plataforma).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limiar", type=float, default=0.5,
                   help="Prob. mínima da doença da planta para o recorte ir à revisão.")
    args = p.parse_args(argv)
    contagem = importa(read_labels(args.labels), args.registro, args.originais, args.out,
                       args.limiar)
    print(json.dumps(contagem, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
