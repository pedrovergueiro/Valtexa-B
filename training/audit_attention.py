"""Audita onde um modelo treinado olha, classe por classe.

Rode depois de treinar, antes de pôr os pesos em produção::

    python -m training.audit_attention --model leaf-foliar --per-class 25

Sorteia imagens de cada classe do dataset, gera o mapa de oclusão de cada uma
(ver :mod:`app.inference.explain`) e resume por classe o quanto a decisão
depende de **uma região** da imagem.

## O que ele pega que a acurácia não pega

O ``rover-understory`` marcou 1,0000 de acurácia e está em quarentena: aprendeu
a sessão de fotografia, não a doença — uma regressão logística que lê só a cor
média da imagem também acerta 100%. Nenhuma métrica de acerto denuncia isso,
porque o atalho acerta.

O mapa denuncia. Um modelo que lê lesão perde confiança quando a lesão é
coberta; um que lê a luz ou a cor da foto quase não perde em lugar nenhum,
porque cobrir 1/49 da imagem mal mexe na cor média dela. Numa classe de doença,
``max_drop`` perto de zero na maioria das imagens é o sinal de alarme.

Para as classes sadias a leitura é outra: a evidência de "sadio" é a ausência de
lesão, espalhada pela folha inteira, e cobrir um pedaço não a remove. Queda
baixa ali é o esperado.

## O segundo teste: só a cor

O mapa pode ser enganado por um atalho que também é local — um fundo de pano
azul atrás de só uma das classes, por exemplo. Por isso cada imagem passa
também por um teste mais bruto: ela é trocada por um quadro chapado com a **cor
média dela**, sem nenhuma lesão, forma ou textura. Um modelo que lê doença não
tem o que ler ali; um que lê a sessão de fotografia tem exatamente o que usava.
"Só a cor" conta as imagens em que o modelo mantém a resposta, com confiança
alta, olhando só para isso.

As sobreposições gravadas em ``--out`` são o que se mostra para um agrônomo:
ele diz em segundos se o vermelho está na lesão, no fundo ou na mão de quem
segurava a folha.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.inference.explain import DEFAULT_GRID, RISE_SAMPLES
from app.inference.preprocess import load_image_from_bytes
from app.model import CornDiseaseModel
from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED
from training.prepare_dataset import is_image

LOW_EVIDENCE = 0.05
"""Queda máxima abaixo da qual nenhuma região sozinha sustenta a resposta.

Não é limiar medido: é o ponto em que cobrir o pedaço mais importante da imagem
mexe menos de 5 pontos na confiança. Serve para contar, não para decidir — o
relatório mostra a mediana ao lado para quem quiser outra régua.
"""

COLOR_ONLY_CONFIDENCE = 0.9
"""Confiança a partir da qual a resposta dada só pela cor conta como atalho.

Abaixo disso, acertar a classe olhando um quadro chapado pode ser sorte — num
modelo de duas classes, é cara ou coroa.
"""


@dataclass(frozen=True)
class AuditRow:
    """O resultado de uma imagem."""

    path: Path
    true_class: str
    predicted_class: str
    base_probability: float
    max_drop: float
    color_only_class: str | None = None
    """O que o modelo responde para o quadro chapado com a cor média da imagem."""

    color_only_confidence: float = 0.0

    @property
    def correct(self) -> bool:
        return self.true_class == self.predicted_class

    @property
    def decided_by_color(self) -> bool:
        """Mesma resposta, com confiança alta, sem nada além da cor média."""
        return (
            self.color_only_class == self.predicted_class
            and self.color_only_confidence >= COLOR_ONLY_CONFIDENCE
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita onde um modelo treinado olha para decidir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Pesos a auditar, ex.: os de models/_quarentena/. Padrão: os do catálogo.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
    parser.add_argument(
        "--method",
        choices=("rise", "occlusion"),
        default="rise",
        help="rise enxerga lesões espalhadas; a oclusão de uma célula não (ver explain.py).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Onde gravar as sobreposições, uma pasta por classe. Vazio: não grava.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    if args.weights is not None:
        # Caminho absoluto em ``weights_name`` faz o ``models_dir / nome`` do
        # catálogo devolver o próprio caminho — os rótulos, a novidade e a
        # qualidade passam a ser procurados ao lado destes pesos.
        spec = dataclasses.replace(spec, weights_name=str(args.weights.resolve()))
    data_dir = args.data_dir or spec.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Diretório de dados não encontrado: {data_dir}")

    model = CornDiseaseModel(spec)
    model.load()
    if not model.is_loaded:
        raise SystemExit(f"Não foi possível carregar os pesos de {spec.weights_path}.")

    items = sample_per_class(data_dir, args.per_class, args.seed)
    print(f"\n=== Auditoria de atenção — {spec.title} ({spec.key}) ===")
    print(f"Pesos: {spec.weights_path}")
    passes = RISE_SAMPLES + 1 if args.method == "rise" else args.grid**2 + 1
    print(f"{len(items)} imagens, {args.method} em grade {args.grid}x{args.grid}")
    print(f"({passes} passadas por imagem)\n")

    rows = audit(model, items, grid=args.grid, out_dir=args.out, method=args.method)
    print_summary(rows)
    if args.out is not None:
        print(f"\nSobreposições em: {args.out}")


def sample_per_class(data_dir: Path, per_class: int, seed: int) -> list[tuple[Path, str]]:
    """Sorteia até ``per_class`` imagens de cada subpasta, sempre as mesmas por semente."""
    rng = np.random.default_rng(seed)
    items = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = sorted(path for path in folder.iterdir() if is_image(path))
        if len(files) > per_class:
            files = [files[i] for i in sorted(rng.choice(len(files), per_class, replace=False))]
        items.extend((path, folder.name) for path in files)
    return items


def audit(
    model: CornDiseaseModel,
    items: list[tuple[Path, str]],
    grid: int = DEFAULT_GRID,
    out_dir: Path | None = None,
    method: str = "rise",
) -> list[AuditRow]:
    """Explica cada imagem pela classe prevista e grava a sobreposição, se pedido.

    Explica a classe **prevista**, não a verdadeira: a pergunta é o que moveu a
    decisão, inclusive quando ela está errada.
    """
    rows = []
    for count, (path, true_class) in enumerate(items, start=1):
        data = path.read_bytes()
        explanation = model.explain(
            path.name, data, grid=grid, overlay=out_dir is not None, method=method
        )
        color_only = model.predict(path.name, mean_color_image(data, path.name), tta=False)
        rows.append(
            AuditRow(
                path=path,
                true_class=true_class,
                predicted_class=explanation.target_class,
                base_probability=explanation.base_probability,
                max_drop=explanation.max_drop,
                color_only_class=color_only.predicted_class,
                color_only_confidence=color_only.confidence,
            )
        )
        if out_dir is not None and explanation.overlay_png:
            folder = out_dir / true_class
            folder.mkdir(parents=True, exist_ok=True)
            verdict = "ok" if rows[-1].correct else f"errou-{explanation.target_class}"
            (folder / f"{path.stem}.{verdict}.png").write_bytes(
                base64.b64decode(explanation.overlay_png)
            )
        print(f"\r  {count}/{len(items)} imagens explicadas...", end="", flush=True)
    print()
    return rows


def mean_color_image(data: bytes, filename: str = "imagem") -> bytes:
    """A imagem trocada por um quadro chapado da cor média dela, em PNG."""
    image = load_image_from_bytes(data, filename)
    mean = np.asarray(image, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    flat = Image.new("RGB", image.size, tuple(int(round(v)) for v in mean))
    buffer = io.BytesIO()
    flat.save(buffer, format="PNG")
    return buffer.getvalue()


def summarize(rows: list[AuditRow], low_evidence: float = LOW_EVIDENCE) -> list[dict]:
    """Uma linha por classe verdadeira: acerto, queda mediana, sem região e só a cor."""
    summary = []
    for class_name in sorted({row.true_class for row in rows}):
        mine = [row for row in rows if row.true_class == class_name]
        drops = [row.max_drop for row in mine]
        summary.append(
            {
                "class_name": class_name,
                "images": len(mine),
                "accuracy": sum(row.correct for row in mine) / len(mine),
                "median_max_drop": statistics.median(drops),
                "low_evidence": sum(drop < low_evidence for drop in drops) / len(mine),
                "color_only": sum(row.decided_by_color for row in mine) / len(mine),
            }
        )
    return summary


def print_summary(rows: list[AuditRow]) -> None:
    print(
        f"\n{'classe':24s} {'imgs':>5s} {'acerto':>7s} {'queda mediana':>14s} "
        f"{'sem região':>11s} {'só a cor':>9s}"
    )
    for line in summarize(rows):
        print(
            f"{line['class_name']:24s} {line['images']:5d} {line['accuracy']:7.1%} "
            f"{line['median_max_drop']:14.3f} {line['low_evidence']:11.1%} "
            f"{line['color_only']:9.1%}"
        )
    print(
        f"\n'sem região' = imagens em que cobrir o pedaço mais decisivo mexe menos de "
        f"{LOW_EVIDENCE:.0%} na confiança.\nNuma classe de doença, muitas assim "
        "indicam que o modelo lê a foto inteira (luz, cor, fundo), não a lesão.\n"
        "Nas classes sadias é o esperado: a evidência de sadio está espalhada.\n"
        f"'só a cor' = mesma resposta, com confiança >= {COLOR_ONLY_CONFIDENCE:.0%}, "
        "para um quadro chapado da cor média.\nAlto em qualquer classe é atalho: "
        "não há lesão num quadro chapado."
    )


if __name__ == "__main__":
    main()
