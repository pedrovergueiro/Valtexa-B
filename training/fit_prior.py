"""Grava a prevalência das classes no treino, ao lado dos pesos.

    python -m training.fit_prior --model banana-foliar --data-dir data/banana-foliar-campo

A prevalência por talhão (:mod:`app.fields_api`) corrige o prior artificial dos
datasets públicos — mas para corrigir precisa saber qual era ele. É este
arquivo: a fração de cada classe nas imagens com que o modelo foi treinado.

Conta o dataset inteiro, não só a parte de treino: o *split* é um sorteio
uniforme, então as proporções são as mesmas a menos de ruído amostral.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.registry import MODEL_SPECS, get_spec
from training.prepare_dataset import is_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prevalência das classes no treino")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    return parser.parse_args()


def class_prior(data_dir: Path, class_names: tuple[str, ...]) -> dict[str, float]:
    """Fração de imagens de cada classe; classe sem pasta vira erro, não zero."""
    counts = {}
    for name in class_names:
        folder = data_dir / name
        if not folder.is_dir():
            raise SystemExit(f"Classe {name!r} sem pasta em {data_dir}.")
        counts[name] = sum(1 for path in folder.iterdir() if is_image(path))
    total = sum(counts.values())
    if total == 0:
        raise SystemExit(f"Nenhuma imagem em {data_dir}.")
    return {name: count / total for name, count in counts.items()}


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    data_dir = args.data_dir or spec.data_dir
    prior = class_prior(data_dir, spec.class_names)
    output = spec.weights_path.with_suffix(".prior.json")
    output.write_text(
        json.dumps({"data_dir": str(data_dir), "prior": prior}, indent=2), encoding="utf-8"
    )
    print(f"{spec.key}: " + ", ".join(f"{k} {v:.1%}" for k, v in prior.items()))
    print(f"Gravado em {output}")


if __name__ == "__main__":
    main()
