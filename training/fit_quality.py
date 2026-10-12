"""Calibra a verificação de qualidade da captura de um modelo.

Rode uma vez por modelo, a qualquer momento — não depende dos pesos::

    python -m training.fit_quality --model leaf-foliar

Mede nitidez, brilho, contraste e pixels estourados de cada imagem do dataset,
**na resolução de entrada do modelo**, e grava os limites em
``models/<modelo>.quality.json``. A API carrega esse arquivo no próximo start e
passa a devolver ``quality`` em cada previsão.

Uma parte do dataset fica de fora da calibração e serve para medir, antes de
gravar, as duas coisas que decidem se a verificação presta: quanto ela marca de
foto boa (alarme falso) e quanto ela pega de foto estragada de propósito.

Não importa o Keras nem o TensorFlow: roda no mesmo ambiente da API.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.inference.preprocess import image_to_array, resize_like_tensorflow
from app.inference.quality import (
    BLURRY,
    DARK,
    DEFAULT_PERCENTILE,
    LOW_CONTRAST,
    OVEREXPOSED,
    CaptureMetrics,
    QualityReference,
    measure,
)
from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED
from training.prepare_dataset import is_image, looks_like_label_mask

ISSUES = (BLURRY, DARK, OVEREXPOSED, LOW_CONTRAST)

DEGRADATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "desfoque sigma 1,5 px": lambda image: image.filter(ImageFilter.GaussianBlur(1.5)),
    "desfoque sigma 2 px": lambda image: image.filter(ImageFilter.GaussianBlur(2.0)),
    "brilho x0,3": lambda image: ImageEnhance.Brightness(image).enhance(0.3),
    "brilho x2,2": lambda image: ImageEnhance.Brightness(image).enhance(2.2),
    "contraste x0,2": lambda image: ImageEnhance.Contrast(image).enhance(0.2),
}
"""Estragos aplicados à imagem **já redimensionada** para a entrada do modelo.

Aplicar depois de reduzir é o que torna o número interpretável: "sigma 2 px" é
desfoque de 2 pixels *no que o modelo vê*, seja a foto original de 128 px ou de
4000 px.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra a verificação de qualidade da captura de um modelo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="A referência é gravada ao lado destes pesos. Padrão: os do catálogo.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Cauda do treino, em cada ponta, que vira limiar.",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="Fração separada para medir alarme falso e detecção antes de gravar.",
    )
    parser.add_argument("--max-images", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    data_dir = args.data_dir or spec.data_dir
    weights = args.weights or spec.weights_path
    output = (
        spec.quality_path if weights == spec.weights_path else weights.with_suffix(".quality.json")
    )
    if not data_dir.is_dir():
        raise SystemExit(f"Diretório de dados não encontrado: {data_dir}")
    if not 0.0 < args.holdout < 0.5:
        raise SystemExit("--holdout precisa ficar entre 0 e 0,5.")

    print(f"\n=== Calibrando qualidade da captura — {spec.title} ({spec.key}) ===")
    print(f"Dados:     {data_dir}")
    print(f"Resolução: {spec.img_size}x{spec.img_size} (a entrada do modelo)\n")

    items = sample_images(data_dir, args.max_images, args.seed)
    if not items:
        raise SystemExit("Nenhuma imagem encontrada — verifique o diretório de dados.")

    order = np.random.default_rng(args.seed).permutation(len(items))
    n_holdout = int(len(items) * args.holdout)
    holdout = [items[i] for i in order[:n_holdout]]
    calibration = [items[i] for i in order[n_holdout:]]

    reference = QualityReference.fit(
        measure_files([path for path, _ in calibration], spec.img_size),
        spec.img_size,
        percentile=args.percentile,
    )
    report(reference, holdout, spec.img_size)

    reference.save(output)
    print(f"\nReferência salva em: {output}")
    print("A API passa a devolver 'quality' no próximo start.")


def sample_images(data_dir: Path, max_images: int, seed: int) -> list[tuple[Path, str]]:
    """Lista ``(arquivo, classe)``, sorteando um subconjunto acima do teto.

    O sorteio é uniforme sobre o dataset inteiro, então a proporção entre as
    classes é preservada — e é ela que define o que é uma foto "normal".
    """
    items = [
        (path, folder.name)
        for folder in sorted(p for p in data_dir.iterdir() if p.is_dir())
        for path in sorted(folder.iterdir())
        # Máscara de segmentação não é foto: as 388 do PlantSeg no milho eram as
        # "fotos quase pretas" que puxavam o limite de brilho para 3,7 de 255.
        if is_image(path) and not looks_like_label_mask(path)
    ]
    if len(items) > max_images:
        chosen = np.random.default_rng(seed).choice(len(items), max_images, replace=False)
        items = [items[i] for i in sorted(chosen)]
    return items


def measure_files(paths: list[Path], img_size: int) -> list[CaptureMetrics]:
    """Mede cada arquivo, pulando os ilegíveis em vez de abortar a calibração."""
    metrics = []
    for count, path in enumerate(paths, start=1):
        try:
            metrics.append(_measure_image(_resized(path, img_size)))
        except (OSError, ValueError) as exc:
            print(f"\n  ignorada {path.name}: {exc}")
        if count % 500 == 0:
            print(f"\r  {count} imagens medidas...", end="", flush=True)
    print(f"\r  {len(metrics)} imagens medidas.      ")
    return metrics


def report(reference: QualityReference, holdout: list[tuple[Path, str]], img_size: int) -> None:
    """Mede, no que ficou fora da calibração, o custo e o alcance da verificação."""
    print(
        f"\nLimites (percentil {reference.percentile:g} de {reference.fitted_on} imagens):\n"
        f"  nitidez mínima    {reference.min_sharpness:8.1f}\n"
        f"  brilho            {reference.min_brightness:8.1f} a {reference.max_brightness:.1f}\n"
        f"  contraste mínimo  {reference.min_contrast:8.1f}\n"
        f"  estourados máx.   {reference.max_highlights:8.2%}"
    )

    resized = []
    classes = []
    for path, class_name in holdout:
        try:
            resized.append(_resized(path, img_size))
            classes.append(class_name)
        except (OSError, ValueError):
            continue

    issues = [reference.assess(_measure_image(image)) for image in resized]
    print(f"\nAlarme falso em {len(resized)} imagens boas, fora da calibração:")
    for class_name in sorted(set(classes)):
        mine = [found for found, c in zip(issues, classes, strict=True) if c == class_name]
        print(f"  {class_name:24s} {_rate(mine):>6}   {_breakdown(mine)}")
    print(f"  {'total':24s} {_rate(issues):>6}")

    print("\nDetecção nas mesmas imagens, estragadas de propósito:")
    for name, degrade in DEGRADATIONS.items():
        found = [reference.assess(_measure_image(degrade(image))) for image in resized]
        print(f"  {name:24s} {_rate(found):>6}   {_breakdown(found)}")


def _resized(path: Path, img_size: int) -> Image.Image:
    """A imagem como o modelo a recebe: pixels do arquivo, redimensionados como no treino."""
    with Image.open(path) as image:
        pixels = resize_like_tensorflow(np.asarray(image.convert("RGB")), img_size)
    return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))


def _measure_image(image: Image.Image) -> CaptureMetrics:
    return measure(image_to_array(image, image.width)[None])[0]


def _rate(issues: list[list[str]]) -> str:
    if not issues:
        return "—"
    return f"{np.mean([bool(found) for found in issues]):.1%}"


def _breakdown(issues: list[list[str]]) -> str:
    if not issues:
        return ""
    parts = []
    for issue in ISSUES:
        share = np.mean([issue in found for found in issues])
        if share > 0:
            parts.append(f"{issue} {share:.1%}")
    return ", ".join(parts)


if __name__ == "__main__":
    main()
