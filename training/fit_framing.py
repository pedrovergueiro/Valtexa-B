"""Calibra o portão de enquadramento (visão-linguagem, zero-shot).

Rode uma vez; vale para todos os modelos foliares::

    python -m training.fit_framing \\
        --negatives ~/yt/frames/cafe_aereo ~/yt/frames/milho_folha data/rover-understory \\
        --positives ~/yt/frames/cafe_folha

O que ele faz, em ordem:

1. calcula os embeddings de texto das cenas de :data:`app.framing.SCENES` — a
   torre de texto não precisa ir para produção;
2. calibra o limiar só com **fotos boas**: a validação de cada modelo foliar, no
   percentil ``--percentile`` (1 = aceitar recusar 1% das folhas boas);
3. mede no **teste** de cada modelo quanto de folha boa é recusado, e em cada
   ``--negatives`` quanto de fora do escopo é pego. Os negativos nunca tocam o
   limiar;
4. grava ``models/framing.npz``.

Precisa de ``transformers`` e de rede na primeira vez, para baixar o modelo.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.framing import (
    CROP_MISMATCH_RATIO,
    DEFAULT_MODEL_ID,
    REFERENCE_NAME,
    SCENES,
    FramingReference,
    judge,
)
from app.registry import get_spec
from training import DEFAULT_SEED
from training.prepare_dataset import is_image, looks_like_label_mask

BATCH = 32
FOLIAR_MODELS = ("leaf-foliar", "coffee-foliar", "banana-foliar", "citrus-foliar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra o portão de enquadramento zero-shot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--models", nargs="+", default=list(FOLIAR_MODELS))
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--max-per-model", type=int, default=3000)
    parser.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Fração (em %%) das folhas boas da validação que o limiar aceita recusar.",
    )
    parser.add_argument("--negatives", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--per-crop-thresholds",
        action="store_true",
        help=(
            "Grava também um limiar por cultura (percentil das fotos boas de cada uma). "
            "Desligado: medido nos recortes de campo, ele deixou passar pseudocaule na "
            "banana e recusou folha boa de citros."
        ),
    )
    parser.add_argument(
        "--positives",
        nargs="*",
        type=Path,
        default=[],
        help="Outras fotos que DEVEM passar (ex.: folha em close fora do catálogo).",
    )
    parser.add_argument(
        "--folha-vencedora",
        nargs="*",
        default=["coffee"],
        help=(
            "Culturas em que a cena mais provável também tem de ser folha (ver "
            "FramingReference.leaf_winner_crops). Padrão: café."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=None, help="Padrão: models/framing.npz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or settings.models_dir / REFERENCE_NAME

    import torch
    from transformers import AutoModel, AutoProcessor

    from app.framing import vision_encoder
    from training.dataset import index_by_class, split_indices

    print(f"\n=== Calibrando enquadramento — {args.model_id} ===")
    model = AutoModel.from_pretrained(args.model_id).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    prompts = [scene.prompt for scene in SCENES]
    tokens = processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        text = model.get_text_features(**tokens)
    text = getattr(text, "pooler_output", text).float().numpy().astype(np.float64)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    logit_scale = float(model.logit_scale.detach().exp())
    del model  # a torre de texto acaba aqui

    encode = vision_encoder(args.model_id)
    rng = np.random.default_rng(args.seed)

    calibration, tests = {}, {}
    for key in args.models:
        spec = get_spec(key)
        paths, _, _ = index_by_class(spec.data_dir)
        _, val_idx, test_idx = split_indices(
            len(paths), args.seed, args.validation_split, args.test_split
        )
        # Máscaras de segmentação não são "fotos boas": 85 delas na validação do
        # milho puxaram o limiar para 0,03 na primeira calibração, e colmo passava.
        val = _cap(_photos(paths[i] for i in val_idx), args.max_per_model, rng)
        test = _cap(_photos(paths[i] for i in test_idx), args.max_per_model, rng)
        print(f"[{key}] calibração {len(val)} · teste {len(test)}")
        calibration[key] = embed(encode, val)
        tests[key] = embed(encode, test)

    draft = FramingReference(
        model_id=args.model_id,
        keys=tuple(scene.key for scene in SCENES),
        prompts=tuple(prompts),
        in_scope=np.array([scene.in_scope for scene in SCENES]),
        crops=tuple(scene.crop or "" for scene in SCENES),
        text_embeddings=text,
        logit_scale=logit_scale,
        threshold=0.0,
        calibrated_on=0,
    )
    scores = draft.in_scope_probability(np.vstack(list(calibration.values())))
    reference = dataclasses.replace(
        draft,
        threshold=float(np.percentile(scores, args.percentile)),
        calibrated_on=len(scores),
    )
    per_crop = crop_thresholds(draft, calibration, args.percentile)
    if args.per_crop_thresholds:
        reference = dataclasses.replace(reference, thresholds=per_crop)
    reference = dataclasses.replace(reference,
                                    leaf_winner_crops=tuple(args.folha_vencedora or ()))
    print(f"\nLimiar geral: P(no escopo) >= {reference.threshold:.4f} ({len(scores)} fotos boas)")
    uso = "gravado" if args.per_crop_thresholds else "só informativo (--per-crop-thresholds)"
    for crop, value in sorted(per_crop.items()):
        print(f"  limiar de {crop}, {uso}: >= {value:.4f}")

    print("\nFolhas boas recusadas, no teste que a calibração não viu (modelo escolhido):")
    for key, embeddings in tests.items():
        crop = get_spec(key).crop
        results = judge(reference, embeddings, crop)
        refused = np.mean([not r.in_scope for r in results])
        crop_ok = np.mean([r.crop == crop for r in results])
        warned = np.mean([r.crop_matches is False for r in results])
        print(
            f"  {key:16s} {refused:6.1%} recusadas · cultura mais parecida certa em "
            f"{crop_ok:.1%} · aviso de cultura trocada em {warned:.1%}"
        )
    print_crop_warning_curve(reference, tests)

    probes = {
        folder: embed(encode, _images_in(folder)) for folder in args.negatives + args.positives
    }
    for label, folders in (
        ("fora do escopo — pego", args.negatives),
        ("devem passar", args.positives),
    ):
        if folders:
            print(f"\n{label.capitalize()}:")
        for folder in folders:
            results = judge(reference, probes[folder])
            caught = np.mean([not r.in_scope for r in results])
            shown = caught if label.startswith("fora") else 1 - caught
            scenes = Counter(r.scene for r in results).most_common(3)
            print(
                f"  {folder.name:22s} {shown:6.1%} de {len(results):4d}   "
                f"cenas: {', '.join(f'{s} {n}' for s, n in scenes)}"
            )

    print_curve(draft, scores, tests, probes, args.negatives)

    reference.save(output)
    print(f"\nReferência salva em: {output}")


def crop_thresholds(
    draft: FramingReference, calibration: dict[str, np.ndarray], percentile: float
) -> dict[str, float]:
    """O limiar de cada cultura, com o portão restrito à folha dela.

    Não é gravado por padrão. O percentil das fotos boas só vale se a validação
    parece o que o rover vê, e não parece: no citros é folha sobre papel (limiar
    0,842, que recusou 58 recortes de folha boa de campo) e na banana tem planta
    inteira (limiar 0,004, que deixou passar o pseudocaule). A restrição à cultura
    com o limiar geral barrou 35 recortes de pseudocaule, corte e tronco na banana
    e nenhuma folha boa de citros — medido em 27/09/2026 nos recortes do PlantSeg.
    """
    by_crop: dict[str, list[np.ndarray]] = {}
    for key, embeddings in calibration.items():
        crop = get_spec(key).crop
        if crop:
            by_crop.setdefault(crop, []).append(embeddings)
    return {
        crop: float(np.percentile(draft.in_scope_probability(np.vstack(parts), crop), percentile))
        for crop, parts in by_crop.items()
    }


def print_crop_warning_curve(reference: FramingReference, tests: dict[str, np.ndarray]) -> None:
    """O preço de cada folga do aviso de cultura trocada.

    Folha boa avisada à toa: o teste do próprio modelo. Folha de outra cultura
    pega: o teste dos outros modelos, mandado a este.
    """
    print(
        "\nAviso de cultura trocada — folga (razão): folha boa avisada à toa / "
        f"folha de outra cultura pega (em uso: {CROP_MISMATCH_RATIO:g})"
    )
    keys = [key for key in tests if get_spec(key).crop]
    for ratio in (1, 2, 4, 8, 16):
        cells = []
        for key in keys:
            crop = get_spec(key).crop
            own = judge(reference, tests[key], crop, mismatch_ratio=ratio)
            others = [
                result
                for other in keys
                if get_spec(other).crop != crop
                for result in judge(reference, tests[other], crop, mismatch_ratio=ratio)
            ]
            false_alarm = np.mean([r.crop_matches is False for r in own])
            caught = np.mean([r.crop_matches is False for r in others]) if others else 0.0
            cells.append(f"{crop} {false_alarm:5.1%}/{caught:5.1%}")
        print(f"  {ratio:>3g}  " + "   ".join(cells))


def print_curve(draft, scores, tests, probes, negatives) -> None:
    """O preço de cada limiar: folha boa recusada contra fora do escopo pego."""
    names = [folder.name[:14] for folder in negatives]
    print("\nCurva — percentil da calibração que vira limiar:")
    print(
        f"  {'pct':>4s} {'limiar':>7s}  "
        + "  ".join(f"{k[:9]:>9s}" for k in tests)
        + "  "
        + "  ".join(f"{n:>14s}" for n in names)
    )
    for percentile in (0.5, 1, 2, 5, 10):
        reference = dataclasses.replace(draft, threshold=float(np.percentile(scores, percentile)))
        refused = [np.mean([not r.in_scope for r in judge(reference, e)]) for e in tests.values()]
        caught = [np.mean([not r.in_scope for r in judge(reference, probes[f])]) for f in negatives]
        print(
            f"  {percentile:>4g} {reference.threshold:7.3f}  "
            + "  ".join(f"{v:9.1%}" for v in refused)
            + "  "
            + "  ".join(f"{v:14.1%}" for v in caught)
        )
    print("  (colunas do meio: folhas boas recusadas; da direita: fora do escopo pego)")


def embed(encode, paths: list[str | Path]) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), BATCH):
        images = []
        for path in paths[start : start + BATCH]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        chunks.append(encode(images))
        print(f"\r    {min(start + BATCH, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    return np.vstack(chunks)


def _cap(paths: list[str], limit: int, rng: np.random.Generator) -> list[str]:
    if len(paths) <= limit:
        return paths
    return [paths[i] for i in sorted(rng.choice(len(paths), limit, replace=False))]


def _photos(paths) -> list[str]:
    return [p for p in paths if not looks_like_label_mask(Path(p))]


def _images_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if is_image(path))


if __name__ == "__main__":
    main()
