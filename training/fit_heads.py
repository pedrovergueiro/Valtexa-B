"""Cabeças dNATY sobre o embedding do portão: diagnóstico e espécie.

O portão (:mod:`app.framing`) já calcula o embedding SigLIP 2 de toda foto e de todo
recorte do rover. Este script treina sobre ele as duas cabeças opcionais de um modelo
(:mod:`app.inference.heads`).

Diagnóstico — ``models/<pesos>.head.npz``::

    python -m training.import_bracol --zip ~/valtexa-raw/bracol/yy2k5y8mxg-v1.zip \\
        --output data/coffee-bracol
    python -m training.fit_heads --model coffee-foliar --extra data/coffee-bracol \\
        --mix cnn=0,head=1 --holdout data/coffee-bracol-holdout
    python -m training.fit_conformal --model coffee-foliar --field data/coffee-bracol-holdout

Espécie — ``models/<pesos>.species.npz``::

    python -m training.fit_heads --model citrus-foliar-cc --species \\
        --negatives ~/valtexa-raw/plantdoc --negatives data/banana-foliar \\
        --negatives data/coffee-foliar

A arquitetura do MLP é do **dNATY** (``dnaty.compress``): busca evolutiva sobre largura
e profundidade das camadas, ativação e saltos, escolhida pela validação, com ajuste
fino no fim. Só o treino precisa dele (``pip install dnaty --no-deps``); a API roda a
rede exportada em NumPy.

As regras são as dos outros ajustes:

- só o split de treino treina — com as quatro orientações do TTA como aumento, para a
  cabeça ler a folha de cabeça para baixo como lê a de pé;
- a validação escolhe (a arquitetura, no dNATY; o limiar, na espécie);
- o teste isolado só mede, e o relatório mostra a CNN sozinha, a cabeça sozinha e a
  mistura que a API vai devolver.

Os negativos da espécie precisam de licença compatível com a do modelo: a versão
comercial só pode aprender com dado CC BY (o PlantDoc é CC BY 4.0).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.config import settings
from app.inference.heads import DiagnosisHead, EmbeddingMLP, SpeciesCheck
from app.inference.preprocess import load_image_from_bytes, preprocess_batch
from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED
from training.prepare_dataset import is_image, looks_like_label_mask

CHUNK = 16
"""Fotos por passada pela torre; com as 4 orientações, 64 imagens."""

DECODERS = 3

DUPLICATE_COSINE = 0.97
"""Acima disto, a foto do ``--extra`` é tratada como cópia de uma do dataset principal.

Conservador de propósito. Medido no BRACOL contra o ``coffee-foliar`` (01/10/2026): as
73 cópias reais passam de 0,99; folhas diferentes no mesmo estilo (inteira, fundo
branco) chegam a 0,95-0,97. Descartar algumas folhas legítimas custa menos que deixar a
mesma folha no treino de um e no teste do outro."""

PROMPTS: dict[str, dict[str, list[str]]] = {
    "coffee": {
        "Cercospora_Leaf_Spot": [
            "a photo of a coffee leaf with cercospora leaf spot",
            "a coffee leaf with brown eye spot disease, round brown spots with a light center",
        ],
        "Healthy": [
            "a photo of a healthy coffee leaf",
            "a healthy green coffee leaf without spots",
        ],
        "Leaf_Miner_Damage": [
            "a photo of a coffee leaf damaged by the coffee leaf miner",
            "a coffee leaf with brown dry blotch mines made by leaf miner larvae",
        ],
        "Phoma_Leaf_Spot": [
            "a photo of a coffee leaf with phoma leaf spot",
            "a coffee leaf with dark brown necrotic lesions at the tip and margins",
        ],
        "Rust": [
            "a photo of a coffee leaf with coffee leaf rust",
            "a coffee leaf with orange-yellow powdery rust spots",
        ],
    },
}
"""Frases do zero-shot, por cultura. Fixadas **antes** de medir (01/10/2026): reescrevê-las
olhando o resultado no conjunto externo transformaria a medição em ajuste."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina as cabeças dNATY sobre o embedding do portão",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--weights", type=Path, default=None, help="Padrão: os do catálogo.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--species", action="store_true", help="Treina a cabeça de espécie.")
    parser.add_argument(
        "--negatives",
        type=Path,
        action="append",
        default=[],
        help="Pasta de folhas de outras plantas (espécie). Repita para várias.",
    )
    parser.add_argument(
        "--field",
        type=Path,
        action="append",
        default=[],
        help="Recortes de campo da própria cultura, como positivos extras (espécie).",
    )
    parser.add_argument(
        "--refuse",
        type=float,
        default=0.02,
        help="Espécie: fração de folhas da cultura (validação) que o limiar pode recusar.",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help=(
            "Diagnóstico: outro dataset com as mesmas classes (<Classe>/<foto>), com corte "
            "próprio. O que for quase cópia do dataset principal sai."
        ),
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=None,
        help=(
            "Diagnóstico: grava aqui (<Classe>/<foto>) as fotos do --extra que a cabeça não "
            "treinou — validação e teste —, para alargar o conformal com fit_conformal --field."
        ),
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Diagnóstico: inclui a leitura zero-shot por frases.",
    )
    parser.add_argument(
        "--mix",
        default=None,
        help=(
            "Pesos da média geométrica, ex.: cnn=1,head=1,zero_shot=1. "
            "Padrão: todas as leituras com peso 1."
        ),
    )
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--license", default=None, help="Padrão: a licença do modelo no catálogo.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def four_views(image: Image.Image, size: int) -> list[Image.Image]:
    """As orientações do TTA, a partir da redução que o processador do SigLIP faria."""
    small = image.resize((size, size), Image.Resampling.BILINEAR)
    return [
        small,
        ImageOps.mirror(small),
        ImageOps.flip(small),
        small.transpose(Image.Transpose.ROTATE_180),
    ]


def embed(
    encoder, paths: list[str], cnn=None, views: int = 4
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(N, views, D)`` embeddings; com ``cnn``, também as probabilidades dela (original)."""
    from app.framing import IMG_SIZE

    decoders = ThreadPoolExecutor(max_workers=DECODERS)
    chunks, probabilities = [], []
    started = time.perf_counter()
    for start in range(0, len(paths), CHUNK):
        images = list(
            decoders.map(
                lambda p: load_image_from_bytes(Path(p).read_bytes(), Path(p).name),
                paths[start : start + CHUNK],
            )
        )
        batch = [v for image in images for v in four_views(image, IMG_SIZE)[:views]]
        chunks.append(np.asarray(encoder(batch), dtype=np.float32).reshape(len(images), views, -1))
        if cnn is not None:
            probabilities.append(cnn._run(preprocess_batch(images, cnn.img_size))[0])
        done, seconds = min(start + CHUNK, len(paths)), time.perf_counter() - started
        print(f"\r  {done}/{len(paths)} fotos ({seconds:.0f}s)", end="", flush=True)
    print()
    return np.concatenate(chunks), (np.concatenate(probabilities) if probabilities else None)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def evolve(x_train, y_train, x_val, y_val, n_out: int, args) -> tuple[EmbeddingMLP, dict]:
    """O MLP do dNATY: semente larga, busca evolutiva, ajuste fino. Escolha pela validação."""
    import torch
    import torch.nn as nn
    from dnaty import compress

    raw_train = l2(x_train)
    mean, std = raw_train.mean(0), raw_train.std(0) + 1e-6

    def standardize(x):
        return ((l2(x) - mean) / std).astype(np.float32)

    width = x_train.shape[1]
    seed_model = nn.Sequential(
        nn.Linear(width, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, n_out),
    )
    started = time.perf_counter()
    result = compress(
        seed_model,
        (standardize(x_train), np.asarray(y_train)),
        val_data=(standardize(x_val), np.asarray(y_val)),
        n_generations=args.generations,
        n_pop=args.population,
        finetune_epochs=args.finetune_epochs,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
        progress_callback=lambda log: print(
            f"  geração {log.gen}: melhor validação {log.best_acc:.4f} ({log.n_params} parâmetros)",
            flush=True,
        ),
    )
    mlp = EmbeddingMLP.from_dnaty(result.model, mean, std)
    info = {
        "camadas": mlp.layer_sizes,
        "ativacoes": list(mlp.activations),
        "saltos": len(mlp.skips),
        "acc_validacao": round(result.accuracy, 4),
        "segundos": round(time.perf_counter() - started),
    }
    print(
        f"  dNATY: {' -> '.join(map(str, mlp.layer_sizes))}, "
        f"ativações {', '.join(mlp.activations)}, {len(mlp.skips)} salto(s); "
        f"validação {result.accuracy:.4f} em {info['segundos']}s"
    )
    return mlp, info


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float((np.asarray(probabilities).argmax(1) == np.asarray(labels)).mean())


def text_embeddings(
    model_id: str, class_names: tuple[str, ...], crop: str
) -> tuple[np.ndarray, float]:
    """Uma direção por classe: a média normalizada das frases dela."""
    import torch
    from transformers import AutoModel, AutoProcessor

    prompts = PROMPTS.get(crop)
    if prompts is None or set(prompts) != set(class_names):
        raise SystemExit(
            f"Sem frases para as classes de {crop!r}: edite PROMPTS antes de usar --zero-shot."
        )
    model = AutoModel.from_pretrained(model_id).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    rows = []
    for name in class_names:
        tokens = processor(
            text=prompts[name], padding="max_length", max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            text = model.get_text_features(**tokens)
        text = l2(getattr(text, "pooler_output", text).float().numpy().astype(np.float64))
        rows.append(l2(text.mean(0)))
    return np.array(rows), float(model.logit_scale.detach().exp())


def parse_mix(text: str | None, names: list[str]) -> dict[str, float]:
    if not text:
        return {name: 1.0 for name in names}
    mix = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in names:
            raise SystemExit(
                f"Leitura desconhecida em --mix: {name!r}. Válidas: {', '.join(names)}."
            )
        mix[name.strip()] = float(value)
    return mix


def link(source: Path, folder: Path) -> None:
    """Põe ``source`` em ``folder`` sem copiar (link físico), ou copia se não der."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def images_under(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.rglob("*") if is_image(p) and not looks_like_label_mask(p))


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    if args.weights is not None:
        spec = dataclasses.replace(spec, weights_name=str(args.weights.resolve()))
    data_dir = args.data_dir or spec.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Diretório de dados não encontrado: {data_dir}")
    if args.species and not args.negatives:
        raise SystemExit("A espécie precisa de --negatives: folhas de outras plantas.")

    from app.framing import REFERENCE_NAME, FramingReference, vision_encoder
    from app.model import CornDiseaseModel
    from training.dataset import index_by_class, split_indices

    reference = FramingReference.load(settings.models_dir / REFERENCE_NAME)
    encoder = vision_encoder(reference.model_id)
    paths, labels, class_names = index_by_class(data_dir)
    train_idx, val_idx, test_idx = split_indices(
        len(paths), args.seed, args.validation_split, args.test_split
    )
    # As máscaras saem depois do corte, como no fit_conformal: as fatias continuam as do treino.
    photos = [i for i in range(len(paths)) if not looks_like_label_mask(Path(paths[i]))]
    keep = set(photos)
    train_idx, val_idx, test_idx = (
        [i for i in part if i in keep] for part in (train_idx, val_idx, test_idx)
    )
    labels = np.asarray(labels)
    license_text = args.license if args.license is not None else spec.license

    if args.species:
        fit_species(
            args,
            spec,
            encoder,
            reference.model_id,
            paths,
            train_idx,
            val_idx,
            test_idx,
            license_text,
        )
        return

    model = CornDiseaseModel(spec)
    model.load()
    if not model.is_loaded:
        raise SystemExit(f"Não foi possível carregar os pesos de {spec.weights_path}.")
    if tuple(class_names) != tuple(model.class_names):
        raise SystemExit(
            "Classes do diretório diferem das do modelo; a cabeça precisa da mesma ordem."
        )

    print(f"\n=== Cabeça de diagnóstico — {spec.title} ({spec.key}) ===")
    print(
        f"Torre: {reference.model_id} · treino {len(train_idx)} fotos x4 · "
        f"validação {len(val_idx)} · teste {len(test_idx)}"
    )
    x_train, _ = embed(encoder, [paths[i] for i in train_idx])
    x_val, _ = embed(encoder, [paths[i] for i in val_idx], views=1)
    x_test, cnn_test = embed(encoder, [paths[i] for i in test_idx], cnn=model, views=1)
    tests = {data_dir.name: (x_test[:, 0], labels[test_idx], cnn_test)}

    train_x = [x_train.reshape(-1, x_train.shape[-1])]
    train_y = [np.repeat(labels[train_idx], 4)]
    val_x, val_y = [x_val[:, 0]], [labels[val_idx]]
    known = l2(np.vstack([x_train[:, 0], x_val[:, 0], x_test[:, 0]]))
    for folder in args.extra:
        extra_paths, extra_labels, extra_classes = index_by_class(folder)
        if tuple(extra_classes) != tuple(class_names):
            raise SystemExit(f"{folder}: classes {extra_classes} diferem das do modelo.")
        extra_labels = np.asarray(extra_labels)
        part = split_indices(len(extra_paths), args.seed, args.validation_split, args.test_split)
        x_extra, cnn_extra = embed(encoder, extra_paths, cnn=model)
        # Quase cópia de uma foto do dataset principal: a mesma folha, talvez em outro
        # split. Fica de fora, para o teste de um não estar no treino do outro.
        copy = (l2(x_extra[:, 0]) @ known.T).max(axis=1) > DUPLICATE_COSINE
        train, val, test = ([i for i in idx if not copy[i]] for idx in part)
        print(
            f"Extra {folder.name}: {len(extra_paths)} fotos, {int(copy.sum())} quase cópias "
            f"fora · treino {len(train)} x4 · validação {len(val)} · teste {len(test)}"
        )
        train_x.append(x_extra[train].reshape(-1, x_extra.shape[-1]))
        train_y.append(np.repeat(extra_labels[train], 4))
        val_x.append(x_extra[val][:, 0])
        val_y.append(extra_labels[val])
        tests[folder.name] = (x_extra[test][:, 0], extra_labels[test], cnn_extra[test])
        if args.holdout is not None:
            for i in sorted(val + test):
                link(Path(extra_paths[i]), args.holdout / class_names[extra_labels[i]])

    mlp, info = evolve(
        np.vstack(train_x),
        np.concatenate(train_y),
        np.vstack(val_x),
        np.concatenate(val_y),
        len(class_names),
        args,
    )
    names = ["cnn", "head"]
    texts, scale = None, 0.0
    if args.zero_shot:
        texts, scale = text_embeddings(
            reference.model_id, tuple(class_names), spec.crop or spec.key
        )
        names.append("zero_shot")
    head = DiagnosisHead(
        class_names=tuple(class_names),
        mlp=mlp,
        encoder=reference.model_id,
        mix=parse_mix(args.mix, names),
        text_embeddings=texts,
        logit_scale=scale,
        trained_on=sum(len(y) for y in train_y) // 4,
        license=license_text,
    )

    for source, (x, y, cnn) in tests.items():
        print(f"\nTeste isolado — {source} ({len(y)} fotos, sem TTA):")
        print(f"  CNN sozinha        {accuracy(cnn, y):.4f}")
        for name, probabilities in head.readings(x).items():
            print(f"  {name:18s} {accuracy(probabilities, y):.4f}")
        print(f"  mistura {head.mix}  {accuracy(head.combine(cnn, x), y):.4f}")

    output = args.output or spec.head_path
    head.save(output)
    print(f"\nCabeça salva em: {output}  ({info})")
    print("Recalibre a predição conformal (training.fit_conformal): a resposta mudou.")


def fit_species(
    args, spec, encoder, model_id, paths, train_idx, val_idx, test_idx, license_text
) -> None:
    """Folha da cultura contra folha de outra planta, com o limiar na validação."""
    rng = np.random.default_rng(args.seed)
    print(f"\n=== Espécie — {spec.title} ({spec.key}): {spec.crop} ou outra planta? ===")

    def thirds(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        order = rng.permutation(len(items))
        a, b = int(0.7 * len(items)), int(0.85 * len(items))
        return (
            [items[i] for i in order[:a]],
            [items[i] for i in order[a:b]],
            [items[i] for i in order[b:]],
        )

    positives = {
        "treino": [paths[i] for i in train_idx],
        "validação": [paths[i] for i in val_idx],
        "teste": [paths[i] for i in test_idx],
    }
    for folder in args.field:
        # Recortes da mesma foto ficam juntos: "<foto>_r<n>.jpg".
        crops = images_under(folder)
        groups = sorted({Path(p).name.rsplit("_r", 1)[0] for p in crops})
        held = set(rng.permutation(groups)[: max(1, len(groups) // 5)].tolist())
        positives["treino"] += [p for p in crops if Path(p).name.rsplit("_r", 1)[0] not in held]
        positives["validação"] += [p for p in crops if Path(p).name.rsplit("_r", 1)[0] in held]
    negatives = {folder.name: thirds(images_under(folder)) for folder in args.negatives}

    pos_train = embed(encoder, positives["treino"])[0]
    pos_val = embed(encoder, positives["validação"])[0]
    pos_test = embed(encoder, positives["teste"])[0]
    neg = {name: [embed(encoder, part)[0] for part in parts] for name, parts in negatives.items()}

    flat = lambda x: x.reshape(-1, x.shape[-1])  # noqa: E731
    x_pos, x_neg = flat(pos_train), np.vstack([flat(parts[0]) for parts in neg.values()])
    repeat = max(1, len(x_pos) // max(1, len(x_neg)))
    x_train = np.vstack([x_pos] + [x_neg] * repeat)
    y_train = np.concatenate([np.ones(len(x_pos), int), np.zeros(len(x_neg) * repeat, int)])
    v_neg = np.vstack([flat(parts[1]) for parts in neg.values()])
    x_val = np.vstack([flat(pos_val), v_neg])
    y_val = np.concatenate([np.ones(len(flat(pos_val)), int), np.zeros(len(v_neg), int)])
    print(
        f"Treino: {len(x_pos)} da cultura, {len(x_neg)} de outras plantas (x{repeat}); "
        f"validação {len(y_val)}"
    )

    mlp, info = evolve(x_train, y_train, x_val, y_val, 2, args)
    scores_val = mlp.probabilities(flat(pos_val))[:, 1]
    threshold = float(np.quantile(scores_val, args.refuse))
    check = SpeciesCheck(
        crop=spec.crop or "",
        mlp=mlp,
        threshold=threshold,
        encoder=model_id,
        trained_on=len(x_pos) // 4,
        license=license_text,
    )
    print(f"\nLimiar {threshold:.4f} (recusa {args.refuse:.0%} da validação da cultura). Teste:")
    refused = 1 - check.matches(flat(pos_test)).mean()
    print(f"  {spec.crop} recusada (teste isolado, 4 orientações)  {refused:.1%}")
    for name, parts in neg.items():
        print(f"  {name:28s} recusada  {1 - check.matches(flat(parts[2])).mean():.1%}")

    output = args.output or spec.species_path
    check.save(output)
    print(f"\nVerificação de espécie salva em: {output}  ({info})")


if __name__ == "__main__":
    main()
