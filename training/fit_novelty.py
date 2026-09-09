"""Calibra a detecção de novidade de um modelo já treinado.

Rode **depois** do ``training.train``, uma vez por modelo::

    python -m training.fit_novelty --model coffee-foliar

Passa o dataset de treino pelo modelo, coleta a assinatura visual (embedding) de
cada imagem e grava a estatística resultante em ``models/<modelo>.novelty.npz``.
A API carrega esse arquivo sozinha no próximo start e passa a marcar
``needs_review`` no que não se parece com nada conhecido.

Sem este passo o modelo funciona normalmente — só não sabe dizer o que nunca viu.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.inference.novelty import DEFAULT_PERCENTILE, DEFAULT_SHRINKAGE, NoveltyReference
from app.keras_backend import to_numpy
from app.registry import MODEL_SPECS, ModelSpec, get_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra a detecção de novidade de um modelo treinado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Pesos a calibrar. Padrão: os do catálogo. A referência é gravada ao lado deles.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Percentil das distâncias de treino que vira o limiar.",
    )
    parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    parser.add_argument(
        "--max-images",
        type=int,
        default=20000,
        help="Teto de imagens usadas na calibração, para não estourar a memória.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    data_dir = args.data_dir or spec.data_dir
    weights = args.weights or spec.weights_path
    # A referência acompanha os pesos que a geraram. Gravá-la sempre no caminho
    # do catálogo faria uma calibração experimental sobrescrever a de produção.
    novelty_path = (
        spec.novelty_path if weights == spec.weights_path else weights.with_suffix(".novelty.npz")
    )

    _validate(weights, data_dir, args.percentile)

    # Import tardio: o backend custa segundos e ``--help`` deve ser instantâneo.
    import keras

    from training.dataset import load_labeled_dataset
    from training.model_def import EMBEDDING_LAYER

    print(f"\n=== Calibrando novidade — {spec.title} ({spec.key}) ===")
    print(f"Pesos:   {weights}")
    print(f"Dados:   {data_dir}\n")

    model = keras.models.load_model(weights)
    try:
        embedding_layer = model.get_layer(EMBEDDING_LAYER)
    except ValueError:
        raise SystemExit(
            f"O modelo não tem a camada {EMBEDDING_LAYER!r}. Ele foi treinado com uma "
            "versão anterior do model_def — retreine para habilitar a novidade."
        ) from None

    # ``model.inputs`` é uma lista; desembrulhada, o extrator aceita o lote como
    # tensor único — que é como ele chega do ``tf.data``, sem aviso de estrutura.
    inputs = model.inputs
    embedder = keras.Model(inputs[0] if len(inputs) == 1 else inputs, embedding_layer.output)
    dataset, class_names = load_labeled_dataset(data_dir, spec.img_size, args.batch_size)
    _warn_on_class_mismatch(spec, class_names)

    embeddings, labels = _collect(embedder, dataset, args.max_images)
    print(f"\n{len(embeddings)} embeddings de {embeddings.shape[1]} dimensões coletados.")

    reference = NoveltyReference.fit(
        embeddings,
        labels,
        tuple(class_names),
        percentile=args.percentile,
        shrinkage=args.shrinkage,
    )
    reference.save(novelty_path)

    _report(reference, embeddings, args.percentile)
    print(f"\nReferência salva em: {novelty_path}")
    print("A API passa a marcar 'needs_review' no próximo start.")


def _validate(weights: Path, data_dir: Path, percentile: float) -> None:
    if not weights.exists():
        raise SystemExit(
            f"Pesos não encontrados: {weights}\nTreine antes com 'python -m training.train'."
        )
    if not data_dir.exists():
        raise SystemExit(f"Diretório de dados não encontrado: {data_dir}")
    if not 50.0 <= percentile < 100.0:
        raise SystemExit("--percentile precisa estar entre 50 e 100 (exclusivo).")


def _collect(embedder, dataset, max_images: int) -> tuple[np.ndarray, np.ndarray]:
    """Roda o dataset pelo extrator de embeddings, respeitando o teto."""
    chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    total = 0

    for images, labels in dataset:
        # O lote sai do ``tf.data`` como tensor do TensorFlow. Passar por numpy
        # no meio do caminho é o que deixa o mesmo código alimentar qualquer
        # backend do Keras, em vez de só o TensorFlow.
        embedded = embedder(np.asarray(images), training=False)
        chunks.append(to_numpy(embedded).astype(np.float64))
        label_chunks.append(np.asarray(labels))
        total += len(labels)
        print(f"\r  {total} imagens processadas...", end="", flush=True)
        if total >= max_images:
            print(f"\n  teto de --max-images ({max_images}) atingido.")
            break

    if not chunks:
        raise SystemExit("Nenhuma imagem encontrada — verifique o diretório de dados.")
    return np.vstack(chunks), np.concatenate(label_chunks)


def _warn_on_class_mismatch(spec: ModelSpec, class_names: list[str]) -> None:
    if tuple(class_names) != tuple(spec.class_names):
        print(
            f"Aviso: classes do diretório ({', '.join(class_names)}) diferem do "
            f"catálogo ({', '.join(spec.class_names)}).\n"
            "A referência é gravada com a ordem do diretório, que precisa bater "
            "com a do .labels.json usado no treino."
        )


def _report(reference: NoveltyReference, embeddings: np.ndarray, percentile: float) -> None:
    """Mostra o que o limiar significa em termos de volume de revisão."""
    scores = reference.score(embeddings)
    flagged = float((scores > 1.0).mean())

    print(f"\nLimiar calibrado: {reference.threshold:.3f} (percentil {percentile:g})")
    print(f"Score médio no próprio treino: {scores.mean():.3f}")
    print(f"Marcaria {flagged:.1%} das imagens de treino para revisão.")
    print(
        "\nEm campo esse percentual sobe — é dado que o modelo nunca viu. Regule "
        "o volume da fila com NOVELTY_SENSITIVITY no .env (menor = mais imagens)."
    )


if __name__ == "__main__":
    main()
