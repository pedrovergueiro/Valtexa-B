"""Treina um dos modelos do catálogo (ver ``MODELS.md``).

Execute a partir da raiz do repositório, como módulo::

    python -m training.train --model leaf-foliar --epochs 15 --fine-tune-epochs 5

O diretório de dados precisa ter uma subpasta por classe. Por padrão é
``data/<modelo>/``, e o ``prepare_dataset`` monta essa estrutura a partir dos
datasets públicos::

    data/leaf-foliar/
      Blight/*.jpg
      Common_Rust/*.jpg
      Gray_Leaf_Spot/*.jpg
      Healthy/*.jpg

Ao final ficam em ``models/`` os pesos da melhor época e o ``.labels.json`` com a
ordem das classes — que é o que a API lê para traduzir índices em nomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from app.registry import MODEL_SPECS, ModelSpec, get_spec
from training import DEFAULT_SEED
from training.model_def import ARCHITECTURES, DEFAULT_ARCH

MONITOR = "val_accuracy"
"""Métrica que decide qual época é a melhor, tanto no checkpoint quanto no
early stopping."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino dos modelos de diagnóstico em milho",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        default=settings.default_model,
        help="Modelo do catálogo a treinar.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Padrão: data/<modelo>/.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Épocas da fase 1 (cabeça).")
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=15, help="Épocas da fase 2. Use 0 para pular."
    )
    parser.add_argument(
        "--arch",
        choices=sorted(ARCHITECTURES),
        default=None,
        help=(
            f"Base convolucional. Padrão: a do catálogo, ou {DEFAULT_ARCH}. "
            "Mais pesada = mais precisa e mais lenta."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Lado da entrada. Padrão: o do catálogo. Aumentar ajuda em lesão pequena.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Avalia o teste com test-time augmentation (média de 4 variações).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fração separada para o teste final, que não influencia nenhuma decisão.",
    )
    parser.add_argument("--patience", type=int, default=4, help="Paciência do early stopping.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="LR da fase 1.")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4, help="LR da fase 2.")
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=40,
        help="Quantas camadas finais da base liberar na fase 2. 0 libera todas.",
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Desliga a compensação de classes desbalanceadas.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Onde cachear as imagens decodificadas. Acelera as épocas seguintes.",
    )
    parser.add_argument(
        "--backgrounds",
        type=Path,
        default=None,
        help=(
            "Pasta de fotos de fundo real (ex.: quadros de pomar). Com ela, folhas "
            "fotografadas sobre papel são coladas sobre esses fundos no treino."
        ),
    )
    parser.add_argument("--background-prob", type=float, default=0.7)
    parser.add_argument("--background-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=None, help="Padrão: o caminho de pesos do modelo."
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ModelSpec:
    """Preenche os caminhos que dependem do modelo escolhido."""
    spec = get_spec(args.model)
    if args.data_dir is None:
        args.data_dir = spec.data_dir
    if args.output is None:
        args.output = spec.weights_path
    if getattr(args, "arch", None) is None:
        # A arquitetura é parte da identidade do modelo no catálogo: sem isto,
        # ``--model leaf-foliar-b`` treinaria a base padrão e a "segunda
        # opinião" sairia cópia da primeira.
        args.arch = spec.arch or DEFAULT_ARCH
    return spec


def main() -> None:
    args = parse_args()
    spec = resolve_paths(args)
    _validate_args(args)

    print(f"\n=== {spec.title} ({spec.key}) ===")
    print(f"{spec.summary}\n")
    if spec.needs_field_data:
        print(
            "Atenção: este modelo depende de coleta própria em campo para as "
            "classes que não têm dataset público. Ver MODELS.md.\n"
        )

    # Imports tardios: carregar o backend custa alguns segundos, e assim
    # ``--help`` e a validação dos argumentos respondem na hora.
    from training.dataset import load_datasets
    from training.model_def import build_model, unfreeze_base

    img_size = args.img_size or spec.img_size
    _, parametros, top1 = ARCHITECTURES[args.arch]
    print(f"Base: {args.arch} ({parametros} params, {top1:.1%} top-1 no ImageNet)")
    print(f"Entrada: {img_size}x{img_size}\n")

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data_dir,
        img_size,
        args.batch_size,
        args.validation_split,
        args.seed,
        test_split=args.test_split,
        cache_dir=args.cache_dir,
        backgrounds=_backgrounds(args, img_size),
        background_prob=args.background_prob,
    )
    _report_class_order(spec, class_names)

    class_weight = None if args.no_class_weight else _class_weight(args.data_dir, class_names)

    # Gravado antes do treino: se a execução for interrompida, o checkpoint já
    # salvo continua acompanhado da ordem de classes correta.
    labels_path = _write_labels(spec, args.output, class_names)

    import keras

    model = build_model(img_size, num_classes=len(class_names), arch=args.arch)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n== Fase 1: treinando a cabeça de classificação (base congelada) ==")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=_build_callbacks(args, args.output, best_so_far=None),
    )
    best_accuracy = _best_metric(history)

    if args.fine_tune_epochs > 0:
        liberadas = args.fine_tune_layers or None
        alvo = f"as {liberadas} camadas finais" if liberadas else "a base inteira"
        print(f"\n== Fase 2: fine-tuning de {alvo} ==")
        base = unfreeze_base(model, trainable_layers=liberadas)
        treinaveis = sum(1 for layer in base.layers if layer.trainable)
        print(f"{treinaveis} de {len(base.layers)} camadas da base liberadas.")

        # Cosine decay: o fine-tuning começa com a taxa cheia, para sair do ponto
        # onde a fase 1 parou, e termina perto de zero, o que deixa o modelo
        # assentar num mínimo em vez de ficar quicando em volta dele. Com taxa
        # constante as últimas épocas costumam oscilar sem ganhar nada.
        passos_por_epoca = max(1, int(train_ds.cardinality()))
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.fine_tune_lr,
            decay_steps=passos_por_epoca * args.fine_tune_epochs,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=schedule),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # ``best_so_far`` impede que a fase 2 sobrescreva um checkpoint melhor
        # da fase 1 — sem isso, o limiar do checkpoint reinicia e a primeira
        # época do fine-tuning grava por cima mesmo sendo pior.
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weight,
            callbacks=_build_callbacks(args, args.output, best_so_far=best_accuracy),
        )
        best_accuracy = max(best_accuracy, _best_metric(history))

    print(f"\nMelhor {MONITOR}: {best_accuracy:.4f}")
    print(f"Pesos salvos em:  {args.output}")
    print(f"Labels salvos em: {labels_path}")

    if test_ds is not None:
        _report_test_performance(args.output, test_ds, class_names, tta=args.tta)

    print(
        "\nLembre-se: métrica medida sobre dataset público não prevê o "
        "desempenho em campo. Valide num conjunto de teste capturado pela sua "
        "própria plataforma antes de prometer número para cliente."
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise SystemExit(
            f"Diretório de dados não encontrado: {args.data_dir}\n"
            "Monte o dataset com 'python -m training.prepare_dataset --help' "
            "ou organize as imagens em subpastas por classe."
        )
    if not 0.0 < args.validation_split < 1.0:
        raise SystemExit("--validation-split precisa estar entre 0 e 1 (exclusivo).")
    if args.epochs < 1:
        raise SystemExit("--epochs precisa ser pelo menos 1.")

    test_split = getattr(args, "test_split", 0.0)
    if test_split < 0.0:
        raise SystemExit("--test-split não pode ser negativo.")
    if not args.validation_split + test_split < 1.0:
        raise SystemExit(
            "--validation-split + --test-split precisa sobrar dado para treinar "
            f"(soma atual: {args.validation_split + test_split})."
        )


def _class_weight(data_dir: Path, class_names: list[str]) -> dict[int, float] | None:
    """Peso por classe, inversamente proporcional ao tamanho dela.

    Sem isso, a classe com menos exemplos contribui menos para a perda e o modelo
    aprende a preferir as abundantes — o que aparece como recall baixo justamente
    na doença mais rara, que costuma ser a que interessa detectar cedo. No
    ``leaf-foliar`` a diferença é de mais de duas vezes entre Gray_Leaf_Spot e
    Common_Rust.

    Devolve ``None`` quando as classes já estão equilibradas, para não introduzir
    um fator que só adiciona ruído.
    """
    from training.prepare_dataset import is_image

    counts = [
        sum(1 for item in (data_dir / name).iterdir() if is_image(item)) for name in class_names
    ]
    if not counts or min(counts) == 0:
        return None

    total = sum(counts)
    if max(counts) / min(counts) < 1.5:
        print("\nClasses equilibradas — sem compensação de peso.")
        return None

    weights = {index: total / (len(counts) * count) for index, count in enumerate(counts)}
    print("\nPesos por classe (compensando o desbalanceamento):")
    for index, name in enumerate(class_names):
        print(f"    {name:<22} n={counts[index]:>6}  peso={weights[index]:.2f}")
    return weights


def _report_test_performance(
    weights_path: Path, test_ds, class_names: list[str], *, tta: bool = False
) -> None:
    """Avalia os pesos da melhor época no conjunto que nunca foi consultado."""
    import keras

    from training.metrics import evaluate_dataset, format_report

    # Recarrega do disco de propósito: é exatamente o artefato que vai para a
    # API, e não o objeto em memória, que pode diferir se o early stopping tiver
    # restaurado outros pesos.
    best = keras.models.load_model(weights_path)

    matrix = evaluate_dataset(best, test_ds, len(class_names))
    print(format_report(matrix, class_names, title="Conjunto de teste (nunca visto)"))

    if tta:
        matrix_tta = evaluate_dataset(best, test_ds, len(class_names), tta=True)
        print(format_report(matrix_tta, class_names, title="Mesmo teste, com TTA"))
        print(
            "\nO TTA custa 4 inferências por imagem. Só vale ligar em produção se "
            "o ganho acima justificar quadruplicar a latência do /predict."
        )


def _build_callbacks(args: argparse.Namespace, output: Path, best_so_far: float | None) -> list:
    """Callbacks de uma fase de treino.

    São recriados por fase de propósito: o early stopping precisa reiniciar sua
    paciência, enquanto o checkpoint precisa herdar o melhor resultado já obtido
    via ``best_so_far``.
    """
    import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor=MONITOR,
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output),
            monitor=MONITOR,
            mode="max",
            save_best_only=True,
            initial_value_threshold=best_so_far,
            verbose=1,
        ),
    ]


def _best_metric(history) -> float:
    values = history.history.get(MONITOR)
    if not values:
        raise SystemExit(f"Métrica {MONITOR!r} ausente no histórico — verifique o dataset.")
    return float(max(values))


def _backgrounds(args: argparse.Namespace, img_size: int):
    """Os recortes de fundo da troca de fundo, se ``--backgrounds`` foi dado."""
    if args.backgrounds is None:
        return None
    from training.background import load_backgrounds

    crops = load_backgrounds(args.backgrounds, img_size, args.background_count, args.seed)
    print(
        f"Troca de fundo: {len(crops)} recortes de {args.backgrounds}, "
        f"probabilidade {args.background_prob:.0%} por imagem de treino."
    )
    return crops


def _report_class_order(spec: ModelSpec, class_names: list[str]) -> None:
    print(f"\nClasses encontradas ({len(class_names)}): {', '.join(class_names)}")
    if tuple(class_names) != tuple(spec.class_names):
        print(
            f"Aviso: a ordem difere do catálogo para {spec.key} "
            f"({', '.join(spec.class_names)}).\n"
            "Não é um problema: a API lê o .labels.json gerado aqui, que tem "
            "precedência. Ajuste o catálogo apenas se quiser manter os dois iguais."
        )


def _write_labels(spec: ModelSpec, output: Path, class_names: list[str]) -> Path:
    """Grava a ordem das classes ao lado dos pesos."""
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_path = (
        spec.labels_path if output == spec.weights_path else output.with_suffix(".labels.json")
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    return labels_path


if __name__ == "__main__":
    main()
