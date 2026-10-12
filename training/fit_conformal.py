"""Calibra a predição conformal de um modelo já treinado.

Rode depois do treino, uma vez por modelo, com o **mesmo corte** usado nele::

    python -m training.fit_conformal --model leaf-foliar --test-split 0.15

Calibra os limiares na **validação** e mede a garantia no **teste isolado**,
que nem o treino nem a calibração viram. Grava ``models/<pesos>.conformal.json``;
a API passa a devolver ``prediction_set`` no próximo start.

Com ``--field``, recortes de campo (``<pasta>/<Classe>/<foto>_r<n>.jpg``, como os
do ``lesion_tiles``) alargam os limiares das classes que aparecem neles: as fotos
são divididas ao meio, a metade A calibra e a metade B mede. Sem isso, a garantia
medida na validação não valia no campo da banana (57% de cobertura)::

    python -m training.fit_conformal --model banana-foliar \
        --data-dir data/banana-foliar-campo --field _plantseg-recortes/test/banana

O arquivo grava os limiares alargados (o conjunto honesto, que a API devolve) **e** os
originais (``base_thresholds``). Os originais continuam alimentando a fusão por planta:
ela divide a evidência de cada recorte pelo tamanho do conjunto, e só alargar derrubou a
detecção da banana de 9 para 7 de 18 plantas doentes (01/10/2026).

A validação já serviu para escolher a época do treino, e isso a deixa um pouco
otimista como calibração: o modelo foi escolhido por ir bem nela. É exatamente
por isso que o relatório mede a cobertura no teste antes de gravar — se a
garantia não se sustentar lá, o número aparece abaixo do pedido, e é ele que
vale.
"""

from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.inference.conformal import DEFAULT_ALPHA, ConformalReference, coverage_report
from app.inference.preprocess import load_image_from_bytes
from app.model import CornDiseaseModel
from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED
from training.prepare_dataset import looks_like_label_mask

BATCH = 32
DECODERS = 3
"""Threads decodificando JPEG. O Pillow solta o GIL na decodificação, e ela — não
a rede — é o gargalo; mais que isso disputa a máquina com quem está usando."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra a predição conformal de um modelo treinado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    parser.add_argument("--weights", type=Path, default=None, help="Padrão: os do catálogo.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="O mesmo do treino. A calibração usa exatamente esta fatia.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="O mesmo do treino. O relatório mede a cobertura nesta fatia.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--field",
        type=Path,
        default=None,
        help="Recortes de campo (<Classe>/<foto>_r<n>.jpg) para alargar os limiares.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    if args.weights is not None:
        spec = dataclasses.replace(spec, weights_name=str(args.weights.resolve()))
    data_dir = args.data_dir or spec.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Diretório de dados não encontrado: {data_dir}")

    # Import tardio: o dataset puxa o TensorFlow, e ``--help`` deve ser instantâneo.
    from training.dataset import index_by_class, split_indices

    model = CornDiseaseModel(spec)
    model.load()
    if not model.is_loaded:
        raise SystemExit(f"Não foi possível carregar os pesos de {spec.weights_path}.")
    if spec.head_path.exists() or spec.species_path.exists():
        # A cabeça (training.fit_heads) lê o embedding do portão. Sem ele carregado, a
        # calibração sairia da CNN sozinha — e a API responde com a mistura.
        from app.framing import framing_model

        framing_model.load()
        if spec.head_path.exists() and not model.safeguards["diagnosis_head"]:
            raise SystemExit(
                f"Há cabeça em {spec.head_path}, mas o portão não carregou: a calibração "
                "sairia de outra resposta que a da API."
            )

    paths, labels, class_names = index_by_class(data_dir)
    if tuple(class_names) != tuple(model.class_names):
        raise SystemExit(
            f"Classes do diretório ({', '.join(class_names)}) diferem das do modelo "
            f"({', '.join(model.class_names)}). A calibração precisa da mesma ordem."
        )
    _, val_idx, test_idx = split_indices(
        len(paths), args.seed, args.validation_split, args.test_split
    )
    labels = np.asarray(labels)
    # Tira as máscaras de segmentação *depois* do corte: as fatias continuam as do
    # treino, só sem imagens que não são foto. Ver prepare_dataset.looks_like_label_mask.
    val_idx = [i for i in val_idx if not looks_like_label_mask(Path(paths[i]))]
    test_idx = [i for i in test_idx if not looks_like_label_mask(Path(paths[i]))]

    print(f"\n=== Calibrando predição conformal — {spec.title} ({spec.key}) ===")
    print(f"Pesos: {spec.weights_path}")
    print(f"Calibração: {len(val_idx)} imagens de validação · teste: {len(test_idx)}")
    print(f"Cobertura pedida: {1 - args.alpha:.0%} por classe\n")

    calibration = collect(model, [paths[i] for i in val_idx])
    reference = ConformalReference.fit(
        calibration, labels[val_idx], tuple(class_names), alpha=args.alpha
    )

    test = collect(model, [paths[i] for i in test_idx])
    for mode in ("plain", "tta"):
        report(reference, test[mode], labels[test_idx], tta=mode == "tta")

    if args.field is not None:
        reference = widen_with_field(model, reference, args.field, tuple(class_names), args.seed)
        print("\nDepois de alargar com o campo, de novo no teste isolado:")
        for mode in ("plain", "tta"):
            report(reference, test[mode], labels[test_idx], tta=mode == "tta")

    output = spec.weights_path.with_suffix(".conformal.json")
    reference.save(output)
    print(f"\nReferência salva em: {output}")
    print("A API passa a devolver 'prediction_set' no próximo start.")


def split_photos(paths: list[Path], labels: list[int], seed: int) -> tuple[list[int], list[int]]:
    """Divide os recortes ao meio **por foto**, classe a classe.

    Recortes da mesma foto são a mesma lesão vista de novo: se uma foto desse
    recortes às duas metades, a medição na metade B seria otimista.
    """
    photos: dict[tuple[int, str], list[int]] = {}
    for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
        photos.setdefault((label, path.name.rsplit("_r", 1)[0]), []).append(index)
    rng = np.random.default_rng(seed)
    first, second = [], []
    for label in sorted({key[0] for key in photos}):
        keys = sorted(key for key in photos if key[0] == label)
        for position, key in enumerate(rng.permutation(len(keys))):
            (first if position % 2 == 0 else second).extend(photos[keys[key]])
    return sorted(first), sorted(second)


def widen_with_field(
    model: CornDiseaseModel,
    reference: ConformalReference,
    folder: Path,
    class_names: tuple[str, ...],
    seed: int,
) -> ConformalReference:
    """Alarga os limiares com a metade A dos recortes de campo e mede na metade B."""
    from training.prepare_dataset import is_image

    paths, labels = [], []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if sub.name not in class_names:
            raise SystemExit(f"Classe de campo {sub.name!r} não existe no modelo.")
        found = sorted(p for p in sub.iterdir() if is_image(p))
        paths += found
        labels += [class_names.index(sub.name)] * len(found)
    first, second = split_photos(paths, labels, seed)
    labels_arr = np.asarray(labels)
    print(f"\nCampo: {len(first)} recortes calibram, {len(second)} medem ({folder})")
    calibration = collect(model, [str(paths[i]) for i in first])
    held_out = collect(model, [str(paths[i]) for i in second])
    widened = reference.widened(calibration, labels_arr[first])
    for name, ref in (("antes", reference), ("depois", widened)):
        print(f"\nCampo, metade que a calibração não viu — {name}:")
        for mode in ("plain", "tta"):
            report(ref, held_out[mode], labels_arr[second], tta=mode == "tta")
    return widened


def collect(model: CornDiseaseModel, paths: list[str]) -> dict[str, np.ndarray]:
    """Probabilidades nos dois modos, pela mesma rota que a API usa."""
    out: dict[str, list] = {"plain": [], "tta": []}
    decoders = ThreadPoolExecutor(max_workers=DECODERS)
    for start in range(0, len(paths), BATCH):
        chunk = [Path(p) for p in paths[start : start + BATCH]]
        # Uma decodificação para os dois modos: é ela, não a rede, que custa.
        images = list(decoders.map(lambda p: load_image_from_bytes(p.read_bytes(), p.name), chunk))
        names = [path.name for path in chunk]
        for mode in out:
            for result in model.predict_images(names, images, tta=mode == "tta"):
                out[mode].append([item.probability for item in result.probabilities])
        print(f"\r  {min(start + BATCH, len(paths))}/{len(paths)} imagens", end="", flush=True)
    print()
    return {mode: np.array(rows) for mode, rows in out.items()}


def report(reference: ConformalReference, probabilities, labels, tta: bool) -> dict:
    """Mostra o que a garantia virou no teste isolado."""
    result = coverage_report(reference, probabilities, labels, tta)
    print(f"\n{'Com' if tta else 'Sem'} TTA — teste isolado:")
    print(f"  cobertura            {result['coverage']:.1%}  (pedida: {reference.coverage:.0%})")
    for name, value in result["coverage_per_class"].items():
        print(f"    {name:22s} {value:.1%}" if value is not None else f"    {name:22s} —")
    print(f"  tamanho médio        {result['mean_set_size']:.2f} classes")
    print(f"  resposta única       {result['singleton_rate']:.1%} das imagens")
    if result["singleton_accuracy"] is not None:
        print(f"    acerto nelas       {result['singleton_accuracy']:.1%}")
    print(f"  conjunto vazio       {result['empty_rate']:.1%}")
    if result["errors_flagged"] is not None:
        print(
            f"  dos {result['errors']} erros do top-1, {result['errors_flagged']:.1%} "
            "viram conjunto com mais de uma classe ou vazio"
        )
    return result


if __name__ == "__main__":
    main()
