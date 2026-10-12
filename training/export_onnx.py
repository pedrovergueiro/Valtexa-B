"""Exporta um modelo treinado para ONNX, com paridade verificada.

::

    python -m training.export_onnx --model leaf-foliar
    python -m training.export_onnx --severity
    python -m training.export_onnx --framing

Grava ``models/<pesos>.onnx`` ao lado do ``.keras``. Nos classificadores o grafo
sai com **duas saídas** — probabilidades e embedding —, para a novidade
continuar vindo da mesma passada.

Antes de gravar, roda as mesmas imagens reais pelos dois caminhos e compara.
Um grafo exportado que responde diferente é o pior tipo de erro: não quebra
nada, só troca diagnósticos em silêncio. Se a diferença passar de
``--tolerance``, o arquivo é apagado e o comando falha.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from app.config import settings
from app.inference.preprocess import load_image_from_bytes, preprocess_batch
from app.keras_backend import configure, no_grad, to_numpy
from app.model import EMBEDDING_LAYER
from app.onnx_runtime import OnnxModel, onnx_path
from app.registry import MODEL_SPECS, get_spec
from app.severity import WEIGHTS_NAME as SEVERITY_WEIGHTS
from app.severity import lesion_fraction
from training import DEFAULT_SEED
from training.prepare_dataset import is_image

TOLERANCE = 1e-4
"""Maior diferença aceita numa probabilidade (ou num pixel da máscara)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta um modelo treinado para ONNX, com paridade verificada",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    target.add_argument("--severity", action="store_true", help="Exporta o modelo de severidade.")
    target.add_argument(
        "--framing",
        action="store_true",
        help="Exporta a torre de visão do portão de enquadramento (precisa do framing.npz).",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Padrão: os do catálogo.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Imagens para a paridade.")
    parser.add_argument("--check", type=int, default=256, help="Imagens reais na paridade.")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.framing:
        export_framing(args)
        return
    if args.severity:
        weights = args.weights or settings.models_dir / SEVERITY_WEIGHTS
        data_dir = args.data_dir or get_spec("leaf-foliar").data_dir
        img_size = 224
    else:
        spec = get_spec(args.model)
        weights = args.weights or spec.weights_path
        data_dir = args.data_dir or spec.data_dir
        img_size = spec.img_size
    if not weights.exists():
        raise SystemExit(f"Pesos não encontrados: {weights}")

    configure()
    import keras

    model = keras.models.load_model(weights, compile=False)
    graph = _with_embedding(keras, model) if not args.severity else model
    img_size = _input_size(model) or img_size
    graph(np.zeros((1, img_size, img_size, 3), np.float32), training=False)  # o export exige

    output = onnx_path(weights)
    print(f"\n=== Exportando para ONNX — {weights.name} ===")
    graph.export(str(output), format="onnx", verbose=False)
    print(f"Grafo: {output} ({output.stat().st_size / 1e6:.1f} MB)")

    _exact_float32()
    batch = _real_batch(data_dir, args.check, img_size)
    session = OnnxModel(output, providers=["CPUExecutionProvider"])
    keras_out, onnx_out = _both_paths(graph, session, batch)

    # A primeira saída (probabilidades, máscara) vive em [0, 1]: diferença
    # absoluta. O embedding tem outra escala: diferença relativa ao maior valor
    # dele — 1e-4 absoluto num vetor de magnitude 10 é ruído de fp32 entre GPU e CPU.
    worst = float(np.abs(keras_out[0] - onnx_out[0]).max())
    for extra_keras, extra_onnx in zip(keras_out[1:], onnx_out[1:], strict=True):
        scale = float(np.abs(extra_keras).max()) or 1.0
        worst = max(worst, float(np.abs(extra_keras - extra_onnx).max()) / scale)
    print(f"\nParidade em {len(batch)} imagens reais:")
    print(f"  maior diferença       {worst:.2e}  (tolerância {args.tolerance:.0e})")
    if args.severity:
        gap = np.abs(np.array(lesion_fraction(keras_out[0])) - lesion_fraction(onnx_out[0]))
        print(f"  severidade, maior Δ   {gap.max():.4f} do quadro")
    else:
        same = (keras_out[0].argmax(1) == onnx_out[0].argmax(1)).mean()
        print(f"  mesma classe          {same:.2%}")
    if worst > args.tolerance:
        output.unlink()
        raise SystemExit("Paridade não se sustentou — o .onnx foi apagado.")

    print(f"\nLatência, 1 imagem, ONNX Runtime em CPU: {_latency(session, batch[:1]):.1f} ms")
    print("Sirva com INFERENCE_RUNTIME=onnx.")


def export_framing(args: argparse.Namespace) -> None:
    """A torre de visão do SigLIP em ONNX, conferida contra o ``transformers``."""
    import torch
    from transformers import SiglipVisionModel

    from app.framing import (
        IMG_SIZE,
        REFERENCE_NAME,
        VISION_ONNX_NAME,
        FramingReference,
        judge,
        onnx_vision_encoder,
        vision_encoder,
    )

    reference = FramingReference.load(settings.models_dir / REFERENCE_NAME)
    output = settings.models_dir / VISION_ONNX_NAME
    tower = SiglipVisionModel.from_pretrained(reference.model_id).eval()

    class Pooled(torch.nn.Module):
        """Só o embedding final — é tudo que o portão consome."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).pooler_output

    print(f"\n=== Exportando a torre de visão — {reference.model_id} ===")
    torch.onnx.export(
        Pooled(tower),
        (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),),
        str(output),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Grafo: {output} ({output.stat().st_size / 1e6:.1f} MB)")

    from PIL import Image

    data_dir = args.data_dir or get_spec("leaf-foliar").data_dir
    files = sorted(path for path in data_dir.rglob("*") if is_image(path))
    rng = np.random.default_rng(DEFAULT_SEED)
    chosen = [files[i] for i in rng.choice(len(files), min(args.check, len(files)), replace=False)]
    images = []
    for path in chosen:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    by_transformers, by_onnx = vision_encoder(reference.model_id), onnx_vision_encoder(output)
    reference_path = np.vstack(
        [by_transformers(images[i : i + 32]) for i in range(0, len(images), 32)]
    )
    onnx_path_ = np.vstack([by_onnx(images[i : i + 32]) for i in range(0, len(images), 32)])
    worst = float(np.abs(reference_path - onnx_path_).max())
    same = np.mean(
        [
            a.in_scope == b.in_scope
            for a, b in zip(
                judge(reference, reference_path), judge(reference, onnx_path_), strict=True
            )
        ]
    )
    print(f"\nParidade em {len(images)} imagens reais (transformers x ONNX + preprocess da API):")
    print(f"  maior diferença no embedding   {worst:.2e}")
    print(f"  mesmo veredito                 {same:.2%}")
    if same < 1.0:
        output.unlink()
        raise SystemExit("Vereditos divergiram — o .onnx foi apagado.")
    print("Sirva com INFERENCE_RUNTIME=onnx: o portão deixa de precisar de torch.")


def _exact_float32() -> None:
    """Desliga o TF32 do torch para a paridade comparar fp32 com fp32.

    Medido em 26/09/2026: com o TF32 que o torch liga por padrão nas convoluções
    em GPU, a MobileNetV2 difere do próprio ONNX (fp32, em CPU) em até 0,11 numa
    probabilidade — sem mudar nenhuma classe em 256 imagens. A imprecisão estava
    na referência, não no grafo exportado. A comparação justa é fp32 dos dois lados.
    """
    try:
        import torch
    except ImportError:
        return
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _both_paths(graph, session: OnnxModel, batch: np.ndarray, chunk: int = 32):
    """As saídas do Keras e do ONNX Runtime para o mesmo lote, em pedaços."""
    keras_parts, onnx_parts = [], []
    for start in range(0, len(batch), chunk):
        piece = batch[start : start + chunk]
        with no_grad():
            keras_parts.append([to_numpy(x) for x in _as_list(graph(piece, training=False))])
        onnx_parts.append(session.run(piece))
    keras_out = [np.concatenate(outs) for outs in zip(*keras_parts, strict=True)]
    onnx_out = [np.concatenate(outs) for outs in zip(*onnx_parts, strict=True)]
    return keras_out, onnx_out


def _with_embedding(keras, model):
    """O modelo de duas saídas, quando ele tem a camada de embedding."""
    try:
        embedding = model.get_layer(EMBEDDING_LAYER).output
    except ValueError:
        print(f"Sem camada {EMBEDDING_LAYER!r}: o grafo sai só com as probabilidades.")
        return model
    return keras.Model(model.inputs[0], [model.output, embedding])


def _input_size(model) -> int | None:
    shape = getattr(model, "input_shape", None)
    return int(shape[1]) if shape and shape[1] else None


def _as_list(outputs) -> list:
    return list(outputs) if isinstance(outputs, list | tuple) else [outputs]


def _real_batch(data_dir: Path, count: int, img_size: int) -> np.ndarray:
    """Imagens de verdade, sorteadas do dataset — ruído aleatório não testa o que importa."""
    files = sorted(path for path in data_dir.rglob("*") if is_image(path))
    if not files:
        raise SystemExit(f"Nenhuma imagem em {data_dir} para verificar a paridade.")
    rng = np.random.default_rng(DEFAULT_SEED)
    chosen = [files[i] for i in rng.choice(len(files), min(count, len(files)), replace=False)]
    images = [load_image_from_bytes(path.read_bytes(), path.name) for path in chosen]
    return preprocess_batch(images, img_size).astype(np.float32)


def _latency(session: OnnxModel, batch: np.ndarray, runs: int = 30) -> float:
    for _ in range(3):
        session.run(batch)
    start = time.perf_counter()
    for _ in range(runs):
        session.run(batch)
    return (time.perf_counter() - start) / runs * 1000


if __name__ == "__main__":
    main()
