"""Etapa de otimização do pipeline do Model Registry (brief, seção 23).

    MODELO TREINADO → VALIDAÇÃO → BENCHMARK → dNATY → MODELO OTIMIZADO →
    NOVO BENCHMARK → VALIDAÇÃO → APROVAÇÃO → PUBLICAÇÃO → ROVER

Este módulo faz o miolo: **benchmark do original → otimização → benchmark do
otimizado → comparação com um critério de aceite configurável**, e devolve um
relatório que o registry da plataforma grava (``POST
/v1/platform/models/{id}/optimizations``) com o original e o otimizado
**separados**. A aprovação continua sendo de uma pessoa de IA & Dados, no
registry; aqui só se mede.

Compressão não vale só porque ficou menor (seção 23): o critério exige que a
qualidade (F1 macro, precisão, recall e o recall **de cada classe**) não caia
além da tolerância **e** que haja ganho real de eficiência (latência, memória,
tamanho ou custo computacional). Um otimizado menor e mais lento, ou menor e
cego para uma doença, é recusado.

Os dois artefatos são medidos do mesmo jeito: pelo **ONNX Runtime**, o mesmo
que roda no robô (``rover/valtexa_rover/edge_ai``), sobre o mesmo conjunto de
validação, com os mesmos *providers*.

Otimizadores (:class:`Otimizador`):

* :class:`DnatyOtimizador` — o **dNATY** (``pip install dnaty==2.2.1``): busca
  evolutiva de arquitetura sobre um ``torch.nn.Module`` e exporta o resultado em
  ONNX. **Licença BSL 1.1**: uso não comercial liberado; **uso comercial pede
  licença própria do autor** — antes de publicar um modelo otimizado pelo dNATY
  para cliente pagante, a ValTexa precisa dessa licença;
* :class:`QuantizacaoDinamicaOnnx` — quantização dinâmica INT8 do ONNX Runtime,
  sem dNATY (útil para modelos que só existem em ONNX). Grava ``dnaty_version``
  nulo: não é dNATY, e o registry não deve dizer que é.

Uso::

    python -m training.dnaty_pipeline --original models/leaf.onnx \\
        --validacao val.npz --otimizador quantizacao --saida relatorio.json

``val.npz`` traz ``x`` (entrada já pré-processada, como o modelo recebe), ``y``
(índice da classe) e ``labels``. **Sem os pesos reais nesta nuvem**, os testes
rodam com modelos minúsculos; os números de um modelo de verdade: a verificar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from training.metrics import confusion_matrix, macro_f1, per_class_report

LICENCA_DNATY = (
    "dNATY é licenciado sob BSL 1.1: uso não comercial liberado; uso comercial "
    "(modelo otimizado servindo cliente pagante) pede licença própria do autor."
)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class Benchmark:
    """Medidas de um artefato ONNX sobre o conjunto de validação."""

    sha256: str
    size_bytes: int
    amostras: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    por_classe: list[dict[str, Any]]
    latency_ms_p50: float
    latency_ms_p95: float
    fps: float
    memory_mb: float | None
    flops: int | None
    providers: list[str]
    hardware: str
    map: float | None = None  # só para detecção; os modelos de hoje são de classificação
    notas: list[str] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloco in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def _rss_mb() -> float | None:
    """Memória residente do processo (Linux). Fora do Linux, ``None`` — não inventa."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            paginas = int(fh.read().split()[1])
    except OSError:
        return None
    return paginas * os.sysconf("SC_PAGE_SIZE") / 2**20


def estimate_flops(path: Path) -> int | None:
    """Custo computacional por amostra: 2·MACs de ``MatMul``/``Gemm``/``Conv``.

    Lê o grafo com inferência de formas do ``onnx``. Dimensões simbólicas
    (``batch``) contam como 1. Operadores que não somam MACs (ativação, soma,
    *pooling*) ficam de fora — é a parte que pesa no custo. ``None`` se o pacote
    ``onnx`` não estiver instalado ou as formas não se resolverem.
    """
    try:
        import onnx
        from onnx import numpy_helper, shape_inference
    except ImportError:  # pragma: no cover - onnx está no requirements-dev
        return None
    modelo = shape_inference.infer_shapes(onnx.load(str(path)))
    formas: dict[str, list[int]] = {}

    def guarda(info) -> None:
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in info.type.tensor_type.shape.dim]
        formas[info.name] = dims

    for info in [*modelo.graph.input, *modelo.graph.value_info, *modelo.graph.output]:
        guarda(info)
    for init in modelo.graph.initializer:
        formas[init.name] = list(numpy_helper.to_array(init).shape)

    total = 0
    for no in modelo.graph.node:
        if no.op_type in ("MatMul", "MatMulInteger", "Gemm"):
            a = formas.get(no.input[0])
            b = formas.get(no.input[1])
            if not a or not b:
                return None
            k = a[-1]
            n = b[0] if no.op_type == "Gemm" and _attr(no, "transB", 0) else b[-1]
            linhas = int(np.prod(a[:-1])) if len(a) > 1 else 1
            total += 2 * linhas * k * n
        elif no.op_type in ("Conv", "ConvInteger"):
            peso = formas.get(no.input[1])
            saida = formas.get(no.output[0])
            if not peso or not saida:
                return None
            # peso: [C_out, C_in/grupo, kH, kW]; saída: [N, C_out, H, W]
            total += 2 * int(np.prod(saida)) * int(np.prod(peso[1:]))
    return total


def _attr(no, nome: str, padrao: int) -> int:
    for a in no.attribute:
        if a.name == nome:
            return int(a.i)
    return padrao


def benchmark_onnx(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    *,
    providers: Sequence[str] | None = None,
    warmup: int = 3,
    latency_samples: int = 50,
) -> Benchmark:
    """Mede um ``.onnx``: qualidade no conjunto inteiro, latência amostra a amostra.

    A latência é de **uma amostra por vez** (como o robô processa um quadro), com
    ``warmup`` passadas antes; FPS = 1000 / p50. Memória: residente a mais depois
    de carregar e rodar (``None`` fora do Linux).
    """
    import onnxruntime as ort

    disponiveis = ort.get_available_providers()
    escolhidos = [p for p in (providers or ["CPUExecutionProvider"]) if p in disponiveis]
    antes = _rss_mb()
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = 1  # medida estável e comparável entre os dois
    sessao = ort.InferenceSession(str(path), opcoes, providers=escolhidos or disponiveis)
    entrada = sessao.get_inputs()[0].name

    def roda(lote: np.ndarray) -> np.ndarray:
        return sessao.run(None, {entrada: np.ascontiguousarray(lote, np.float32)})[0]

    previsoes = []
    for inicio in range(0, len(x), 64):
        previsoes.append(np.argmax(roda(x[inicio:inicio + 64]), axis=-1))
    y_pred = np.concatenate(previsoes) if previsoes else np.zeros(0, dtype=int)
    depois = _rss_mb()

    for i in range(warmup):
        roda(x[i % len(x)][None])
    tempos = []
    for i in range(min(latency_samples, max(1, len(x)))):
        t0 = time.perf_counter()
        roda(x[i % len(x)][None])
        tempos.append((time.perf_counter() - t0) * 1000)
    p50 = float(np.percentile(tempos, 50))

    matriz = confusion_matrix(np.asarray(y), y_pred, len(labels))
    relatorio = per_class_report(matriz, labels)
    acertos = float(np.trace(matriz)) / max(1, int(matriz.sum()))
    return Benchmark(
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        amostras=len(x),
        accuracy=round(acertos, 6),
        precision=round(float(np.mean([r["precision"] for r in relatorio])), 6),
        recall=round(float(np.mean([r["recall"] for r in relatorio])), 6),
        f1=round(macro_f1(relatorio), 6),
        por_classe=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in relatorio],
        latency_ms_p50=round(p50, 4),
        latency_ms_p95=round(float(np.percentile(tempos, 95)), 4),
        fps=round(1000.0 / p50, 2) if p50 > 0 else 0.0,
        memory_mb=None if antes is None or depois is None else round(max(0.0, depois - antes), 3),
        flops=estimate_flops(path),
        providers=sessao.get_providers(),
        hardware=f"{platform.machine()} {platform.processor() or platform.system()}".strip(),
    )


# ---------------------------------------------------------------------------
# Critério de aceite
# ---------------------------------------------------------------------------


@dataclass
class CriterioDeAceite:
    """O que o otimizado precisa cumprir. Configurável por modelo.

    Quedas são em pontos absolutos (0,01 = 1 p.p.). ``ganho_minimo`` é a fração
    mínima de melhora em **pelo menos uma** das métricas de eficiência listadas
    em ``ganhos_validos``. ``latencia_max_relativa`` impede que fique mais lento
    que isso (1,10 = até 10 % mais lento, se o ganho veio de outro lado).
    """

    max_queda_f1: float = 0.01
    max_queda_precisao: float = 0.02
    max_queda_recall: float = 0.02
    max_queda_recall_por_classe: float = 0.03
    latencia_max_relativa: float = 1.10
    ganho_minimo: float = 0.10
    ganhos_validos: tuple[str, ...] = ("latency_ms_p50", "memory_mb", "size_bytes", "flops")
    ruido_latencia_ms: float = 0.5
    """Diferença de latência menor que isto é ruído de medição (CPU disputada): não
    conta como ganho nem como perda. Modelo de borda real roda em milissegundos."""
    ruido_memoria_mb: float = 16.0
    """Idem para a memória, medida pela residente a mais do processo — a primeira
    sessão do ONNX Runtime paga a inicialização da biblioteca e a segunda não."""

    @classmethod
    def from_dict(cls, dados: dict | None) -> CriterioDeAceite:
        dados = dict(dados or {})
        if "ganhos_validos" in dados:
            dados["ganhos_validos"] = tuple(dados["ganhos_validos"])
        return cls(**dados)


def comparar(original: dict, otimizado: dict) -> dict[str, Any]:
    """Diferenças otimizado - original (qualidade) e razão otimizado / original (eficiência)."""
    saida: dict[str, Any] = {}
    for chave in ("accuracy", "precision", "recall", "f1"):
        saida[f"delta_{chave}"] = round(otimizado[chave] - original[chave], 6)
    for chave in ("latency_ms_p50", "latency_ms_p95", "memory_mb", "size_bytes", "flops", "fps"):
        a, b = original.get(chave), otimizado.get(chave)
        saida[f"razao_{chave}"] = round(b / a, 6) if a and b is not None else None
    por_classe = {}
    for o, n in zip(original.get("por_classe", []), otimizado.get("por_classe", []), strict=False):
        por_classe[o["class_name"]] = round(n["recall"] - o["recall"], 6)
    saida["delta_recall_por_classe"] = por_classe
    return saida


def _diferenca(a: dict, b: dict, chave: str) -> float:
    """``a[chave] - b[chave]`` (0 se faltar algum dos dois)."""
    if a.get(chave) is None or b.get(chave) is None:
        return 0.0
    return float(a[chave]) - float(b[chave])


def _ruido(criterio: CriterioDeAceite, chave: str) -> float:
    if chave.startswith("latency_ms"):
        return criterio.ruido_latencia_ms
    if chave == "memory_mb":
        return criterio.ruido_memoria_mb
    return 0.0


def avaliar(original: dict, otimizado: dict, criterio: CriterioDeAceite) -> dict[str, Any]:
    """Aceita ou recusa, com cada motivo em português. Não aceita só porque ficou menor."""
    c = comparar(original, otimizado)
    motivos: list[str] = []
    if c["delta_f1"] < -criterio.max_queda_f1:
        motivos.append(f"F1 caiu {-c['delta_f1']:.4f} (máximo {criterio.max_queda_f1}).")
    if c["delta_precision"] < -criterio.max_queda_precisao:
        motivos.append(f"Precisão caiu {-c['delta_precision']:.4f} "
                       f"(máximo {criterio.max_queda_precisao}).")
    if c["delta_recall"] < -criterio.max_queda_recall:
        motivos.append(f"Recall caiu {-c['delta_recall']:.4f} "
                       f"(máximo {criterio.max_queda_recall}).")
    for classe, delta in c["delta_recall_por_classe"].items():
        if delta < -criterio.max_queda_recall_por_classe:
            motivos.append(f"Recall de {classe} caiu {-delta:.4f} "
                           f"(máximo {criterio.max_queda_recall_por_classe} por classe).")
    razao_lat = c.get("razao_latency_ms_p50")
    if razao_lat is not None and razao_lat > criterio.latencia_max_relativa \
            and _diferenca(otimizado, original, "latency_ms_p50") >= criterio.ruido_latencia_ms:
        motivos.append(f"Latência ficou {razao_lat:.2f}x a do original "
                       f"(máximo {criterio.latencia_max_relativa}x).")
    ganhos = {}
    for chave in criterio.ganhos_validos:
        razao = c.get(f"razao_{chave}")
        if razao is None:
            continue
        piso = _ruido(criterio, chave)
        if razao < 1 and _diferenca(original, otimizado, chave) < piso:
            razao = 1.0  # melhora dentro do ruído de medição não é ganho
        ganhos[chave] = round(1 - razao, 6)
    melhor = max(ganhos.values(), default=0.0)
    if melhor < criterio.ganho_minimo:
        motivos.append(f"Sem ganho de eficiência de ao menos {criterio.ganho_minimo:.0%} "
                       f"(melhor: {melhor:.1%}).")
    return {"aceito": not motivos, "motivos": motivos, "ganhos": ganhos, "comparacao": c,
            "criterio": {**asdict(criterio), "ganhos_validos": list(criterio.ganhos_validos)}}


# ---------------------------------------------------------------------------
# Otimizadores
# ---------------------------------------------------------------------------


@dataclass
class EntradaOtimizacao:
    original_onnx: Path
    destino: Path
    torch_model: Any = None
    """O ``nn.Module`` treinado — o dNATY otimiza o modelo, não o grafo ONNX."""
    x_treino: np.ndarray | None = None
    y_treino: np.ndarray | None = None
    input_shape: tuple[int, ...] | None = None


@dataclass
class SaidaOtimizacao:
    path: Path
    ferramenta: str
    versao: str
    dnaty_version: str | None
    parametros: dict[str, Any]


class Otimizador(Protocol):
    nome: str

    def otimizar(self, entrada: EntradaOtimizacao) -> SaidaOtimizacao: ...


class OtimizadorIndisponivelError(RuntimeError):
    """A ferramenta não está instalada ou não serve para este modelo. A mensagem diz por quê."""


class QuantizacaoDinamicaOnnx:
    """INT8 dinâmico do ONNX Runtime (pesos de ``MatMul``/``Gemm`` em 8 bits). Não é dNATY."""

    nome = "quantizacao"

    def otimizar(self, entrada: EntradaOtimizacao) -> SaidaOtimizacao:
        import onnxruntime
        from onnxruntime.quantization import QuantType, quantize_dynamic

        entrada.destino.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(str(entrada.original_onnx), str(entrada.destino),
                         weight_type=QuantType.QInt8)
        return SaidaOtimizacao(entrada.destino, "onnxruntime.quantize_dynamic",
                               onnxruntime.__version__, None, {"weight_type": "QInt8"})


class DnatyOtimizador:
    """O dNATY (BSL 1.1 — ver :data:`LICENCA_DNATY`) sobre o ``nn.Module`` treinado.

    ``dnaty.compress`` busca, por evolução, uma arquitetura menor com ``target_flops``
    do custo original, faz o ajuste fino e exporta ONNX. Precisa do modelo torch e
    dos dados de treino (não otimiza um ONNX já exportado).
    """

    nome = "dnaty"

    def __init__(self, target_flops: float = 0.5, n_generations: int = 30, n_pop: int = 15,
                 finetune_epochs: int = 30, seed: int | None = 0) -> None:
        self.parametros = {"target_flops": target_flops, "n_generations": n_generations,
                           "n_pop": n_pop, "finetune_epochs": finetune_epochs, "seed": seed}

    def otimizar(self, entrada: EntradaOtimizacao) -> SaidaOtimizacao:
        try:
            import dnaty
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise OtimizadorIndisponivelError(
                "dNATY não instalado (pip install --no-deps dnaty==2.2.1 scipy)."
            ) from exc
        if entrada.torch_model is None or entrada.x_treino is None or entrada.y_treino is None:
            raise OtimizadorIndisponivelError(
                "O dNATY otimiza o nn.Module treinado com os dados de treino; só o ONNX "
                "não basta. Use --otimizador quantizacao para um modelo que só existe em ONNX."
            )
        if entrada.input_shape is None:
            raise OtimizadorIndisponivelError(
                "Informe input_shape (sem o lote) para exportar o ONNX."
            )
        dados = DataLoader(
            TensorDataset(torch.as_tensor(entrada.x_treino, dtype=torch.float32),
                          torch.as_tensor(entrada.y_treino, dtype=torch.long)),
            batch_size=64, shuffle=True,
        )
        resultado = dnaty.compress(entrada.torch_model, dados, device="cpu", verbose=False,
                                   **self.parametros)
        entrada.destino.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            # o dNATY exporta pelo exportador TorchScript (dynamo=False), que o torch
            # 2.9+ marca como obsoleto; o ONNX gerado é conferido no benchmark.
            warnings.simplefilter("ignore", DeprecationWarning)
            resultado.export_onnx(str(entrada.destino), input_shape=entrada.input_shape)
        return SaidaOtimizacao(entrada.destino, "dnaty", dnaty.__version__, dnaty.__version__,
                               dict(self.parametros))


# ---------------------------------------------------------------------------
# A etapa inteira
# ---------------------------------------------------------------------------


def executar(
    entrada: EntradaOtimizacao,
    otimizador: Otimizador,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: Sequence[str],
    *,
    criterio: CriterioDeAceite | None = None,
    providers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Benchmark → otimização → benchmark → comparação. O original nunca é tocado."""
    criterio = criterio or CriterioDeAceite()
    sha_antes = file_sha256(entrada.original_onnx)
    original = asdict(benchmark_onnx(entrada.original_onnx, x_val, y_val, labels,
                                     providers=providers))
    saida = otimizador.otimizar(entrada)
    if saida.path.resolve() == entrada.original_onnx.resolve():
        raise ValueError("O otimizado precisa ser um arquivo separado do original (seção 23).")
    if file_sha256(entrada.original_onnx) != sha_antes:
        raise RuntimeError("O otimizador alterou o artefato original: abortado.")
    otimizado = asdict(benchmark_onnx(saida.path, x_val, y_val, labels, providers=providers))
    veredito = avaliar(original, otimizado, criterio)
    return {
        "tool": saida.ferramenta,
        "tool_version": saida.versao,
        "dnaty_version": saida.dnaty_version,
        "parametros": saida.parametros,
        "licenca": LICENCA_DNATY if saida.dnaty_version else None,
        "original": {"path": str(entrada.original_onnx), "benchmark": original},
        "otimizado": {"path": str(saida.path), "benchmark": otimizado},
        **veredito,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--validacao", type=Path, required=True, help=".npz com x, y e labels")
    p.add_argument("--otimizador", choices=["quantizacao"], default="quantizacao",
                   help="dNATY pede o nn.Module: use executar() a partir do treino")
    p.add_argument("--destino", type=Path)
    p.add_argument("--criterio", type=Path, help="JSON com os campos de CriterioDeAceite")
    p.add_argument("--saida", type=Path, required=True)
    args = p.parse_args(argv)
    val = np.load(args.validacao, allow_pickle=False)
    destino = args.destino or args.original.with_name(args.original.stem + ".otimizado.onnx")
    criterio = CriterioDeAceite.from_dict(
        json.loads(args.criterio.read_text(encoding="utf-8")) if args.criterio else None
    )
    relatorio = executar(EntradaOtimizacao(args.original, destino), QuantizacaoDinamicaOnnx(),
                         val["x"], val["y"], [str(r) for r in val["labels"]], criterio=criterio)
    args.saida.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(("ACEITO" if relatorio["aceito"] else "RECUSADO") + ": "
          + ("; ".join(relatorio["motivos"]) or "dentro do critério"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = ["CriterioDeAceite", "DnatyOtimizador", "QuantizacaoDinamicaOnnx", "avaliar",
           "benchmark_onnx", "comparar", "executar"]
