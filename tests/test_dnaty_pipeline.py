"""Etapa dNATY do registry: benchmark → otimização → benchmark → critério de aceite.

Sem pesos reais aqui: os modelos são grafos ONNX pequenos montados na hora (e, no
teste do dNATY, um MLP torch minúsculo). Os números de um modelo de verdade
ficam "a verificar com o modelo real".
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from training.dnaty_pipeline import (
    CriterioDeAceite,
    DnatyOtimizador,
    EntradaOtimizacao,
    QuantizacaoDinamicaOnnx,
    SaidaOtimizacao,
    avaliar,
    benchmark_onnx,
    estimate_flops,
    executar,
    file_sha256,
)

LABELS = ["Healthy", "Rust", "Miner", "Phoma"]
D, H = 32, 512


def _dados(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(D, len(LABELS))).astype(np.float32)
    x = rng.normal(size=(n, D)).astype(np.float32)
    y = np.argmax(x @ w, axis=1)
    return x, y, w


def _onnx_mlp(path: Path, w: np.ndarray, hidden: int = H, seed: int = 1) -> Path:
    """``x @ W1 @ W2`` com ``W1 @ W2 == w``: acerta tudo, mas custa 2·D·H + 2·H·C FLOPs."""
    from onnx import TensorProto, helper

    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=(D, hidden)).astype(np.float32)
    w2 = (np.linalg.pinv(w1) @ w).astype(np.float32)
    grafo = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W1"], ["h"]),
         helper.make_node("MatMul", ["h", "W2"], ["logits"]),
         helper.make_node("Softmax", ["logits"], ["probs"], axis=1)],
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", D])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", len(LABELS)])],
        initializer=[helper.make_tensor("W1", TensorProto.FLOAT, w1.shape, w1.flatten()),
                     helper.make_tensor("W2", TensorProto.FLOAT, w2.shape, w2.flatten())],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    modelo.ir_version = 8
    path.write_bytes(modelo.SerializeToString())
    return path


def _onnx_linear(path: Path, w: np.ndarray) -> Path:
    """Um grafo direto ``x @ w`` (barato)."""
    import onnx
    from onnx import TensorProto, helper

    grafo = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["logits"]),
         helper.make_node("Softmax", ["logits"], ["probs"], axis=1)],
        "linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", D])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", len(LABELS)])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.flatten())],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    modelo.ir_version = 8
    onnx.checker.check_model(modelo)
    path.write_bytes(modelo.SerializeToString())
    return path


class _Fixo:
    """Otimizador de teste: devolve o grafo que o teste preparou (simula uma ferramenta)."""

    nome = "fixo"

    def __init__(self, fonte: Path) -> None:
        self.fonte = fonte

    def otimizar(self, entrada: EntradaOtimizacao) -> SaidaOtimizacao:
        shutil.copyfile(self.fonte, entrada.destino)
        return SaidaOtimizacao(entrada.destino, "teste", "0", None, {})


def test_benchmark_mede_qualidade_custo_e_tempo(tmp_path: Path) -> None:
    x, y, w = _dados()
    b = benchmark_onnx(_onnx_mlp(tmp_path / "o.onnx", w), x, y, LABELS)
    assert b.f1 > 0.99 and b.precision > 0.99 and b.recall > 0.99 and b.accuracy > 0.99
    assert b.flops == 2 * D * H + 2 * H * len(LABELS)
    assert b.size_bytes == (tmp_path / "o.onnx").stat().st_size
    assert b.latency_ms_p50 > 0 and b.latency_ms_p95 >= b.latency_ms_p50 and b.fps > 0
    assert len(b.por_classe) == len(LABELS) and b.amostras == len(x)
    assert b.sha256 == file_sha256(tmp_path / "o.onnx")
    assert b.memory_mb is None or b.memory_mb >= 0


def test_quantizacao_real_aceita_com_original_e_otimizado_separados(tmp_path: Path) -> None:
    x, y, w = _dados()
    original = _onnx_mlp(tmp_path / "original.onnx", w)
    sha = file_sha256(original)
    # Latência medida em CPU compartilhada oscila (já deu 1,23x com outro processo
    # pesado rodando): aqui o ponto é tamanho + qualidade. A regra de latência tem
    # teste próprio, com números fixos (test_criterio_configuravel_e_latencia).
    rel = executar(EntradaOtimizacao(original, tmp_path / "otimizado.onnx"),
                   QuantizacaoDinamicaOnnx(), x, y, LABELS,
                   criterio=CriterioDeAceite(latencia_max_relativa=10.0))
    assert file_sha256(original) == sha, "o original não pode mudar"
    o, n = rel["original"]["benchmark"], rel["otimizado"]["benchmark"]
    assert o["sha256"] != n["sha256"] and Path(rel["otimizado"]["path"]).exists()
    assert n["size_bytes"] < 0.5 * o["size_bytes"]  # INT8: ~4x menor
    assert rel["aceito"], rel["motivos"]
    assert rel["ganhos"]["size_bytes"] > 0.5
    assert rel["dnaty_version"] is None and rel["licenca"] is None  # não é dNATY
    assert rel["tool"] == "onnxruntime.quantize_dynamic"


def test_menor_mas_pior_e_recusado(tmp_path: Path) -> None:
    """Seção 23: compressão não vale só porque ficou menor."""
    x, y, w = _dados()
    original = _onnx_mlp(tmp_path / "original.onnx", w)
    ruim = w.copy()
    ruim[:, 3] = -10.0  # o "otimizado" nunca mais vê a classe Phoma
    pior = _onnx_linear(tmp_path / "pior.onnx", ruim)
    rel = executar(EntradaOtimizacao(original, tmp_path / "otimizado.onnx"), _Fixo(pior),
                   x, y, LABELS)
    assert rel["otimizado"]["benchmark"]["size_bytes"] < rel["original"]["benchmark"]["size_bytes"]
    assert not rel["aceito"]
    assert any("Recall de Phoma" in m for m in rel["motivos"])
    assert any("F1 caiu" in m for m in rel["motivos"])


def test_igual_ao_original_e_recusado_por_falta_de_ganho(tmp_path: Path) -> None:
    x, y, w = _dados()
    original = _onnx_mlp(tmp_path / "original.onnx", w)
    rel = executar(EntradaOtimizacao(original, tmp_path / "otimizado.onnx"), _Fixo(original),
                   x, y, LABELS)
    assert not rel["aceito"]
    assert any("Sem ganho de eficiência" in m for m in rel["motivos"])


def test_criterio_configuravel_e_latencia() -> None:
    base = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9,
            "latency_ms_p50": 10.0, "latency_ms_p95": 12.0, "memory_mb": 50.0,
            "size_bytes": 1000, "flops": 1000, "fps": 100.0,
            "por_classe": [{"class_name": "A", "recall": 0.9}]}
    menor_lento = {**base, "size_bytes": 300, "latency_ms_p50": 15.0, "f1": 0.895}
    v = avaliar(base, menor_lento, CriterioDeAceite())
    assert not v["aceito"] and any("Latência" in m for m in v["motivos"])
    frouxo = CriterioDeAceite(latencia_max_relativa=2.0)
    assert avaliar(base, menor_lento, frouxo)["aceito"]
    # só o tamanho como ganho válido, e exigindo 80 %: 70 % não basta
    so_tamanho = CriterioDeAceite.from_dict({"ganhos_validos": ["size_bytes"],
                                             "ganho_minimo": 0.8, "latencia_max_relativa": 2})
    assert not avaliar(base, menor_lento, so_tamanho)["aceito"]


def test_diferenca_dentro_do_ruido_de_medicao_nao_e_ganho() -> None:
    """O mesmo modelo medido duas vezes: a latência de um MLP minúsculo oscila ±25 %
    e a memória da primeira sessão inclui a inicialização do ONNX Runtime. Nenhuma
    das duas pode virar "ganho" (nem "perda") — era o que fazia o idêntico passar."""
    base = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9,
            "latency_ms_p50": 0.05, "latency_ms_p95": 0.08, "memory_mb": 8.8,
            "size_bytes": 1000, "flops": 1000, "fps": 20000.0,
            "por_classe": [{"class_name": "A", "recall": 0.9}]}
    mesmo = {**base, "latency_ms_p50": 0.04, "memory_mb": 0.0}
    v = avaliar(base, mesmo, CriterioDeAceite())
    assert not v["aceito"] and v["ganhos"]["latency_ms_p50"] == 0
    assert v["ganhos"]["memory_mb"] == 0
    assert not any("Latência" in m for m in v["motivos"])
    # 0,07 ms mais lento também é ruído, não reprovação por latência
    assert not any("Latência" in m for m in avaliar(
        base, {**base, "latency_ms_p50": 0.12}, CriterioDeAceite())["motivos"])
    # com ruído zero volta o comportamento estrito (para hardware dedicado e estável)
    estrito = CriterioDeAceite(ruido_latencia_ms=0, ruido_memoria_mb=0)
    assert avaliar(base, mesmo, estrito)["aceito"]


def test_otimizador_que_mexe_no_original_e_abortado(tmp_path: Path) -> None:
    x, y, w = _dados(n=40)
    original = _onnx_mlp(tmp_path / "original.onnx", w)

    class Estraga:
        nome = "estraga"

        def otimizar(self, entrada):
            entrada.original_onnx.write_bytes(entrada.original_onnx.read_bytes() + b"\0")
            shutil.copyfile(entrada.original_onnx, entrada.destino)
            return SaidaOtimizacao(entrada.destino, "x", "0", None, {})

    with pytest.raises(RuntimeError, match="alterou o artefato original"):
        executar(EntradaOtimizacao(original, tmp_path / "o2.onnx"), Estraga(), x, y, LABELS)


def test_flops_de_conv(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    peso = np.zeros((8, 3, 3, 3), dtype=np.float32)
    grafo = helper.make_graph(
        [helper.make_node("Conv", ["x", "W"], ["y"], pads=[1, 1, 1, 1])], "c",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, peso.shape, peso.flatten())],
    )
    m = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    (tmp_path / "c.onnx").write_bytes(m.SerializeToString())
    assert onnx.load(str(tmp_path / "c.onnx"))
    assert estimate_flops(tmp_path / "c.onnx") == 2 * (8 * 16 * 16) * (3 * 3 * 3)


def test_dnaty_de_verdade_otimiza_e_passa_pelo_criterio(tmp_path: Path) -> None:
    """O pacote ``dnaty`` 2.2.1 roda aqui (CPU), com busca curtíssima para caber no teste.

    O veredito (aceito ou não) depende do que a busca encontrou; o que se confere é a
    etapa inteira: original intacto, otimizado separado, benchmark dos dois, licença
    BSL anotada e ``dnaty_version`` gravada.
    """
    dnaty = pytest.importorskip("dnaty")
    torch = pytest.importorskip("torch")
    x, y, w = _dados(n=256)
    torch.manual_seed(0)
    modelo = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.ReLU(),
                                 torch.nn.Linear(64, len(LABELS)))
    original = tmp_path / "original.onnx"
    with warnings.catch_warnings():  # exportador TorchScript, o mesmo que o dNATY usa
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(modelo, torch.zeros(1, D), str(original), dynamo=False,
                          input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "batch"}})
    otimizador = DnatyOtimizador(target_flops=0.5, n_generations=1, n_pop=2,
                                 finetune_epochs=1, seed=0)
    rel = executar(
        EntradaOtimizacao(original, tmp_path / "otimizado.onnx", torch_model=modelo,
                          x_treino=x, y_treino=y, input_shape=(D,)),
        otimizador, x, y, LABELS,
    )
    assert rel["dnaty_version"] == dnaty.__version__ == "2.2.1"
    assert "BSL 1.1" in rel["licenca"] and "comercial" in rel["licenca"]
    assert rel["original"]["benchmark"]["sha256"] != rel["otimizado"]["benchmark"]["sha256"]
    assert isinstance(rel["aceito"], bool) and "comparacao" in rel
