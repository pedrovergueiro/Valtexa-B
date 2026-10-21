"""Nível 2, 5.1: o tipo de cada atualização, a procedência da cabeça adaptada e a plasticidade
do dNATY (esta só onde houver torch e dnaty — no WSL)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from app.inference.atualizacao import TIPOS, classifica, procedencia, verifica_procedencia
from app.inference.continual import AdaptConfig
from app.inference.heads import DiagnosisHead, EmbeddingMLP
from tests.test_continual import CLASSES, TORRE, _acc, _memoria, mundo  # noqa: F401

SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64


def _cabeca(ocultos: int = 8, d: int = 6, encoder: str = TORRE,
            ativacao: str = "relu") -> DiagnosisHead:
    rng = np.random.default_rng(ocultos)
    mlp = EmbeddingMLP(np.zeros(d), np.ones(d), (rng.normal(size=(ocultos, d)),),
                       (np.zeros(ocultos),), (ativacao,), rng.normal(size=(3, ocultos)),
                       np.zeros(3))
    return DiagnosisHead(CLASSES, mlp, encoder, {"cnn": 0.0, "head": 1.0})


def _registro(**muda) -> dict:
    base = {"tipo": "cabeca", "modelo": "coffee-foliar", "encoder": TORRE, "cabeca": SHA_B,
            "cabeca_base": SHA_A, "rodada": "r1", "escopo": "T01",
            "quando": "2026-10-18T00:00:00Z", "aprovado_por": "Eng. Agr. Teste",
            "decisao": "APROVADA", "motivos": [], "laudo_sha256": SHA_C,
            "dados": {"campo_sha256": SHA_C}}
    return procedencia(**{**base, **muda})


def test_classifica_pela_forma() -> None:
    a = _cabeca()
    pesos = dataclasses.replace(a, mlp=dataclasses.replace(
        a.mlp, out_bias=a.mlp.out_bias + 0.1))
    assert classifica(a, pesos) == "cabeca"
    assert classifica(a, _cabeca(ocultos=9)) == "estrutura"
    assert classifica(a, _cabeca(ativacao="gelu")) == "estrutura"
    assert classifica(a, _cabeca(encoder="outra/torre")) == "embedding"
    assert {t for t, v in TIPOS.items() if v.continuo_pode} == {"cabeca", "prior"}
    assert "motor TensorRT" in TIPOS["embedding"].invalida


def test_procedencia_valida_e_o_que_a_derruba() -> None:
    r = _registro()
    assert verifica_procedencia(r, SHA_B, TORRE) == []
    assert any("outra cabeça" in p for p in verifica_procedencia(r, SHA_A))
    assert any("torre" in p for p in verifica_procedencia(r, SHA_B, "outra/torre"))
    for chave, valor, trecho in (("aprovado_por", "  ", "aprovador"),
                                 ("tipo", "embedding", "tipo"),
                                 ("formato", 2, "formato"),
                                 ("dados", {}, "dado de campo")):
        assert any(trecho in p for p in verifica_procedencia({**r, chave: valor}, SHA_B)), chave
    reprovada = {**r, "portao": {**r["portao"], "decisao": "REPROVADA"}}
    assert any("REPROVADA" in p for p in verifica_procedencia(reprovada, SHA_B))
    forcada = {**reprovada, "forcada": True, "motivo": "surto, agrônomo assumiu"}
    assert verifica_procedencia(forcada, SHA_B) == []
    assert verifica_procedencia("lixo", SHA_B) == ["procedência não é um objeto"]
    with pytest.raises(ValueError, match="aprovador"):
        _registro(aprovado_por="")


# ---------------------------------------------------------------------------------------------
# Plasticidade do dNATY (precisa de torch e dnaty)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def dnaty():
    pytest.importorskip("torch")
    return pytest.importorskip("dnaty.training.plasticity")


def test_ida_e_volta_pelo_dnaty_computa_o_mesmo(dnaty, mundo) -> None:  # noqa: F811
    import torch

    from app.inference.continual import standardize
    from training.motor_dnaty import para_dnaty

    base = mundo["base"]
    x = mundo["antigo_teste"][0]
    modelo = para_dnaty(base.mlp)
    with torch.no_grad():
        logits = modelo(torch.as_tensor(standardize(base.mlp, x), dtype=torch.float32)).numpy()
    assert np.allclose(logits, base.mlp.logits(x), atol=1e-4)
    volta = EmbeddingMLP.from_dnaty(modelo, base.mlp.mean, base.mlp.std)
    assert np.allclose(volta.logits(x), base.mlp.logits(x), atol=1e-4)


def test_renasce_so_os_dormentes_e_preserva_a_funcao(dnaty, mundo) -> None:  # noqa: F811
    from training.motor_dnaty import Plasticidade, renasce

    base = mundo["base"]
    x = mundo["antigo"][0]
    # Três neurônios mortos de propósito: entrada que nunca passa do zero da ReLU.
    pesos, vieses = base.mlp.weights[0].copy(), base.mlp.biases[0].copy()
    pesos[:3], vieses[:3] = 0.0, -5.0
    morta = dataclasses.replace(base, mlp=dataclasses.replace(
        base.mlp, weights=(pesos,), biases=(vieses,)))
    nova, rel = renasce(morta, x, Plasticidade(fracao=0.25, limiar_relativo=0.1))
    assert rel["renascidos"] == 3 and rel["por_camada"] == [3]
    assert not np.allclose(nova.mlp.weights[0][:3], 0.0)  # entrada nova
    assert np.all(nova.mlp.out_weight[:, :3] == 0.0)  # saída zerada: silenciosos
    assert np.array_equal(nova.mlp.weights[0][3:], morta.mlp.weights[0][3:])  # mesmos bits
    assert np.allclose(nova.mlp.logits(x), morta.mlp.logits(x), atol=1e-9)
    # Sem neurônio dormente de verdade, nada renasce (o limiar relativo segura).
    intacta, rel = renasce(base, x, Plasticidade(fracao=0.25, limiar_relativo=1e-6))
    assert rel["renascidos"] == 0 and intacta is base


def test_motor_dnaty_passa_pelo_mesmo_ciclo(dnaty, mundo) -> None:  # noqa: F811
    from training.motor_dnaty import MOTOR_DNATY, Plasticidade, adapta

    base = mundo["base"]
    x_nov, y_nov = mundo["novo"]
    memoria = _memoria(base, *mundo["antigo"])
    cfg = AdaptConfig(epocas=15, taxa=1e-2)
    res, rel = adapta(base, x_nov, y_nov, memoria, x_nov[::5], y_nov[::5], cfg,
                      mundo["antigo_val"], Plasticidade(fracao=0.1, limiar_relativo=None))
    assert rel["motor"] == MOTOR_DNATY and rel["renascidos"] > 0
    assert "rank_efetivo_final" in rel
    if res.epoch:
        assert _acc(res.head, *mundo["novo_teste"]) > _acc(base, *mundo["novo_teste"])
    else:  # sem ganho: a instalada volta, não a renascida
        assert res.head is base
