"""Prescrição variável preditiva (``app/aplicacao_localizada.py``): trecho, taxa, PWM, checagem."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from app import aplicacao_localizada as al
from app.gemeo import clima, grafo
from tests.test_gemeo_api import field_dir  # noqa: F401

PRODUTO = al.ProdutoDaBula("Fungicida X", "12345", ("coffee",), ("Rust",), 200.0, 400.0, 500.0,
                           30, residual_dias=14, vento_max_ms=2.5)
MAQUINA = al.Equipamento("pulverizador de seções", vazao_nominal_l_min=1.2,
                         faixa_por_bico_m=0.5, bicos_por_lado=4, velocidade_m_s=1.0,
                         velocidade_min_m_s=0.5, velocidade_max_m_s=2.0)


def _grafo():
    plantas = [{"plant_id": f"G{r + 1}-{i:03d}", "row_id": f"G{r + 1}", "x_along_m": i * 0.7,
                "observations": {"p0": {"time": "2026-01-01T10:00:00+00:00", "n_tiles": 1,
                                        "posterior": {"Rust": 0.05}}}}
               for r in range(2) for i in range(30)]
    return grafo.construir({"field_id": "t", "crop": "coffee", "diseases": ["Healthy", "Rust"],
                            "priors": {"Healthy": 0.95, "Rust": 0.05}, "plant_spacing_m": 0.7,
                            "row_spacing_m": 3.5, "passes": [{"pass_id": "p0"}],
                            "plants": plantas})


def _previsao(g, risco: np.ndarray, faixa: float = 0.1):
    return SimpleNamespace(risco=risco, risco_p10=np.clip(risco - faixa, 0, 1),
                           risco_p90=np.clip(risco + faixa, 0, 1),
                           esporulando_agora=risco / 2, hoje=date(2026, 2, 1),
                           horizonte=date(2026, 2, 15), doenca="Rust", avisos=[])


def test_taxa_por_trecho_presa_a_bula() -> None:
    g = _grafo()
    # G1: risco 0,05 no começo, 0,30 no meio, 0,90 no fim; G2 inteira em 0,9 mas excluída.
    risco = np.where(g.x_m < 7, 0.05, np.where(g.x_m < 14, 0.30, 0.90))
    risco = np.where(g.rua == 1, 0.9, risco)
    zonas = al.taxas(al.mapa_de_risco(g, _previsao(g, risco), 7.0,
                                      exclusoes=[al.Exclusao("G2", 0, 100, "APP")]),
                     PRODUTO, al.Regras())
    g1 = [z for z in zonas if z.row_id == "G1"]
    assert g1[0].taxa_l_ha == 0.0 and "abaixo do limiar" in g1[0].motivo
    assert PRODUTO.dose_min_l_ha < g1[1].taxa_l_ha < PRODUTO.dose_l_ha
    assert g1[2].taxa_l_ha == PRODUTO.dose_l_ha  # nunca acima da dose padrão
    assert all(z.taxa_l_ha == 0.0 and "APP" in z.motivo for z in zonas if z.row_id == "G2")


def test_residual_de_aplicacao_anterior_protege() -> None:
    g = _grafo()
    zonas = al.taxas(al.mapa_de_risco(
        g, _previsao(g, np.full(g.n, 0.9)), 7.0,
        anteriores=[al.AplicacaoAnterior(date(2026, 1, 25), ("G1",), 14)]), PRODUTO, al.Regras())
    assert all(z.taxa_l_ha == 0 and "residual" in z.motivo for z in zonas if z.row_id == "G1")
    assert all(z.taxa_l_ha > 0 for z in zonas if z.row_id == "G2")


def test_vazao_e_pwm() -> None:
    assert al.vazao_por_bico_l_min(300, 1.0, 0.5) == pytest.approx(0.9)
    z = [al.Zona("G1", 0, 5, 7, .5, .5, .4, .6, .2, None, None, taxa_l_ha=300.0),
         al.Zona("G1", 5, 10, 7, .5, .5, .4, .6, .2, None, None, taxa_l_ha=50.0),
         al.Zona("G1", 10, 15, 7, .5, .5, .4, .6, .2, None, None, taxa_l_ha=0.0)]
    al.plano_de_bicos(z, MAQUINA)
    assert z[0].bico["ciclo"] == pytest.approx(0.75) and z[0].bico["estado"] == "ok"
    # 50 L/ha a 1 m/s dá 12,5% de ciclo: abaixo do mínimo; a 1,6 m/s chega a 20%.
    assert z[1].bico["estado"] == "acelerar" and z[1].bico["ciclo"] == pytest.approx(0.2)
    assert z[1].bico["velocidade_m_s"] == pytest.approx(1.6)
    assert z[2].bico["estado"] == "fechado"
    lenta = al.Equipamento(**{**MAQUINA.__dict__, "velocidade_max_m_s": 1.2})
    z2 = [al.Zona("G1", 0, 5, 7, .5, .5, .4, .6, .2, None, None, taxa_l_ha=50.0)]
    al.plano_de_bicos(z2, lenta)
    assert z2[0].bico["estado"] == "inviavel" and not z2[0].bico["viavel"]


def test_verificacoes_bloqueiam() -> None:
    g = _grafo()
    prev = _previsao(g, np.full(g.n, 0.6))
    ventando = clima.DiaClimatico(date(2026, 2, 1), 24, 18, 29, 70, 0, 2, vento_ms=4.0,
                                  vento_u_ms=4.0, vento_v_ms=0.0)
    corpo = al.mapa_de_aplicacao(g, prev, PRODUTO, MAQUINA, dia=ventando, dias_ate_colheita=10)
    falhas = {c["codigo"] for c in corpo["verificacoes"] if not c["ok"]}
    assert falhas == {"vento", "carencia"} and corpo["estado"] == "proposta"
    calmo = clima.DiaClimatico(date(2026, 2, 1), 24, 18, 29, 70, 0, 2, vento_ms=1.0,
                               vento_u_ms=1.0, vento_v_ms=0.0)
    ok = al.mapa_de_aplicacao(g, prev, PRODUTO, MAQUINA, dia=calmo, dias_ate_colheita=60)
    assert ok["bloqueios"] == [] and ok["totais"]["litros"] > 0
    outro_alvo = al.ProdutoDaBula(**{**PRODUTO.__dict__, "alvos": ("Phoma_Leaf_Spot",)})
    assert "alvo" in {c["codigo"] for c in al.mapa_de_aplicacao(
        g, prev, outro_alvo, MAQUINA, dia=calmo, dias_ate_colheita=60)["verificacoes"]
        if not c["ok"]}
    # O hash muda quando muda o que se executa.
    mais = al.mapa_de_aplicacao(g, _previsao(g, np.full(g.n, 0.9)), PRODUTO, MAQUINA, dia=calmo,
                                dias_ate_colheita=60)
    assert ok["sha256"] != mais["sha256"]
    assert ok["sha256"] == al.mapa_de_aplicacao(g, prev, PRODUTO, MAQUINA, dia=calmo,
                                                dias_ate_colheita=60)["sha256"]


def test_bula_incoerente_recusa() -> None:
    with pytest.raises(al.AplicacaoImpossivelError, match="dose"):
        al.ProdutoDaBula("X", "1", ("coffee",), ("Rust",), 500, 400, 600, 30)
    with pytest.raises(al.AplicacaoImpossivelError, match="registro"):
        al.ProdutoDaBula("X", " ", ("coffee",), ("Rust",), 100, 200, 300, 30)


def test_rota_no_gemeo(client, field_dir) -> None:  # noqa: F811
    from tests.test_gemeo_api import INICIO, _dias, _passada

    for k in range(3):
        _passada(client, k)
    client.post("/fields/C1/clima", json={"dias": _dias(INICIO.date(), 40)})
    client.post("/fields/C1/gemeo/ajuste", params={"doenca": "Rust", "max_iter": 30})
    hoje = (INICIO + timedelta(days=28)).date().isoformat()
    corpo = {"doenca": "Rust", "hoje": hoje, "horizonte_dias": 7, "membros": 2,
             "dias_ate_colheita": 90, "trecho_m": 3.0,
             "produto": {"nome": "Fungicida X", "registro_mapa": "12345",
                         "culturas": ["coffee"], "alvos": ["Rust"], "dose_min_l_ha": 200,
                         "dose_l_ha": 400, "dose_max_l_ha": 500, "carencia_dias": 30},
             "equipamento": {"nome": "seções", "vazao_nominal_l_min": 1.2,
                             "faixa_por_bico_m": 0.5, "bicos_por_lado": 4,
                             "velocidade_m_s": 1.0, "velocidade_min_m_s": 0.5,
                             "velocidade_max_m_s": 2.0},
             "regras": {"limiar_risco": 0.02, "limiar_dose_cheia": 0.3}}
    r = client.post("/fields/C1/gemeo/aplicacao-localizada", json=corpo)
    assert r.status_code == 200, r.text
    mapa = r.json()
    assert mapa["estado"] == "proposta" and len(mapa["sha256"]) == 64 and mapa["zonas"]
    assert all(0 <= z["taxa_l_ha"] <= 400 for z in mapa["zonas"])
    ruim = {**corpo, "produto": {**corpo["produto"], "dose_min_l_ha": 600}}
    assert client.post("/fields/C1/gemeo/aplicacao-localizada", json=ruim).status_code == 422
