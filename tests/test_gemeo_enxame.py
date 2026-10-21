"""O gêmeo no enxame: canal por fonte, evidência de outros dispositivos e prioridade de
inspeção pelo valor da informação."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from app.gemeo import ajuste, epidemia, grafo, previsao, prioridade
from training.mede_gemeo import talhao_sintetico

RUST = epidemia.PRESETS["Rust"]
ROVER = epidemia.canal_de(RUST)
ECO = prioridade.Economia(custo_tratar=2.0, perda_planta=40.0, eficacia=0.85)


# ---------------------------------------------------------------------------------------------
# Canal por fonte
# ---------------------------------------------------------------------------------------------


def test_canal_de_cada_fonte() -> None:
    assert epidemia.Canal(RUST.sensibilidade, RUST.alarme_falso) == ROVER
    assert epidemia.canal_de(RUST, "drone") is epidemia.CANAIS["drone"]
    with pytest.raises(KeyError, match="sem canal"):
        epidemia.canal_de(RUST, "satelite")
    with pytest.raises(ValueError):
        epidemia.Canal(0.1, 0.2)


def test_fonte_pior_pesa_menos_com_a_mesma_evidencia() -> None:
    base = epidemia.Estado(np.full(1, 0.8), np.full(1, 0.1), np.full(1, 0.1), np.zeros(1))
    razao = np.array([30.0])
    vista = np.ones(1, dtype=bool)
    pelo_rover = epidemia.estado_da_observacao(razao, vista, RUST, base, ROVER)
    pelo_drone = epidemia.estado_da_observacao(razao, vista, RUST, base,
                                               epidemia.CANAIS["drone"])
    assert pelo_rover.I[0] > pelo_drone.I[0] > base.I[0]


def test_filtro_usa_o_canal_da_fonte_de_cada_passada() -> None:
    t = talhao_sintetico(ruas=4, por_rua=20, passadas=3, drone_a_cada=7, semente=2)
    obs = t.ate(3)
    g = grafo.construir(obs)
    assert "drone" in g.fontes and g.fontes[0] == "rover"
    cl = t.serie.periodo(g.inicio, g.inicio + timedelta(days=int(g.dia_obs.max()) + 1))
    certo = list(epidemia.filtrar(g, g.razao("Rust"), cl, RUST))[-1].assimilado
    tudo_rover = grafo.construir({**obs, "passes": [{**p, "fonte": "rover"}
                                                    for p in obs["passes"]]})
    errado = list(epidemia.filtrar(tudo_rover, tudo_rover.razao("Rust"), cl, RUST))[-1]
    assert not np.allclose(certo.I, errado.assimilado.I)


def test_juntar_externas_preserva_a_evidencia_e_ordena_pelo_instante() -> None:
    t = talhao_sintetico(ruas=2, por_rua=5, passadas=2, semente=0)
    obs = t.ate(2)
    alvo, outra = obs["plants"][0]["plant_id"], obs["plants"][1]["plant_id"]
    externa = {"fonte": "drone", "pass_id": "voo-1",
               # 11h de Brasília = 14h UTC: depois da primeira passada (10h UTC) do mesmo dia.
               "time": "2025-11-01T11:00:00-03:00",
               "evidencia": {"Rust": {alvo: 2.0, outra: -1.0, "fantasma": 3.0}}}
    junto, info = grafo.juntar_externas(obs, [externa])
    assert info == {"passadas_externas": 1, "plantas_desconhecidas": 1}
    assert [p["pass_id"] for p in junto["passes"]] == ["p0", "drone:voo-1", "p1"]
    g = grafo.construir(junto)
    k = g.passadas.index("drone:voo-1")
    assert g.fonte(k) == "drone"
    i = g.plant_ids.index(alvo)
    assert np.log(g.razao("Rust")[k, i]) == pytest.approx(2.0)
    assert g.dia_obs[k, g.plant_ids.index(obs["plants"][2]["plant_id"])] == -1


# ---------------------------------------------------------------------------------------------
# Valor da informação
# ---------------------------------------------------------------------------------------------


def test_voi_zero_quando_a_decisao_ja_esta_tomada_e_maximo_perto_do_limiar() -> None:
    p = np.array([1e-6, 0.059, 0.3, 0.6, 0.999999])
    voi = prioridade.valor_da_informacao(p, p, ROVER, ECO)
    assert voi[0] == pytest.approx(0, abs=1e-6) and voi[4] == pytest.approx(0, abs=1e-4)
    assert voi[1] > voi[2] > 0  # o limiar é c/(L e) = 0,059
    # Com 60%, nem o "não vi nada" do Rover (s = 0,9) desce abaixo do limiar: trata de
    # qualquer jeito, olhar não muda a decisão.
    assert voi[3] == pytest.approx(0, abs=1e-12)
    assert np.all(prioridade.valor_da_informacao(np.linspace(0, 1, 50), np.linspace(0, 1, 50),
                                                 ROVER, ECO) >= 0)


def test_sensor_inutil_e_latente_invisivel_nao_valem_nada() -> None:
    inutil = epidemia.Canal(0.3, 0.3 - 1e-12)
    p = np.array([0.06, 0.3])
    assert np.allclose(prioridade.valor_da_informacao(p, p, inutil, ECO), 0, atol=1e-9)
    assert np.allclose(prioridade.informacao_mutua(p, p, inutil), 0, atol=1e-9)
    # Só latente (nada esporula): a câmera não vê, o sinal é o do alarme falso.
    assert np.allclose(prioridade.valor_da_informacao(p, np.zeros(2), ROVER, ECO), 0,
                       atol=1e-9)


def test_sensor_perfeito_vale_o_arrependimento_esperado() -> None:
    perfeito = epidemia.Canal(1.0, 0.0)
    p = np.array([0.2])
    escala = ECO.perda_planta * ECO.eficacia
    # Sem olhar: trata (0,2 > 0,059) e paga c para nada em 80% das vezes; olhando, só os 20%.
    esperado = escala * (p * (1 - 0.0588) - (p - 2.0 / escala))
    assert prioridade.valor_da_informacao(p, p, perfeito, ECO)[0] == pytest.approx(
        float(esperado[0]), rel=1e-3)


def test_consequencia_pelo_contagio() -> None:
    t = talhao_sintetico(ruas=3, por_rua=10, passadas=1, semente=0)
    g = grafo.construir(t.ate(1))
    nucleo = epidemia.nucleo_de_contagio(g, RUST)
    c = prioridade.consequencia_por_contagio(nucleo, np.ones(g.n), RUST.beta, 0.6, 21)
    assert np.all(c >= 1) and np.all(c <= 1 + prioridade.R_MAXIMO)
    meio, canto = g.plant_ids.index("G2-005"), g.plant_ids.index("G1-000")
    assert c[meio] > c[canto]  # mais vizinhas suscetíveis, mais a perder


# ---------------------------------------------------------------------------------------------
# Tarefas
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cenario():
    t = talhao_sintetico(ruas=6, por_rua=40, passadas=4, focos=2, semente=5)
    g = grafo.construir(t.ate(3))
    m = ajuste.ajustar(g, "Rust", t.serie, max_iter=60)
    base = g.inicio + timedelta(days=epidemia.dia_da_ultima_assimilacao(g))
    prev = previsao.prever(g, "Rust", m, t.serie, hoje=base, horizonte_dias=3, membros=2)
    return g, m, prev


def test_cada_planta_numa_tarefa_so(cenario) -> None:
    g, _m, _prev = cenario
    for tipo in ("inspecao", "sobrevoo"):
        grupos = prioridade.agrupar(g, tipo, segmento_m=10, bloco_ruas=3, bloco_m=20)
        todas = np.concatenate(grupos)
        assert len(todas) == g.n and len(np.unique(todas)) == g.n
    assert all(len({int(g.rua[i]) for i in grupo}) == 1
               for grupo in prioridade.agrupar(g, "inspecao", segmento_m=10))
    assert max(len({int(g.rua[i]) for i in grupo})
               for grupo in prioridade.agrupar(g, "sobrevoo", bloco_ruas=3)) == 3


def test_priorizar_ordena_e_avisa(cenario) -> None:
    g, m, prev = cenario
    pr = prioridade.priorizar(g, prev, m, epidemia.canal_de(m.parametros), nome_canal="rover",
                              economia=ECO, segmento_m=10)
    valores = [t.valor for t in pr.tarefas]
    assert valores == sorted(valores, reverse=True) and pr.criterio == "valor_da_informacao"
    assert pr.tarefas[0].dias_sem_ver is not None and pr.tarefas[0].motivo
    assert np.isclose(sum(valores), pr.valor_por_planta.sum())
    drone = prioridade.priorizar(g, prev, m, epidemia.CANAIS["drone"], nome_canal="drone")
    assert drone.criterio == "informacao" and drone.tarefas[0].tipo == "sobrevoo"
    assert any("suposição de projeto" in a for a in drone.avisos)
    assert any("Sem os números do produtor" in a for a in drone.avisos)
    corpo = pr.to_dict(g, max_tarefas=3, com_plantas=True)
    assert len(corpo["tarefas"]) == 3 and corpo["tarefas_total"] == len(pr.tarefas)
    assert corpo["tarefas"][0]["plant_ids"]


def test_poligono_quando_ha_gps(cenario) -> None:
    g, m, prev = cenario
    com_gps = replace(g, lat=np.full(g.n, -21.2) + g.x_m * 1e-6,
                      lon=np.full(g.n, -45.0) + g.rua * 3e-5)
    pr = prioridade.priorizar(com_gps, prev, m, epidemia.canal_de(m.parametros),
                              nome_canal="rover", segmento_m=10)
    poli = pr.tarefas[0].poligono
    assert poli is not None and poli[0] == poli[-1] and len(poli) == 5
    sem_gps = prioridade.priorizar(g, prev, m, epidemia.canal_de(m.parametros),
                                   nome_canal="rover", segmento_m=10)
    assert sem_gps.tarefas[0].poligono is None
