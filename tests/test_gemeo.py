"""O gêmeo digital (:mod:`app.gemeo`): tempo, grafo, física, filtro, ajuste, previsão, rede e
prescrição por risco — num talhão sintético pequeno, onde a verdade é conhecida."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from app.gemeo import ajuste, clima, epidemia, grafo, prescricao_risco, previsao, rede
from app.prescricao import Economia, PrescricaoImpossivelError, Produto
from training.mede_gemeo import talhao_sintetico

RUST = epidemia.PRESETS["Rust"]


@pytest.fixture(scope="module")
def talhao():
    return talhao_sintetico(ruas=6, por_rua=30, passadas=5, focos=3, semente=3)


@pytest.fixture(scope="module")
def ajustado(talhao):
    g = grafo.construir(talhao.ate(4))
    return g, ajuste.ajustar(g, "Rust", talhao.serie, max_iter=80)


# ---------------------------------------------------------------------------------------------
# Clima
# ---------------------------------------------------------------------------------------------


def _horas(dia: date, n: int, ur: float = 95.0, chuva: float = 0.0) -> list[dict]:
    base = datetime(dia.year, dia.month, dia.day, tzinfo=timezone.utc)
    return [{"time": (base + timedelta(hours=h)).isoformat(), "temperatura_c": 20.0 + h % 3,
             "ur": ur if h < 10 else 60.0, "chuva_mm": chuva} for h in range(n)]


def test_registros_horarios_viram_dia_e_dia_incompleto_sai() -> None:
    dias = clima.dias_de_registros_horarios(
        _horas(date(2026, 1, 1), 24) + _horas(date(2026, 1, 2), 10), "estacao")
    assert [d.data for d in dias] == [date(2026, 1, 1)]  # 10 horas: fora
    d = dias[0]
    assert d.molhamento_h == pytest.approx(10.0)  # UR >= 90 nas 10 primeiras horas
    assert not d.molhamento_medido and d.t_min <= d.t_media <= d.t_max


def test_estacao_vence_previsao_e_climatologia_preenche(tmp_path) -> None:
    serie = clima.SerieClimatica.vazia()
    dia = date(2026, 1, 1)
    previsto = clima.DiaClimatico(dia, 30, 25, 35, 50, 0, 0, fonte="open-meteo:previsao")
    medido = clima.DiaClimatico(dia, 20, 15, 25, 90, 5, 10, fonte="estacao")
    serie.junta([previsto]).junta([medido]).junta([previsto])
    assert serie.dias[dia].fonte == "estacao"
    periodo = serie.periodo(dia, dia + timedelta(days=2))
    assert [d.fonte for d in periodo] == ["estacao", "climatologia", "climatologia"]
    serie.save(tmp_path / "c.json")
    assert clima.SerieClimatica.load(tmp_path / "c.json").dias == serie.dias
    with pytest.raises(ValueError, match="CLIMA_OPEN_METEO"):
        clima.SerieClimatica.vazia().climatologia(dia)


def test_climatologia_no_29_de_fevereiro() -> None:
    serie = clima.SerieClimatica.vazia().junta(
        [clima.DiaClimatico(date(2027, 2, 27) + timedelta(days=k), 22, 17, 27, 80, 1, 6)
         for k in range(5)])
    assert serie.climatologia(date(2028, 2, 29)).fonte == "climatologia"


# ---------------------------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------------------------


def test_grafo_ordem_natural_e_razao(talhao) -> None:
    g = grafo.construir(talhao.ate(3))
    assert g.ruas[:3] == ["G1", "G2", "G3"] and g.n == 180 and g.t == 3
    razao = g.razao("Rust")
    nao_vistas = g.dia_obs < 0
    assert np.all(razao[nao_vistas] == 1.0)
    # A razão desfaz o prior da fusão: posterior igual ao prior não diz nada.
    assert epidemia.razao_de_verossimilhanca(np.array([0.1]), 0.1)[0] == pytest.approx(1.0)


def test_vizinhanca_exata_contra_a_soma_bruta() -> None:
    """O operador O(N) dá a mesma soma que a matriz cheia, com posições irregulares, ruas de
    tamanhos diferentes, pés na mesma posição e alcance pequeno (x/a grande)."""
    rng = np.random.default_rng(0)
    rua = np.concatenate([np.full(k, r) for r, k in enumerate((30, 1, 0, 25, 40))])
    x = np.concatenate([np.sort(rng.uniform(0, 300, k)) for k in (30, 1, 0, 25, 40)])
    x[5] = x[4]  # dois pés no mesmo ponto
    v = rng.random(len(rua)) * (rng.random(len(rua)) < 0.5)
    viz = grafo.Vizinhanca.construir(rua, x, 3.5)
    for a, b in ((1.5, 0.8), (0.2, 3.0), (3.0, 0.1)):
        dx = np.abs(x[:, None] - x[None, :])
        dr = np.abs(rua[:, None] - rua[None, :])
        w = np.exp(-dx / a) * np.exp(-dr * 3.5 / b) * (dr <= grafo.CORTE_RUAS)
        np.fill_diagonal(w, 0.0)
        assert np.allclose(viz.aplicar(v, a, b), w @ v, rtol=1e-9, atol=1e-12)


def test_arestas_dos_focos_simetricas() -> None:
    rua = np.array([0, 0, 0, 1, 1, 3])
    x = np.array([0.0, 0.7, 5.0, 0.3, 9.0, 0.0])
    o, d, dl, de = grafo.arestas(rua, x, 3.5, corte_rua_m=1.0, corte_ruas=1)
    pares = set(zip(o.tolist(), d.tolist(), strict=True))
    assert all((b, a) in pares for a, b in pares) and all(a != b for a, b in pares)
    assert (0, 1) in pares and (0, 3) in pares and (0, 2) not in pares
    assert dl.max() <= 1.0 and de.max() <= 3.5


def test_grafo_recusa_talhao_sem_passada() -> None:
    with pytest.raises(ValueError, match="passada fechada"):
        grafo.construir({"field_id": "x", "crop": "coffee", "diseases": ["Healthy", "Rust"],
                         "passes": [], "plants": [{"plant_id": "a", "row_id": "1",
                                                   "x_along_m": 0.0, "observations": {}}]})


def test_focos_juntam_vizinhas(talhao) -> None:
    g = grafo.construir(talhao.ate(1))
    marcadas = np.zeros(g.n, dtype=bool)
    marcadas[[0, 1, 2]] = True  # três pés seguidos na rua G1
    marcadas[g.n - 1] = True  # um isolado na última rua
    grupos = grafo.componentes(g, marcadas, 1.1 * g.espacamento_entre_ruas_m)
    assert sorted(len(x) for x in grupos) == [1, 3]


# ---------------------------------------------------------------------------------------------
# Física e observação
# ---------------------------------------------------------------------------------------------


def test_resposta_ao_tempo() -> None:
    assert epidemia.resposta_temperatura(RUST.t_opt, RUST) == pytest.approx(1.0)
    assert epidemia.resposta_temperatura(RUST.t_min, RUST) == 0.0
    assert epidemia.resposta_temperatura(RUST.t_max + 1, RUST) == 0.0
    seco = clima.DiaClimatico(date(2026, 1, 1), 22, 17, 27, 50, 0, 0)
    molhado = replace(seco, molhamento_h=14, chuva_mm=20)
    assert epidemia.favorabilidade(molhado, RUST) > 10 * epidemia.favorabilidade(seco, RUST)
    # Bicho-mineiro é praga de tempo seco: o molhamento desfavorece.
    mineiro = epidemia.PRESETS["Leaf_Miner_Damage"]
    assert epidemia.favorabilidade(seco, mineiro) > epidemia.favorabilidade(molhado, mineiro)


def test_passo_conserva_e_sem_inoculo_nao_adoece(talhao) -> None:
    g = grafo.construir(talhao.ate(1))
    dia = clima.DiaClimatico(date(2026, 1, 1), 22, 17, 27, 90, 10, 12)
    sem_fundo = replace(RUST, fundo=0.0)
    limpo = epidemia.passo(epidemia.Estado.sadio(g.n), dia,
                           epidemia.nucleo_de_contagio(g, sem_fundo), sem_fundo)
    assert np.all(limpo.S == 1.0)
    e = epidemia.Estado.sadio(g.n)
    e.I[0], e.S[0] = 0.8, 0.2
    traj = epidemia.simular(e, [dia] * 30, g, RUST)
    for s in traj.estados:
        assert np.allclose(s.S + s.E + s.I + s.R, 1.0)
        assert np.all(s.S >= -1e-12) and np.all(s.I >= 0)
    vizinha, longe = 1, g.n - 1  # o pé seguinte na mesma rua; o último da última rua
    assert traj.estados[-1].E[vizinha] + traj.estados[-1].I[vizinha] > (
        traj.estados[-1].E[longe] + traj.estados[-1].I[longe])


def test_tratamento_protege() -> None:
    t = talhao_sintetico(ruas=3, por_rua=10, passadas=2, semente=1)
    g = grafo.construir(t.ate(1))
    e = epidemia.Estado.sadio(g.n)
    e.I[5], e.S[5] = 0.9, 0.1
    dias = [clima.DiaClimatico(date(2026, 1, 1), 22, 17, 27, 90, 10, 12)] * 20
    nada = epidemia.simular(e, dias, g, RUST).estados[-1]
    todas = np.ones(g.n)
    com = epidemia.simular(e, dias, g, RUST, [epidemia.Tratamento(0, todas, 0.9, 20)]).estados[-1]
    assert com.E.sum() < 0.2 * nada.E.sum()


def test_assimilacao_bayesiana_por_planta() -> None:
    base = epidemia.Estado(np.full(3, 0.7), np.full(3, 0.1), np.full(3, 0.2), np.zeros(3))
    razao = np.array([50.0, 1.0, 0.02])  # sinal forte, nada, ausência forte
    pos = epidemia.estado_da_observacao(razao, np.ones(3, dtype=bool), RUST, base)
    assert pos.I[0] > 0.2 and pos.I[1] == pytest.approx(0.2) and pos.I[2] < 0.2
    assert np.allclose(pos.S + pos.E + pos.I + pos.R, 1.0)
    # S, E e R não se distinguem pela câmera: a proporção entre eles se mantém.
    assert pos.E[0] / pos.S[0] == pytest.approx(0.1 / 0.7)
    nao_vista = epidemia.estado_da_observacao(razao, np.zeros(3, dtype=bool), RUST, base)
    assert np.allclose(nao_vista.I, base.I)


def test_prevalencia_empirica_recupera_a_fracao() -> None:
    rng = np.random.default_rng(0)
    p = replace(RUST, alarme_falso=0.03, sensibilidade=0.9)
    doente = rng.random(5000) < 0.08
    sinal = rng.random(5000) < np.where(doente, p.sensibilidade, p.alarme_falso)
    x = rng.normal(np.where(sinal, 2.0, -2.0), 1.5)
    razao = np.exp(2 * 2.0 * x / 1.5**2)
    assert epidemia.prevalencia_empirica(razao, p) == pytest.approx(0.08, abs=0.02)


def test_espaco_livre_e_serializacao() -> None:
    for nome, v in (("beta", 0.08), ("alarme_falso", 0.03), ("sensibilidade", 0.9)):
        assert epidemia.de_livre(nome, epidemia.para_livre(nome, v)) == pytest.approx(v)
    m = epidemia.ModeloAjustado(RUST, ajuste.LIVRES,
                                np.array([epidemia.para_livre(k, getattr(RUST, k))
                                          for k in ajuste.LIVRES]),
                                np.eye(len(ajuste.LIVRES)) * 4.0, {"identificado": True})
    volta = epidemia.ModeloAjustado.from_dict(m.to_dict())
    assert volta.parametros == RUST and np.allclose(volta.covariancia_livre, m.covariancia_livre)
    # Covariância larga: o sorteio que violaria a < s é descartado, nunca devolvido.
    for amostra in m.amostras(50, np.random.default_rng(0)):
        assert amostra.alarme_falso < amostra.sensibilidade


# ---------------------------------------------------------------------------------------------
# Filtro, ajuste
# ---------------------------------------------------------------------------------------------


def test_filtro_relogio_e_dia_base(talhao) -> None:
    g = grafo.construir(talhao.ate(4))
    cl = talhao.serie.periodo(g.inicio, g.inicio + timedelta(days=int(g.dia_obs.max()) + 1))
    passos = list(epidemia.filtrar(g, g.razao("Rust"), cl, RUST))
    assert [s.k for s in passos] == [0, 1, 2, 3]
    assert passos[-1].previsto_esporulando is None
    assert passos[-1].dia == epidemia.dia_da_ultima_assimilacao(g)
    assert all(a.dia < b.dia for a, b in itertools.pairwise(passos))


def test_ajuste_identifica_e_da_incerteza(ajustado) -> None:
    g, m = ajustado
    d = m.diagnostico
    assert d["perda_por_planta_ajuste"] < d["perda_por_planta_prior"]
    # "Identificado" é o ganho total (razão de verossimilhança), não por planta.
    assert d["identificado"] == (d["ganho_total_nats"] >= ajuste.GANHO_IDENTIFICADO)
    assert d["laplace"] == "Hessiana no ótimo"
    for k in ajuste.LIVRES:
        lo, hi = d["intervalo_95"][k]
        assert lo <= getattr(m.parametros, k) <= hi
    assert m.parametros.alarme_falso < m.parametros.sensibilidade


def test_uma_passada_fica_no_prior(talhao) -> None:
    m = ajuste.ajustar(grafo.construir(talhao.ate(1)), "Rust", talhao.serie)
    assert not m.diagnostico["identificado"] and m.parametros == RUST


# ---------------------------------------------------------------------------------------------
# Previsão
# ---------------------------------------------------------------------------------------------


def test_previsao_risco_faixa_e_focos(talhao, ajustado) -> None:
    g, m = ajustado
    base = g.inicio + timedelta(days=epidemia.dia_da_ultima_assimilacao(g))
    prev = previsao.prever(g, "Rust", m, talhao.serie, hoje=base, horizonte_dias=14, membros=6)
    assert prev.data_base == base and prev.horizonte == base + timedelta(days=14)
    assert np.all((prev.risco >= 0) & (prev.risco <= 1))
    assert np.all(prev.risco_p10 <= prev.risco_p90 + 1e-12)
    assert len(prev.diario) == 15
    # O risco no horizonte inclui o latente: nunca abaixo de P(esporulando) agora menos o
    # que sai para R em 14 dias.
    saida = 1 - math.exp(-14 / m.parametros.infecciosidade_dias)
    assert np.all(prev.risco >= prev.esporulando_agora * (1 - saida) - 1e-9)
    corpo = prev.to_dict(g)
    assert corpo["plantas"] == g.n and "plantas_detalhe" in corpo
    assert all(0 <= p["risco"] <= 1 for p in corpo["plantas_detalhe"])
    with pytest.raises(ValueError, match="antes da última passada"):
        previsao.prever(g, "Rust", m, talhao.serie, hoje=base - timedelta(days=1))


def test_janelas_de_infeccao() -> None:
    datas = [date(2026, 1, 1) + timedelta(days=k) for k in range(6)]
    janelas = previsao.janelas_de_infeccao(datas, np.array([0.1, 0.6, 0.7, 0.2, 0.9, 0.1]))
    assert [(j["inicio"], j["dias"]) for j in janelas] == [("2026-01-02", 2), ("2026-01-05", 1)]


def test_perturbacao_do_clima_so_no_futuro() -> None:
    hoje = date(2026, 1, 10)
    medido = clima.DiaClimatico(hoje - timedelta(days=1), 22, 17, 27, 80, 5, 8, fonte="estacao")
    previsto = replace(medido, data=hoje + timedelta(days=5), fonte="open-meteo:previsao")
    a, b = previsao.perturbar_clima([medido, previsto], hoje, np.random.default_rng(0))
    assert a == medido and b != previsto and 0 <= b.molhamento_h <= 24


# ---------------------------------------------------------------------------------------------
# Rede residual
# ---------------------------------------------------------------------------------------------


def test_rede_portao_e_serializacao(talhao, ajustado) -> None:
    g, m = ajustado
    cl = talhao.serie.periodo(g.inicio, g.inicio + timedelta(days=int(g.dia_obs.max()) + 1))
    hist = rede.historico(g, "Rust", cl, m.parametros)
    assert len(hist.pares) == g.t - 1 and hist.final.shape == (g.n, len(rede.ENTRADAS))
    residual, diag = rede.ajustar_rede(g, hist, m.parametros)
    assert "ganho_por_planta" in diag
    assert (residual is not None) == diag["adotada"]
    # Uma rede com saída zero é a física pura, e volta igual da serialização.
    r = rede._treinar(g, hist.pares, (1.5, 0.8))
    r.v[:] = 0.0
    r.c = 0.0
    fisica = hist.pares[0].fisica
    assert np.allclose(r.corrigir(g, hist.pares[0].entradas, fisica), np.clip(fisica, 1e-4,
                                                                               1 - 1e-4))
    volta = rede.RedeResidual.from_dict(r.to_dict())
    assert np.allclose(volta.w, r.w)
    r.diagnostico = {"intervalo_mediano_dias": 14.0}
    assert rede.aplicavel(r, 14) and not rede.aplicavel(r, 40)


def test_rede_sem_pares_suficientes(talhao) -> None:
    g = grafo.construir(talhao.ate(2))
    cl = talhao.serie.periodo(g.inicio, g.inicio + timedelta(days=int(g.dia_obs.max()) + 1))
    hist = rede.historico(g, "Rust", cl, RUST)
    residual, diag = rede.ajustar_rede(g, hist, RUST)
    assert residual is None and not diag["adotada"]


# ---------------------------------------------------------------------------------------------
# Prescrição por risco
# ---------------------------------------------------------------------------------------------

PRODUTO = Produto(nome="Fungicida X", registro_mapa="12345", culturas=("coffee",),
                  doencas=("Rust",), dose_ml_por_planta=50, dose_min_ml_por_planta=40,
                  dose_max_ml_por_planta=60, preco_por_litro=80.0, eficacia=0.85,
                  carencia_dias=30)


def test_prescricao_por_risco_e_cenarios(talhao, ajustado) -> None:
    g, m = ajustado
    base = g.inicio + timedelta(days=epidemia.dia_da_ultima_assimilacao(g))
    prev = previsao.prever(g, "Rust", m, talhao.serie, hoje=base, horizonte_dias=21, membros=4)
    r = prescricao_risco.prescrever_por_risco(
        g, prev, m, talhao.serie, produto=PRODUTO, economia=Economia(perda_por_planta=40.0),
        residual_dias=21, membros=4)
    c = r.cenarios
    tratadas = c["prescricao_na_data"]["plantas_tratadas"]
    assert tratadas == r.prescricao.plantas_tratadas
    if tratadas:
        assert c["prescricao_na_data"]["infectadas_no_horizonte"] < (
            c["sem_aplicacao"]["infectadas_no_horizonte"])
    assert c["talhao_inteiro_na_data"]["infectadas_no_horizonte"] <= (
        c["prescricao_na_data"]["infectadas_no_horizonte"] + 1e-6)
    assert r.prescricao.contagio_ajustado == m.diagnostico["identificado"]
    assert r.to_dict()["cenarios"] == c
    with pytest.raises(PrescricaoImpossivelError, match="residual"):
        prescricao_risco.prescrever_por_risco(g, prev, m, talhao.serie, produto=PRODUTO,
                                              economia=Economia(perda_por_planta=40.0),
                                              residual_dias=0)
    sem_bula = replace(PRODUTO, culturas=("banana",))
    with pytest.raises(PrescricaoImpossivelError):
        prescricao_risco.prescrever_por_risco(g, prev, m, talhao.serie, produto=sem_bula,
                                              economia=Economia(perda_por_planta=40.0),
                                              residual_dias=21)


def test_data_de_aplicacao_e_vespera_da_janela(talhao, ajustado) -> None:
    g, m = ajustado
    base = g.inicio + timedelta(days=epidemia.dia_da_ultima_assimilacao(g))
    prev = previsao.prever(g, "Rust", m, talhao.serie, hoje=base, horizonte_dias=10, membros=1)
    prev.janelas = [{"inicio": (base + timedelta(days=4)).isoformat(),
                     "fim": (base + timedelta(days=5)).isoformat(), "dias": 2,
                     "favorabilidade_max": 0.8}]
    data, motivo = prescricao_risco.data_de_aplicacao(prev)
    assert data == base + timedelta(days=3) and "janela" in motivo
    prev.janelas = []
    assert prescricao_risco.data_de_aplicacao(prev)[0] is None
