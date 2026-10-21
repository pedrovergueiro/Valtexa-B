"""Física da dispersão de esporos (``app/gemeo/dispersao.py``): a equação, a massa, o vento."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from app.gemeo import clima, grafo
from app.gemeo.dispersao import (
    Grade,
    NucleoComVento,
    ParametrosDispersao,
    bessel_k0,
    green,
    vento_no_dossel,
)
from app.gemeo.epidemia import nucleo_de_contagio, parametros_para
from app.gemeo.grafo import GrafoLavoura


def test_k0_contra_a_tabela() -> None:
    # Abramowitz & Stegun, tabela 9.8.
    x = np.array([0.1, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0])
    k0 = np.array([2.4270690247, 0.9244190712, 0.4210244382, 0.1138938727, 0.0623475532,
                   0.0036910983, 1.7780062e-5])
    assert np.allclose(bessel_k0(x), k0, rtol=2e-6)


@pytest.mark.parametrize("ux", [0.0, 0.6])
def test_massa_se_conserva(ux) -> None:
    """λ ∫ G dA = 1: o que sai da fonte deposita em algum lugar (com ou sem vento)."""
    h = 0.05
    oy, ox = np.mgrid[-40:40:h, -60:60:h]
    g = green(ox, oy, ux, 0.0, 0.8, 0.05, h / 2.0)
    assert 0.05 * g.sum() * h * h == pytest.approx(1.0, abs=0.02)


def test_satisfaz_a_equacao_fora_da_fonte() -> None:
    """u.∇G - D∇²G + λG = 0 longe da fonte (diferenças finitas centradas)."""
    h, ux, uy, d, lam = 0.02, 0.5, -0.2, 0.8, 0.05
    oy, ox = np.mgrid[0.5:3.0:h, 0.5:3.0:h]
    g = green(ox, oy, ux, uy, d, lam, h / 2.0)
    gx = (g[1:-1, 2:] - g[1:-1, :-2]) / (2 * h)
    gy = (g[2:, 1:-1] - g[:-2, 1:-1]) / (2 * h)
    lap = (g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4 * g[1:-1, 1:-1]) / h ** 2
    residuo = ux * gx + uy * gy - d * lap + lam * g[1:-1, 1:-1]
    assert np.abs(residuo).max() < 1e-3 * np.abs(lam * g[1:-1, 1:-1]).max() * 50


def test_vento_leva_a_nuvem() -> None:
    h = 0.1
    oy, ox = np.mgrid[-20:20:h, -30:30:h]
    calmo = green(ox, oy, 0.0, 0.0, 0.5, 0.05, h / 2.0)
    ventando = green(ox, oy, 0.8, 0.0, 0.5, 0.05, h / 2.0)
    assert (calmo * ox).sum() / calmo.sum() == pytest.approx(0.0, abs=0.01)
    assert (ventando * ox).sum() / ventando.sum() > 5.0  # a favor do vento


def test_fft_igual_a_soma_direta() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 30, 40).astype(float)
    y = rng.integers(0, 12, 40).astype(float) * 3.0
    pontos = set(zip(x, y, strict=True))
    x, y = np.array([p[0] for p in pontos]), np.array([p[1] for p in pontos])
    fonte = rng.random(len(x))
    grade = Grade(x, y, 1.0)
    c = grade.concentracao(fonte, 0.4, 0.1, 0.6, 0.05)
    dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
    g = green(dx, dy, 0.4, 0.1, 0.6, 0.05, 0.5)
    np.fill_diagonal(g, 0.0)
    assert np.allclose(c, g @ fonte, rtol=1e-6, atol=1e-12)


def _grafo(ruas: int = 12, plantas: int = 60) -> GrafoLavoura:
    lista = [{"plant_id": f"G{r + 1}-{i:03d}", "row_id": f"G{r + 1}", "x_along_m": i * 0.7,
              "observations": {"p0": {"time": "2026-01-01T10:00:00+00:00", "n_tiles": 1,
                                      "posterior": {"Rust": 0.05}}}}
             for r in range(ruas) for i in range(plantas)]
    return grafo.construir({"field_id": "t", "crop": "coffee", "diseases": ["Healthy", "Rust"],
                            "priors": {"Healthy": 0.95, "Rust": 0.05}, "plant_spacing_m": 0.7,
                            "row_spacing_m": 3.5, "passes": [{"pass_id": "p0"}],
                            "plants": lista})


def _dia(ms: float, de_graus: float) -> clima.DiaClimatico:
    u = -ms * math.sin(math.radians(de_graus))
    v = -ms * math.cos(math.radians(de_graus))
    return clima.DiaClimatico(date(2026, 1, 1), 22, 17, 27, 90, 5, 12, vento_ms=ms,
                              vento_u_ms=u, vento_v_ms=v)


def test_nucleo_com_vento_no_gemeo() -> None:
    g = _grafo()
    contato = nucleo_de_contagio(g, parametros_para("Rust"))
    nucleo = NucleoComVento(contato, g, ParametrosDispersao(fracao_aerea=1.0), 0.0)
    parado = clima.DiaClimatico(date(2026, 1, 1), 22, 17, 27, 90, 5, 12)
    uniforme = np.ones(g.n)
    assert nucleo.para(parado) is contato  # sem vento medido: só o contato
    # Mesma massa: no miolo do talhão, fonte uniforme dá a mesma pressão pelos dois caminhos.
    miolo = (g.rua > 3) & (g.rua < 8) & (g.x_m > 12) & (g.x_m < 30)
    ar = nucleo.para(_dia(0.0001, 270.0))(uniforme)
    assert ar[miolo].mean() == pytest.approx(contato(uniforme)[miolo].mean(), rel=0.2)
    # Uma fonte no meio, vento de oeste com as ruas para o norte: o ar vai para o leste, que
    # é atravessando as ruas, para o lado das ruas de número maior.
    fonte = np.zeros(g.n)
    centro = int(np.flatnonzero((g.rua == 6) & np.isclose(g.x_m, 21.0))[0])
    fonte[centro] = 1.0
    pressao = nucleo.para(_dia(4.0, 270.0))(fonte)
    leste = pressao[(g.rua == 7) & np.isclose(g.x_m, 21.0)][0]
    oeste = pressao[(g.rua == 5) & np.isclose(g.x_m, 21.0)][0]
    assert leste > 3 * oeste
    ao_longo, atravessando, _d = vento_no_dossel(_dia(4.0, 270.0), ParametrosDispersao(), 0.0)
    assert ao_longo == pytest.approx(0.0, abs=1e-9) and atravessando > 0


def test_vento_horario_vira_vetor_do_dia() -> None:
    inicio = datetime(2026, 1, 1)
    registros = [{"time": (inicio + timedelta(hours=h)).isoformat(), "temperatura_c": 22,
                  "ur": 80, "chuva_mm": 0, "vento_ms": 2.0, "vento_dir_graus": 270.0}
                 for h in range(24)]
    dia = clima.dias_de_registros_horarios(registros)[0]
    assert dia.tem_vento and dia.vento_ms == pytest.approx(2.0)
    assert dia.vento_u_ms == pytest.approx(2.0) and dia.vento_v_ms == pytest.approx(0.0, abs=1e-9)
    sem = clima.dias_de_registros_horarios([{**r, "vento_ms": None} for r in registros])[0]
    assert not sem.tem_vento


def _com_vento(serie: clima.SerieClimatica, ms: float = 3.0) -> clima.SerieClimatica:
    from dataclasses import replace

    dias = {d: replace(x, vento_ms=ms, vento_u_ms=ms, vento_v_ms=0.0)
            for d, x in serie.dias.items()}
    return clima.SerieClimatica(serie.lat, serie.lon, dias)


def test_escolha_de_omega_no_ajuste() -> None:
    """Talhão sintético com contágio só por contato: com vento na série, o ajuste não o adota
    (ou adota pouco); sem vento medido, nem tenta."""
    from app.gemeo import ajuste
    from app.gemeo.dispersao import ParametrosDispersao
    from training.mede_gemeo import talhao_sintetico

    talhao = talhao_sintetico(ruas=6, por_rua=30, passadas=4, focos=3, semente=3)
    g = grafo.construir(talhao.ate(4))
    modelo = ajuste.ajustar(g, "Rust", talhao.serie, max_iter=60)
    sem = ajuste.escolher_dispersao(g, "Rust", talhao.serie, modelo, ParametrosDispersao(), 0.0)
    assert sem["fracao_aerea"] == 0.0 and "vento medido" in sem["motivo"]
    com = ajuste.escolher_dispersao(g, "Rust", _com_vento(talhao.serie), modelo,
                                    ParametrosDispersao(), 0.0)
    assert set(com["por_fracao"]) == {"0.0", "0.3", "0.6", "0.9"}
    assert com["fracao_aerea"] <= 0.3 and g.dispersao is None  # o grafo volta como estava
    # Ligada no grafo, o núcleo do gêmeo passa a ter a parte do vento.
    from app.gemeo.epidemia import nucleo_de_contagio

    g.dispersao = (ParametrosDispersao(fracao_aerea=0.5), 0.0)
    assert hasattr(nucleo_de_contagio(g, modelo.parametros), "para")
