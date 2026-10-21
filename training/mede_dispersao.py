"""Mede se a dispersão pelo vento prevê melhor onde a doença vai aparecer (talhão sintético).

    python -m training.mede_dispersao --sementes 0 1 2 3 4 --saida mede_dispersao.json

A verdade é um talhão de café (30 ruas de 100 plantas, 0,7 x 3,5 m) com SEIR estocástico por
planta, em que a pressão de infecção é a mistura de contato e ar (``ω`` verdadeiro) com vento
predominante de oeste, as ruas para o norte (o ar atravessa as ruas). Do dia 0 ao 49, a partir
de 5 focos. Com o estado verdadeiro de quem esporula (o melhor caso — o gêmeo real estima isso
com a fusão), cada modelo:

* ajusta o seu ``β`` por máxima verossimilhança nas infecções novas dos dias 28 a 34;
* prevê a probabilidade de cada planta sadia pegar nos dias 35 a 41;

e é julgado pela log-perda e pela AUC dessa previsão. Modelos: ``contato`` (o núcleo do gêmeo
hoje), ``vento`` (a dispersão com o vento certo), ``vento_errado`` (direção invertida: o
anemômetro ou o azimute errado) e, num segundo talhão **sem** componente aérea, os mesmos —
para ver se o vento atrapalha quando não precisa.

É dado sintético gerado com a mesma física do modelo: mede se a peça funciona e quanto a
direção do vento importa quando ele existe, **não** que a ferrugem do café se espalha assim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from app.gemeo import clima, grafo
from app.gemeo.dispersao import NucleoComVento, ParametrosDispersao
from app.gemeo.epidemia import nucleo_de_contagio, parametros_para

RUAS, POR_RUA = 30, 100
LATENCIA, DIAS = 10, 49
JANELA_AJUSTE, JANELA_TESTE = (28, 35), (35, 42)


def talhao():
    plantas = [{"plant_id": f"G{r + 1}-{i:03d}", "row_id": f"G{r + 1}", "x_along_m": i * 0.7,
                "observations": {"p0": {"time": "2026-01-01T10:00:00+00:00", "n_tiles": 1,
                                        "posterior": {"Rust": 0.05}}}}
               for r in range(RUAS) for i in range(POR_RUA)]
    return grafo.construir({"field_id": "s", "crop": "coffee", "diseases": ["Healthy", "Rust"],
                            "priors": {"Healthy": 0.95, "Rust": 0.05}, "plant_spacing_m": 0.7,
                            "row_spacing_m": 3.5, "passes": [{"pass_id": "p0"}],
                            "plants": plantas})


def dias_de_vento(rng, n: int, de_graus: float, ms: float) -> list[clima.DiaClimatico]:
    saida = []
    for _ in range(n):
        direcao = de_graus + rng.normal(0, 20)
        v = max(0.3, ms + rng.normal(0, 0.8))
        saida.append(clima.DiaClimatico(date(2026, 1, 1), 22, 17, 27, 85, 2, 10, vento_ms=v,
                                        vento_u_ms=-v * math.sin(math.radians(direcao)),
                                        vento_v_ms=-v * math.cos(math.radians(direcao))))
    return saida


def simula(g, nucleo, dias, beta: float, fundo: float, rng):
    """SEIR estocástico; devolve o dia em que cada planta pegou (-1 = nunca) e o estado I(t)."""
    n = g.n
    pegou = np.full(n, -1)
    focos = rng.choice(n, 5, replace=False)
    pegou[focos] = -LATENCIA
    infecciosa = np.zeros((DIAS, n), dtype=bool)
    for d in range(DIAS):
        esporula = (pegou >= 0) & (d - pegou >= LATENCIA) | (pegou == -LATENCIA)
        infecciosa[d] = esporula
        forca = beta * nucleo.para(dias[d])(esporula.astype(float)) + fundo
        novas = (pegou == -1) & (rng.random(n) < 1 - np.exp(-forca))
        pegou[novas] = d
    return pegou, infecciosa


def _pressao_acumulada(nucleo, dias, infecciosa, janela) -> np.ndarray:
    return sum(nucleo.para(dias[d])(infecciosa[d].astype(float)) for d in range(*janela))


def _ajusta_beta(nucleo, dias, pegou, infecciosa, fundo: float) -> tuple[float, float]:
    """``(β, log-verossimilhança)`` na janela de ajuste, por varredura."""
    a0, a1 = JANELA_AJUSTE
    sus = (pegou == -1) | (pegou >= a0)
    y = ((pegou >= a0) & (pegou < a1))[sus]
    pa = _pressao_acumulada(nucleo, dias, infecciosa, JANELA_AJUSTE)[sus]
    melhor, beta = -np.inf, 0.0
    for b in np.geomspace(1e-4, 1.0, 120):
        p = np.clip(1 - np.exp(-(b * pa + fundo * (a1 - a0))), 1e-9, 1 - 1e-9)
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if ll > melhor:
            melhor, beta = ll, b
    return beta, melhor


def avalia(nucleo, dias, pegou, infecciosa, fundo: float) -> dict:
    a0, a1 = JANELA_AJUSTE
    t0, t1 = JANELA_TESTE
    sus_ajuste = (pegou == -1) | (pegou >= a0)
    y_ajuste = (pegou >= a0) & (pegou < a1)
    pa = _pressao_acumulada(nucleo, dias, infecciosa, JANELA_AJUSTE)[sus_ajuste]
    ya = y_ajuste[sus_ajuste]
    base = fundo * (a1 - a0)
    melhor, beta = -np.inf, 0.0
    for b in np.geomspace(1e-4, 1.0, 120):  # máxima verossimilhança por varredura
        p = np.clip(1 - np.exp(-(b * pa + base)), 1e-9, 1 - 1e-9)
        ll = float(np.sum(ya * np.log(p) + (1 - ya) * np.log(1 - p)))
        if ll > melhor:
            melhor, beta = ll, b
    sus = (pegou == -1) | (pegou >= t0)
    y = ((pegou >= t0) & (pegou < t1))[sus]
    pt = _pressao_acumulada(nucleo, dias, infecciosa, JANELA_TESTE)[sus]
    p = np.clip(1 - np.exp(-(beta * pt + fundo * (t1 - t0))), 1e-9, 1 - 1e-9)
    logperda = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ordem = np.argsort(p)
    postos = np.empty(len(p))
    postos[ordem] = np.arange(1, len(p) + 1)
    pos, neg = int(y.sum()), int((~y).sum())
    auc = float((postos[y].sum() - pos * (pos + 1) / 2) / max(1, pos * neg))
    return {"beta": round(beta, 5), "logperda": round(logperda, 5), "auc": round(auc, 4),
            "novas_no_teste": pos}


def experimento(semente: int) -> dict:
    rng = np.random.default_rng(semente)
    g = talhao()
    contato = nucleo_de_contagio(g, replace(parametros_para("Rust"), alcance_na_rua_m=1.8,
                                             alcance_entre_ruas_m=0.9))
    dias = dias_de_vento(rng, DIAS, 270.0, 3.0)
    errados = [replace(d, vento_u_ms=-d.vento_u_ms, vento_v_ms=-d.vento_v_ms) for d in dias]
    saida = {"semente": semente}
    for nome, omega in (("com_vento", 0.6), ("sem_vento", 0.0)):
        p = ParametrosDispersao(fracao_aerea=0.6)
        verdade = NucleoComVento(contato, g, replace(p, fracao_aerea=omega), 0.0)
        pegou, infecciosa = simula(g, verdade, dias, beta=0.10, fundo=2e-5, rng=rng)
        modelos = {"contato": (NucleoComVento(contato, g, replace(p, fracao_aerea=0.0), 0.0),
                               dias),
                   "vento": (NucleoComVento(contato, g, p, 0.0), dias),
                   "vento_errado": (NucleoComVento(contato, g, p, 0.0), errados)}
        # ω escolhido pelo próprio talhão: a verossimilhança da janela de ajuste decide.
        candidatos = {w: NucleoComVento(contato, g, replace(p, fracao_aerea=w), 0.0)
                      for w in (0.0, 0.3, 0.6, 0.9)}
        omega = max(candidatos, key=lambda w: _ajusta_beta(candidatos[w], dias, pegou,
                                                           infecciosa, 2e-5)[1])
        modelos["vento_omega_ajustado"] = (candidatos[omega], dias)
        saida[nome] = {"infectadas_no_fim": int((pegou >= 0).sum() + 5), "omega": omega,
                       **{m: avalia(nuc, ds, pegou, infecciosa, 2e-5)
                          for m, (nuc, ds) in modelos.items()}}
        print(semente, nome, {m: (v["logperda"], v["auc"]) for m, v in saida[nome].items()
                              if isinstance(v, dict)}, flush=True)
    return saida


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--saida", type=Path, default=None)
    args = p.parse_args(argv)
    execucoes = [experimento(s) for s in args.sementes]
    resumo = {}
    for cenario in ("com_vento", "sem_vento"):
        resumo[cenario] = {}
        resumo[cenario]["omega_escolhido"] = [e[cenario]["omega"] for e in execucoes]
        for m in ("contato", "vento", "vento_errado", "vento_omega_ajustado"):
            lp = [e[cenario][m]["logperda"] for e in execucoes]
            auc = [e[cenario][m]["auc"] for e in execucoes]
            resumo[cenario][m] = {"logperda": (round(float(np.mean(lp)), 5),
                                               round(float(np.std(lp)), 5)),
                                  "auc": (round(float(np.mean(auc)), 4),
                                          round(float(np.std(auc)), 4))}
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    if args.saida:
        args.saida.write_text(json.dumps({"resumo": resumo, "execucoes": execucoes},
                                         ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
