"""Mede o que o enxame acrescenta, num talhão sintético onde a verdade é conhecida.

    python -m training.mede_enxame --saida mede_enxame.json

Dois experimentos, no gerador de ``training.mede_gemeo`` (epidemia estocástica binária
planta a planta, fusão calibrada):

**A — fusão entre dispositivos.** O Rover passa a cada 14 dias; o drone sobrevoa a cada 7,
com o canal dele (a copa vista de cima: sensibilidade 0,45, alarme falso 6% — a suposição de
projeto de :data:`app.gemeo.epidemia.CANAIS`). Ajusta e prevê a passada seguinte do Rover:

* só o Rover;
* Rover e drone, cada um com o seu canal (a fusão do gêmeo);
* Rover e drone **com o canal do Rover** para os dois — a fusão ingênua, que trata uma
  fonte pior como se fosse boa.

**B — priorização da inspeção.** Com o mesmo orçamento (uma fração dos trechos de rua de
20 m), qual política acha mais doença e decide melhor? O valor da informação
(:mod:`app.gemeo.prioridade`), a informação em bits, só o risco, só a incerteza, a varredura
em ordem (o que se faz hoje) e o sorteio. A inspeção é simulada no dia da visita com a
verdade; a decisão por planta é a mesma para todas (tratar se P(infectada) passa do limiar
dos números do produtor); a perda é contada com a verdade.

Só mede. CPU, uma thread.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from app.gemeo import ajuste, epidemia, grafo, previsao, prioridade
from training.mede_gemeo import TalhaoSintetico, auc, talhao_sintetico

ECONOMIAS = {
    # limiar c/(L e) ~ 6%: tratar é barato perto do que se perde — trata-se quase tudo.
    "barato": prioridade.Economia(custo_tratar=2.0, perda_planta=40.0, eficacia=0.85),
    # limiar ~ 16%: a perda por planta menor (lavoura de produtividade média) — a dúvida pesa.
    "caro": prioridade.Economia(custo_tratar=2.0, perda_planta=15.0, eficacia=0.85),
}


def _como_rover(observacoes: dict[str, Any]) -> dict[str, Any]:
    return {**observacoes, "passes": [{**p, "fonte": "rover"} for p in observacoes["passes"]]}


def fusao(talhao: TalhaoSintetico, passadas: int, membros: int = 8) -> dict[str, Any]:
    alvo_obs = talhao.ate(passadas + 1, fontes=("rover",))
    alvo = grafo.construir(alvo_obs)
    k_alvo = alvo.t - 1
    razao = alvo.razao("Rust")[k_alvo]
    vista = alvo.dia_obs[k_alvo] >= 0
    dia_alvo = alvo.dia_da_passada(k_alvo)
    infectada = np.isin(talhao.estados[dia_alvo], (1, 2))
    variantes = {
        "so_rover": talhao.ate(passadas, fontes=("rover",)),
        "rover_e_drone": talhao.ate(passadas),
        "rover_e_drone_canal_errado": _como_rover(talhao.ate(passadas)),
    }
    saida: dict[str, Any] = {"passadas_rover": passadas,
                             "passadas_drone": sum(d <= talhao.dias_passadas[passadas - 1]
                                                   for d in talhao.dias_drone)}
    for nome, obs in variantes.items():
        g = grafo.construir(obs)
        m = ajuste.ajustar(g, "Rust", talhao.serie, max_iter=150)
        base_dia = epidemia.dia_da_ultima_assimilacao(g)
        data_base = g.inicio + timedelta(days=base_dia)
        prev = previsao.prever(g, "Rust", m, talhao.serie, hoje=data_base,
                               horizonte_dias=dia_alvo - base_dia, membros=membros)
        anterior = np.nan_to_num(alvo.posterior["Rust"][k_alvo - 1], nan=0.0)
        limpas = anterior < 0.5
        saida[nome] = {
            "log_verossimilhanca_por_planta": round(float(epidemia.log_verossimilhanca(
                prev.sinal_no_horizonte[vista], razao[vista]).mean()), 5),
            "auc_infectada_oculta": _r(auc(infectada[limpas], prev.risco[limpas])),
            "sensibilidade_rover_ajustada": round(m.parametros.sensibilidade, 4),
            "alarme_falso_rover_ajustado": round(m.parametros.alarme_falso, 4),
        }
    return saida


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def _posterior_de_d(p: np.ndarray, q: np.ndarray, canal: epidemia.Canal,
                    razao: np.ndarray) -> np.ndarray:
    """P(D | evidência) com a razão da fusão como evidência virtual (o que o sistema faria)."""
    l1, l0 = prioridade.verossimilhancas(p, q, canal)
    v1 = l1 * razao + (1 - l1)
    v0 = l0 * razao + (1 - l0)
    return p * v1 / np.maximum(p * v1 + (1 - p) * v0, 1e-12)


def priorizacao(talhao: TalhaoSintetico, passadas: int, fracoes: tuple[float, ...],
                economia: prioridade.Economia, dias_ate_visita: int = 3, membros: int = 8,
                semente: int = 0, mu: float = 2.0, sigma: float = 1.5,
                sorteios: int = 20) -> dict[str, Any]:
    rng = np.random.default_rng(semente + 1000)
    g = grafo.construir(talhao.ate(passadas, fontes=("rover",)))
    m = ajuste.ajustar(g, "Rust", talhao.serie, max_iter=150)
    base_dia = epidemia.dia_da_ultima_assimilacao(g)
    data_base = g.inicio + timedelta(days=base_dia)
    prev = previsao.prever(g, "Rust", m, talhao.serie, hoje=data_base,
                           horizonte_dias=dias_ate_visita, membros=membros)
    canal = epidemia.canal_de(m.parametros)
    com_economia = prioridade.priorizar(g, prev, m, canal, nome_canal="rover",
                                        economia=economia)
    sem_economia = prioridade.priorizar(g, prev, m, canal, nome_canal="rover")
    dia_visita = base_dia + dias_ate_visita
    verdade = talhao.estados[dia_visita]
    infectada = np.isin(verdade, (1, 2))
    esporula = verdade == 2
    # A inspeção simulada: o sinal de cada planta (canal verdadeiro do Rover) e a LLR da fusão.
    v = talhao.verdade
    sinal = rng.random(g.n) < np.where(esporula, v.sensibilidade, v.alarme_falso)
    x = rng.normal(np.where(sinal, mu, -mu), sigma)
    razao_inspecao = np.exp(np.clip(2 * mu * x / sigma**2, -50, 50))
    p = np.clip(prev.risco, 0, 1)
    q = np.minimum(np.clip(prev.esporulando_no_horizonte, 0, 1), p)
    consequencia = prioridade.consequencia_por_contagio(
        epidemia.nucleo_de_contagio(g, m.parametros), 1 - p, m.parametros.beta,
        float(np.mean([d["favorabilidade"] for d in prev.diario])), dias_ate_visita)
    limiar = economia.custo_tratar / (economia.perda_planta * economia.eficacia * consequencia)

    def resultado(tarefas: list[prioridade.Tarefa]) -> dict[str, float]:
        inspecionadas = np.zeros(g.n, dtype=bool)
        for t in tarefas:
            inspecionadas[t.plantas] = True
        post = np.where(inspecionadas, _posterior_de_d(p, q, canal, razao_inspecao), p)
        tratar = post > limiar
        perda = (economia.custo_tratar * tratar
                 + economia.perda_planta * consequencia * infectada
                 * np.where(tratar, 1 - economia.eficacia, 1.0))
        achadas = inspecionadas & infectada & tratar
        return {"perda": float(perda.sum()), "infectadas_achadas": int(achadas.sum()),
                "plantas_inspecionadas": int(inspecionadas.sum()),
                "plantas_tratadas": int(tratar.sum())}

    tarefas = com_economia.tarefas
    ordens = {
        "valor_da_informacao": tarefas,
        "informacao_bits": sem_economia.tarefas,
        "so_risco": sorted(tarefas, key=lambda t: -t.risco_medio),
        "so_incerteza": sorted(tarefas, key=lambda t: -t.incerteza_bits),
        "varredura_em_ordem": sorted(tarefas, key=lambda t: (t.ruas[0] not in g.ruas
                                                            or g.ruas.index(t.ruas[0]),
                                                            t.x_inicio_m)),
    }
    sem_inspecao = resultado([])
    saida: dict[str, Any] = {
        "trechos": len(tarefas), "infectadas_verdade": int(infectada.sum()),
        "infectadas_que_esporulam": int(esporula.sum()),
        "limiar_mediano": round(float(np.median(limiar)), 4),
        "sem_inspecao": sem_inspecao, "por_orcamento": {}}
    for f in fracoes:
        n = max(1, int(round(f * len(tarefas))))
        linha: dict[str, Any] = {"trechos_inspecionados": n}
        for nome, ordem in ordens.items():
            linha[nome] = resultado(list(ordem[:n]))
        sorteados = [resultado([tarefas[i] for i in rng.choice(len(tarefas), n, replace=False)])
                     for _ in range(sorteios)]
        linha["sorteio"] = {k: float(np.mean([s[k] for s in sorteados])) for k in sorteados[0]}
        saida["por_orcamento"][f"{f:.0%}"] = linha
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--passadas", type=int, default=5)
    ap.add_argument("--fracoes", type=float, nargs="+", default=[0.15, 0.30])
    ap.add_argument("--baixada", type=float, default=1.0)
    ap.add_argument("--so", choices=["fusao", "priorizacao"], default=None)
    ap.add_argument("--economia", choices=sorted(ECONOMIAS), default="caro")
    ap.add_argument("--saida", type=Path)
    args = ap.parse_args()
    resultado: dict[str, Any] = {"baixada": args.baixada, "economia": args.economia,
                                 "sementes": []}
    for semente in args.sementes:
        rodada: dict[str, Any] = {"semente": semente}
        if args.so in (None, "fusao"):
            t = talhao_sintetico(passadas=args.passadas + 1, drone_a_cada=7, semente=semente,
                                 baixada=args.baixada)
            rodada["fusao"] = f = fusao(t, args.passadas)
            print(f"\n=== semente {semente} — fusão ({f['passadas_rover']} passadas do Rover, "
                  f"{f['passadas_drone']} do drone)")
            for nome in ("so_rover", "rover_e_drone", "rover_e_drone_canal_errado"):
                print(f"  {nome:28s} {f[nome]}")
        if args.so in (None, "priorizacao"):
            # Uma passada a mais no gerador: dias depois da última, para a visita simulada.
            t = talhao_sintetico(passadas=args.passadas + 1, semente=semente, baixada=args.baixada)
            rodada["priorizacao"] = pr = priorizacao(t, args.passadas, tuple(args.fracoes),
                                                     ECONOMIAS[args.economia], semente=semente)
            print(f"\n=== semente {semente} — priorização ({pr['trechos']} trechos, "
                  f"{pr['infectadas_verdade']} infectadas de verdade, "
                  f"{pr['infectadas_que_esporulam']} esporulando); sem inspeção: "
                  f"perda {pr['sem_inspecao']['perda']:.0f}")
            for f, linha in pr["por_orcamento"].items():
                print(f"  orçamento {f} ({linha['trechos_inspecionados']} trechos):")
                for nome, r in linha.items():
                    if isinstance(r, dict):
                        print(f"    {nome:22s} perda {r['perda']:8.1f}  achadas "
                              f"{r['infectadas_achadas']:6.1f}")
        resultado["sementes"].append(rodada)
    if args.saida:
        args.saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
