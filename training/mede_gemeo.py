"""Mede o gêmeo digital num talhão sintético onde a verdade é conhecida.

    python -m training.mede_gemeo --saida mede_gemeo.json

O talhão **não** é gerado pelo próprio modelo (seria medir o modelo contra ele mesmo):

* a epidemia é **estocástica e binária**, planta a planta — cada pé está em S, E, I ou R, e a
  cada dia sorteia a transição com as taxas da física do gêmeo (:mod:`app.gemeo.epidemia`)
  calculadas com os vizinhos que **de fato** esporulam. O gêmeo carrega probabilidades (campo
  médio); a diferença entre os dois é o erro de modelo que o campo vai ter;
* cada passada cobre parte do talhão (``cobertura``) e leva dois dias;
* o Rover sinaliza a planta que esporula com probabilidade ``sensibilidade`` e a sadia com
  ``alarme_falso``; a fusão vê o sinal através de um classificador com LLR gaussiano
  (``x ~ N(±mu, sd)``, LLR verdadeiro ``2.mu.x/sd²``) — uma fusão **calibrada**, como a do Rover
  depois da calibração por planta;
* o tempo é sintético, com chuva e molhamento;
* ``--baixada F`` multiplica por ``F`` a suscetibilidade de um trecho do talhão (seis ruas por
  30 m): a baixada úmida, a variedade sensível — o que é do talhão e a física não sabe. É o
  cenário em que a rede residual (:mod:`app.gemeo.rede`) tem o que aprender; sem ele (``F =
  1``) o portão deve recusá-la.

Mede (1) a recuperação dos parâmetros com 3, 5 e 7 passadas e (2) a previsão da passada que o
ajuste não viu, contra a persistência (a última passada repetida) e a persistência suavizada
pelos vizinhos — se o gêmeo não ganhar delas, ele não serve.

Só mede: nada aqui treina modelo que vá para produção. Tudo em CPU, segundos.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.gemeo import ajuste, epidemia, grafo
from app.gemeo.clima import DiaClimatico, SerieClimatica

INICIO = date(2025, 11, 1)


@dataclass
class TalhaoSintetico:
    observacoes: dict[str, Any]
    """No formato de ``PlantFusionStore.pass_observations``."""
    serie: SerieClimatica
    verdade: epidemia.ParametrosDoenca
    estados: np.ndarray
    """``(dias + 1, N)``: o estado verdadeiro de cada planta (0 S, 1 E, 2 I, 3 R)."""
    dias_passadas: list[int]
    dias_drone: list[int] = field(default_factory=list)

    def ate(self, passadas: int, fontes: tuple[str, ...] = ("rover", "drone")
            ) -> dict[str, Any]:
        """As ``passadas`` primeiras passadas do Rover e, das outras ``fontes``, as que vieram
        até o dia da última delas (a última assimilada é sempre do Rover)."""
        rover = [p for p in self.observacoes["passes"] if p.get("fonte", "rover") == "rover"]
        limite = self.dias_passadas[passadas - 1] if passadas else -1
        manter = {p["pass_id"] for p in rover[:passadas]} if "rover" in fontes else set()
        for p, d in zip([q for q in self.observacoes["passes"] if q.get("fonte") == "drone"],
                        self.dias_drone, strict=True):
            if "drone" in fontes and d <= limite:
                manter.add(p["pass_id"])
        return {**self.observacoes,
                "passes": [p for p in self.observacoes["passes"] if p["pass_id"] in manter],
                "plants": [{**p, "observations": {k: v for k, v in p["observations"].items()
                                                  if k in manter}}
                           for p in self.observacoes["plants"]]}


def clima_sintetico(dias: int, rng: np.random.Generator) -> list[DiaClimatico]:
    saida = []
    for d in range(dias):
        t = 21 + 2 * math.sin(d / 15) + rng.normal(0, 1)
        chuva = max(0.0, rng.normal(4, 8))
        molhamento = float(np.clip(rng.normal(8 if chuva > 1 else 4, 2), 0, 24))
        saida.append(DiaClimatico(INICIO + timedelta(days=d), t, t - 5, t + 6, 80.0, chuva,
                                  molhamento, fonte="sintetico"))
    return saida


def talhao_sintetico(*, ruas: int = 20, por_rua: int = 80, passadas: int = 7,
                     intervalo_dias: int = 14, focos: int = 4, cobertura: float = 0.9,
                     verdade: epidemia.ParametrosDoenca | None = None,
                     mu: float = 2.0, sigma: float = 1.5, prior_fusao: float = 0.1,
                     baixada: float = 1.0, drone_a_cada: int | None = None,
                     drone_desde: int = 7, canal_drone: epidemia.Canal | None = None,
                     semente: int = 0) -> TalhaoSintetico:
    rng = np.random.default_rng(semente)
    verdade = verdade or replace(epidemia.PRESETS["Rust"], beta=0.12, alcance_na_rua_m=1.8,
                                 alcance_entre_ruas_m=0.9, fundo=3e-5, alarme_falso=0.03,
                                 sensibilidade=0.85)
    dias = intervalo_dias * (passadas - 1) + 2
    clima = clima_sintetico(dias + 30, rng)
    plantas = [{"plant_id": f"G{r + 1}-{i:03d}", "row_id": f"G{r + 1}", "x_along_m": i * 0.7,
                "lat": None, "lon": None, "observations": {}}
               for r in range(ruas) for i in range(por_rua)]
    base = {"field_id": "sintetico", "crop": "coffee", "diseases": ["Healthy", "Rust"],
            "priors": {"Healthy": 1 - prior_fusao, "Rust": prior_fusao},
            "plant_spacing_m": 0.7, "row_spacing_m": 3.5}
    g = grafo.construir({**base, "passes": [{"pass_id": "p0"}],
                         "plants": [{**p, "observations": {"p0": {
                             "time": "2025-11-01T10:00:00+00:00", "n_tiles": 1,
                             "posterior": {"Rust": prior_fusao}}}} for p in plantas]})
    nucleo = epidemia.nucleo_de_contagio(g, verdade)
    suscetibilidade = np.where((g.rua >= ruas // 2) & (g.rua < ruas // 2 + 6)
                               & (g.x_m >= 10.0) & (g.x_m < 40.0), baixada, 1.0)
    estado = np.zeros(g.n, dtype=np.int8)
    estado[rng.choice(g.n, focos, replace=False)] = 2
    estados = [estado.copy()]
    for d in range(dias):
        dia = clima[d]
        fav = epidemia.favorabilidade(dia, verdade)
        forca = suscetibilidade * (
            verdade.beta * fav * epidemia.pressao(nucleo, (estado == 2).astype(float))
            + verdade.fundo * fav)
        sorteio = rng.random(g.n)
        novo = estado.copy()
        novo[(estado == 0) & (sorteio < 1 - np.exp(-forca))] = 1
        novo[(estado == 1) & (sorteio < 1 - math.exp(-1 / epidemia.latencia_dias(dia.t_media,
                                                                                verdade)))] = 2
        novo[(estado == 2) & (sorteio < 1 - math.exp(-1 / verdade.infecciosidade_dias))] = 3
        estado = novo
        estados.append(estado.copy())
    estados_arr = np.array(estados)
    dias_passadas = [k * intervalo_dias for k in range(passadas)]
    inicio_hora = datetime(2025, 11, 1, 10, tzinfo=timezone.utc)
    passes = []
    for k, d0 in enumerate(dias_passadas):
        pid = f"p{k}"
        passes.append({"pass_id": pid, "fonte": "rover",
                       "first_time": (inicio_hora + timedelta(days=d0)).isoformat()})
        vista = rng.random(g.n) < cobertura
        dia_visita = d0 + (np.arange(g.n) // por_rua >= ruas // 2)  # metade no dia seguinte
        est = estados_arr[dia_visita, np.arange(g.n)]
        sinal = rng.random(g.n) < np.where(est == 2, verdade.sensibilidade, verdade.alarme_falso)
        x = rng.normal(np.where(sinal, mu, -mu), sigma)
        llr = 2 * mu * x / sigma**2
        post = 1 / (1 + np.exp(-(math.log(prior_fusao / (1 - prior_fusao)) + llr)))
        for i, p in enumerate(plantas):
            if vista[i]:
                quando = datetime(2025, 11, 1, 10, tzinfo=timezone.utc) + timedelta(
                    days=int(dia_visita[i]))
                p["observations"][pid] = {"time": quando.isoformat(), "n_tiles": 40,
                                          "posterior": {"Healthy": float(1 - post[i]),
                                                        "Rust": float(post[i])}}
    # O drone (depois do Rover no sorteio: sem drone, o talhão é o mesmo de antes): sobrevoa
    # o talhão inteiro num dia, com o canal dele — a copa vista de cima.
    dias_drone: list[int] = []
    if drone_a_cada:
        cd = canal_drone or epidemia.CANAIS["drone"]
        dias_drone = list(range(drone_desde, dias + 1, drone_a_cada))
        for j, d in enumerate(dias_drone):
            pid = f"d{j}"
            passes.append({"pass_id": pid, "fonte": "drone",
                           "first_time": (inicio_hora + timedelta(days=d)).isoformat()})
            est = estados_arr[d]
            sinal = rng.random(g.n) < np.where(est == 2, cd.sensibilidade, cd.alarme_falso)
            x = rng.normal(np.where(sinal, mu, -mu), sigma)
            post = 1 / (1 + np.exp(-(math.log(prior_fusao / (1 - prior_fusao))
                                     + 2 * mu * x / sigma**2)))
            for i, p in enumerate(plantas):
                p["observations"][pid] = {"time": (inicio_hora + timedelta(days=d)).isoformat(),
                                          "n_tiles": 1,
                                          "posterior": {"Healthy": float(1 - post[i]),
                                                        "Rust": float(post[i])}}
        passes.sort(key=lambda q: q["first_time"])
    serie = SerieClimatica(None, None, {c.data: c for c in clima})
    return TalhaoSintetico({**base, "passes": passes, "plants": plantas}, serie, verdade,
                           estados_arr, dias_passadas, dias_drone)


def recuperacao(talhao: TalhaoSintetico, passadas: int, max_iter: int = 200,
                meia_vida_pares: float | None = None
                ) -> tuple[dict[str, Any], epidemia.ModeloAjustado]:
    g = grafo.construir(talhao.ate(passadas))
    t0 = time.perf_counter()
    m = ajuste.ajustar(g, "Rust", talhao.serie, max_iter=max_iter,
                       meia_vida_pares=meia_vida_pares)
    return {"passadas": passadas, "segundos": round(time.perf_counter() - t0, 1),
            "identificado": m.diagnostico["identificado"], "laplace": m.diagnostico["laplace"],
            "parametros": {k: {"verdade": getattr(talhao.verdade, k),
                               "prior": getattr(epidemia.PRESETS["Rust"], k),
                               "ajuste": round(getattr(m.parametros, k), 6),
                               "intervalo_95": m.diagnostico["intervalo_95"][k]}
                           for k in ajuste.LIVRES},
            "perda_prior": m.diagnostico["perda_por_planta_prior"],
            "perda_ajuste": m.diagnostico["perda_por_planta_ajuste"]}, m


def auc(rotulo: np.ndarray, escore: np.ndarray) -> float | None:
    """Área sob a curva ROC (Mann-Whitney, empates pela média dos postos)."""
    pos, neg = int(rotulo.sum()), int((~rotulo).sum())
    if not pos or not neg:
        return None
    ordem = np.argsort(escore, kind="stable")
    postos = np.empty(len(escore))
    s = escore[ordem]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        postos[ordem[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((postos[rotulo].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _z(escore: np.ndarray) -> np.ndarray:
    e = np.clip(escore, 1e-4, 1 - 1e-4)
    return np.log(e / (1 - e))


def platt(escore: np.ndarray, razao: np.ndarray, iteracoes: int = 50) -> tuple[float, float]:
    """Os dois parâmetros que recalibram um escore em P(sinal), pela verossimilhança da
    evidência ``razao``."""
    z = _z(escore)
    a, b = 1.0, 0.0
    for _ in range(iteracoes):
        melhor = None
        for da, db in ((0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05), (0, 0)):
            q = 1 / (1 + np.exp(-(a + da) * z - (b + db)))
            v = float(epidemia.log_verossimilhanca(q, razao).sum())
            if melhor is None or v > melhor[0]:
                melhor = (v, a + da, b + db)
        if melhor[1] == a and melhor[2] == b:
            break
        a, b = melhor[1], melhor[2]
    return a, b


def aplica_platt(escore: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return 1 / (1 + np.exp(-ab[0] * _z(escore) - ab[1]))


def previsao_da_passada(talhao: TalhaoSintetico, passadas: int,
                        modelo: epidemia.ModeloAjustado, membros: int = 16) -> dict[str, Any]:
    """Prevê a passada ``passadas`` (a primeira que o ajuste não viu) e compara com as bases."""
    from app.gemeo import previsao, rede
    from app.prescricao import Contagio, _nucleo, _risco

    g = grafo.construir(talhao.ate(passadas))
    alvo = grafo.construir(talhao.ate(passadas + 1))
    razao = alvo.razao("Rust")[passadas]
    vista = alvo.dia_obs[passadas] >= 0
    dia_alvo = alvo.dia_da_passada(passadas)
    base_dia = epidemia.dia_da_ultima_assimilacao(g)
    data_base = g.inicio + timedelta(days=base_dia)
    horizonte = dia_alvo - base_dia
    escores: dict[str, np.ndarray] = {}
    riscos: dict[str, np.ndarray] = {}
    for nome, m in (("gemeo_prior", epidemia.ModeloAjustado(epidemia.PRESETS["Rust"])),
                    ("gemeo_ajustado", modelo)):
        prev = previsao.prever(g, "Rust", m, talhao.serie, hoje=data_base,
                               horizonte_dias=horizonte, membros=membros)
        escores[nome] = prev.sinal_no_horizonte
        riscos[nome] = prev.risco
    clima = talhao.serie.periodo(g.inicio, g.inicio + timedelta(days=dia_alvo + 2))
    hist = rede.historico(g, "Rust", clima, modelo.parametros)
    residual, diag_rede = rede.ajustar_rede(g, hist, modelo.parametros)
    if residual is not None:
        prev = previsao.prever(g, "Rust", modelo, talhao.serie, hoje=data_base,
                               horizonte_dias=horizonte, membros=membros, rede=residual)
        escores["gemeo_corrigido"] = (prev.sinal_corrigido if prev.sinal_corrigido is not None
                                      else prev.sinal_no_horizonte)
        riscos["gemeo_corrigido"] = prev.risco
    # As bases viram P(sinal) por Platt calibrado no par anterior (passada t-2 prevendo a
    # t-1): é o que um sistema em campo conseguiria. ``calibrada_no_alvo`` repete com Platt
    # ajustado na própria passada avaliada — o teto otimista delas, só para transparência.
    def _post(k: int) -> np.ndarray:
        linha = g.posterior["Rust"][k]
        return np.nan_to_num(linha, nan=float(np.nanmean(linha)))

    nucleo = _nucleo(g.x_m, g.rua.astype(float), g.espacamento_entre_ruas_m, Contagio())
    anterior, penultima = _post(g.t - 1), _post(g.t - 2)
    razao_calibracao = g.razao("Rust")[g.t - 1]
    bases = {"persistencia": (anterior, penultima),
             "heuristica_do_produto": (_risco(anterior, nucleo, np.ones(g.n))[0],
                                       _risco(penultima, nucleo, np.ones(g.n))[0])}
    no_alvo: dict[str, np.ndarray] = {}
    for nome, (agora, antes) in bases.items():
        escores[nome] = aplica_platt(agora, platt(antes, razao_calibracao))
        no_alvo[nome] = aplica_platt(agora, platt(agora, razao))
        riscos[nome] = agora
    # Rótulos, entre as plantas que a última passada deu como limpas (posterior < 0,5):
    # (a) sinal novo na passada avaliada — o que o Rover vai ver;
    # (b) a verdade — infectada (latente ou esporulando) na data avaliada, inclusive a que o
    #     Rover ainda não vê. É a pergunta do produtor: onde a doença está escondida.
    sinal_antes = anterior >= 0.5
    sinal_novo = vista & (np.nan_to_num(alvo.posterior["Rust"][passadas]) >= 0.5) & ~sinal_antes
    est_alvo = talhao.estados[dia_alvo]
    candidatas = ~sinal_antes
    infectada = np.isin(est_alvo, (1, 2))
    saida: dict[str, Any] = {"passadas_no_ajuste": passadas, "horizonte_dias": horizonte,
                             "plantas_avaliadas": int(vista.sum()),
                             "sinais_novos": int(sinal_novo.sum()),
                             "infectadas_ocultas_verdade": int((infectada & candidatas).sum()),
                             "latentes_verdade": int(((est_alvo == 1) & candidatas).sum()),
                             "rede": diag_rede, "modelos": {}}
    for nome, q in escores.items():
        saida["modelos"][nome] = {
            "log_verossimilhanca_por_planta": round(float(
                epidemia.log_verossimilhanca(q[vista], razao[vista]).mean()), 5),
            "auc_sinal_novo": _r(auc(sinal_novo[candidatas & vista], q[candidatas & vista])),
            "auc_infectada_oculta_verdade": _r(auc(infectada[candidatas],
                                                   riscos[nome][candidatas])),
        }
        if nome in no_alvo:
            saida["modelos"][nome]["log_verossimilhanca_calibrada_no_alvo"] = round(float(
                epidemia.log_verossimilhanca(no_alvo[nome][vista], razao[vista]).mean()), 5)
    return saida


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--passadas", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--membros", type=int, default=16)
    ap.add_argument("--meia-vida", type=float, default=None,
                    help="desconto dos pares antigos no ajuste (pares); padrão: sem desconto")
    ap.add_argument("--baixada", type=float, default=1.0,
                    help="fator de suscetibilidade do trecho que a física não conhece")
    ap.add_argument("--saida", type=Path)
    args = ap.parse_args()
    resultado: dict[str, Any] = {"baixada": args.baixada, "sementes": []}
    for semente in args.sementes:
        talhao = talhao_sintetico(passadas=max(args.passadas) + 1, semente=semente,
                                  baixada=args.baixada)
        esporulando = [int((talhao.estados[d] == 2).sum()) for d in talhao.dias_passadas]
        print(f"\n=== semente {semente}: esporulando (verdade) por passada {esporulando}")
        rodada: dict[str, Any] = {"semente": semente, "esporulando_por_passada": esporulando,
                                  "recuperacao": [], "previsao": []}
        for n in args.passadas:
            r, modelo = recuperacao(talhao, n, meia_vida_pares=args.meia_vida)
            rodada["recuperacao"].append(r)
            print(f"\n{n} passadas ({r['segundos']}s) identificado={r['identificado']} "
                  f"laplace={r['laplace']} perda {r['perda_prior']} -> {r['perda_ajuste']}")
            for k, v in r["parametros"].items():
                print(f"  {k:22s} verdade {v['verdade']:<9.4g} prior {v['prior']:<9.4g} "
                      f"ajuste {v['ajuste']:<9.4g} IC95 {v['intervalo_95']}")
            pv = previsao_da_passada(talhao, n, modelo, args.membros)
            rodada["previsao"].append(pv)
            print(f"  previsão da passada {n + 1} ({pv['horizonte_dias']} dias; "
                  f"{pv['sinais_novos']} sinais novos; {pv['infectadas_ocultas_verdade']} "
                  f"infectadas entre as limpas, {pv['latentes_verdade']} latentes):")
            print(f"    correção: {pv['rede'].get('escolha')} "
                  f"(ganho retido {pv['rede'].get('ganho_por_planta')})")
            for nome, v in pv["modelos"].items():
                print(f"    {nome:22s} logv {v['log_verossimilhanca_por_planta']:<9} "
                      f"AUC sinal novo {v['auc_sinal_novo']}  "
                      f"AUC infectada oculta {v['auc_infectada_oculta_verdade']}  "
                      f"(no alvo {v.get('log_verossimilhanca_calibrada_no_alvo', '-')})")
        resultado["sementes"].append(rodada)
    if args.saida:
        args.saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
