"""Ajuste do modelo epidemiológico às passadas do talhão.

Livres: ``β`` (transmissão), ``a_rua`` e ``a_entre`` (alcance do contágio), ``fundo`` (inóculo
de fora), o alarme falso e a sensibilidade por planta. Fixos: os biológicos da doença
(:data:`app.gemeo.epidemia.PRESETS`) — temperatura, molhamento, latência, período infeccioso —
até haver série longa que os identifique.

* **Filtro e previsão um passo à frente.** O estado atravessa as passadas: cada uma é
  assimilada por Bayes sobre o estado que o modelo trouxe
  (:func:`~app.gemeo.epidemia.estado_da_observacao`) e o modelo prevê a seguinte. A perda é
  menos a log-verossimilhança preditiva da evidência da fusão
  (:func:`~app.gemeo.epidemia.log_verossimilhanca`) — a verossimilhança do filtro, sem
  inventar alvo: a razão que a fusão somou é exatamente o que ela sabe da planta.
* **MAP, não máxima verossimilhança.** Um prior normal no espaço livre em volta do preset
  (log para taxas e alcances, logit para as probabilidades — :func:`~app.gemeo.epidemia.
  para_livre`) segura o ajuste quando há duas ou três passadas.
* **Nelder-Mead** no espaço livre, com limites (alcances até
  :data:`~app.gemeo.grafo.ALCANCE_MAXIMO_M`), polido por Levenberg-Marquardt. Sem SciPy: a API
  não depende dele.
* **Laplace**: a Hessiana numérica no ótimo dá a covariância no espaço livre — a incerteza
  que a previsão (:mod:`app.gemeo.previsao`) propaga por um conjunto de parâmetros. Um
  parâmetro encostado no limite não tem curvatura honesta: a Hessiana é feita só nos que
  ficaram dentro, e os do limite levam a variância do prior (e o diagnóstico diz quais).

``diagnostico['identificado']`` diz se os dados moveram os parâmetros para longe do prior com
ganho real de ajuste; quando não, a previsão avisa (o modelo usado é o mesmo MAP, perto do
prior, com a incerteza de Laplace).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import timedelta

import numpy as np

from app.gemeo.clima import DiaClimatico, SerieClimatica
from app.gemeo.epidemia import (
    ModeloAjustado,
    ParametrosDoenca,
    canal_de,
    de_livre,
    filtrar,
    log_verossimilhanca,
    para_livre,
    parametros_para,
)
from app.gemeo.grafo import ALCANCE_MAXIMO_M, GrafoLavoura

LIVRES = ("beta", "alcance_na_rua_m", "alcance_entre_ruas_m", "fundo", "alarme_falso",
          "sensibilidade")
SIGMA_LIVRE = {"beta": 1.0, "alcance_na_rua_m": 0.7, "alcance_entre_ruas_m": 0.7, "fundo": 1.5,
               "alarme_falso": 0.7, "sensibilidade": 1.0}
"""Desvio do prior no espaço livre: log (fator e^sigma) ou logit (sensibilidade 0,9 com sigma 1
cobre de 0,77 a 0,96)."""
LIMITES = {"beta": (1e-4, 5.0), "alcance_na_rua_m": (0.2, ALCANCE_MAXIMO_M),
           "alcance_entre_ruas_m": (0.1, ALCANCE_MAXIMO_M), "fundo": (1e-7, 0.05),
           "alarme_falso": (1e-4, 0.25), "sensibilidade": (0.3, 0.999)}
REINICIOS = 2
GANHO_IDENTIFICADO = 3.0
"""Melhora mínima do objetivo (log-verossimilhança + prior, em nats) para dizer que as passadas
moveram o ajuste."""
POLIMENTO = 6
PASSO_HESSIANA = 0.02


def perda_um_passo(grafo: GrafoLavoura, doenca: str, clima: Sequence[DiaClimatico],
                   p: ParametrosDoenca, meia_vida_pares: float | None = None
                   ) -> tuple[float, int]:
    """Menos a log-verossimilhança preditiva média por planta, e quantas plantas previstas
    (o filtro de :func:`~app.gemeo.epidemia.filtrar`).

    ``meia_vida_pares``: desconta os pares antigos (peso cai à metade a cada tantos pares).
    Medido (3 sementes, 16/10/2026) e **deixado desligado**: com 7 passadas, melhorou a previsão
    no talhão com baixada (0,247 para 0,259) e piorou no homogêneo (0,182 para 0,162). Fica
    para reproduzir a medição (``training.mede_gemeo --meia-vida``).
    """
    razao = grafo.razao(doenca)
    total, contagem = 0.0, 0
    for s in filtrar(grafo, razao, clima, p):
        if s.previsto_esporulando is None or s.vista_proxima is None or not s.vista_proxima.any():
            continue
        vistas = s.vista_proxima
        c = canal_de(p, grafo.fonte(s.k + 1))
        sinal = c.alarme_falso + (c.sensibilidade - c.alarme_falso) * s.previsto_esporulando[vistas]
        peso = 1.0 if meia_vida_pares is None else 0.5 ** ((grafo.t - 2 - s.k) / meia_vida_pares)
        total -= peso * float(log_verossimilhanca(sinal, razao[s.k + 1][vistas]).sum())
        contagem += int(vistas.sum())
    return (total / contagem if contagem else float("nan")), contagem


def nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray, passo: float = 0.4,
                max_iter: int = 250, tol: float = 1e-5) -> tuple[np.ndarray, float, int]:
    """Nelder-Mead padrão (reflexão 1, expansão 2, contração 0,5, encolhimento 0,5)."""
    n = len(x0)
    simplex = [x0] + [x0 + passo * np.eye(n)[i] for i in range(n)]
    valores = [f(x) for x in simplex]
    for it in range(max_iter):
        ordem = np.argsort(valores)
        simplex = [simplex[i] for i in ordem]
        valores = [valores[i] for i in ordem]
        if abs(valores[-1] - valores[0]) < tol * (abs(valores[0]) + tol):
            return simplex[0], valores[0], it
        centro = np.mean(simplex[:-1], axis=0)
        refletido = centro + (centro - simplex[-1])
        fr = f(refletido)
        if fr < valores[0]:
            expandido = centro + 2.0 * (centro - simplex[-1])
            fe = f(expandido)
            simplex[-1], valores[-1] = (expandido, fe) if fe < fr else (refletido, fr)
        elif fr < valores[-2]:
            simplex[-1], valores[-1] = refletido, fr
        else:
            contraido = centro + 0.5 * (simplex[-1] - centro)
            fc = f(contraido)
            if fc < valores[-1]:
                simplex[-1], valores[-1] = contraido, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (s - simplex[0])
                                          for s in simplex[1:]]
                valores = [valores[0]] + [f(s) for s in simplex[1:]]
    melhor = int(np.argmin(valores))
    return simplex[melhor], valores[melhor], max_iter


def _hessiana(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 0.05) -> np.ndarray:
    n = len(x)
    hess = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.eye(n)[i] * h
        hess[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h**2
        for j in range(i + 1, n):
            ej = np.eye(n)[j] * h
            v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h**2)
            hess[i, j] = hess[j, i] = v
    return hess


def _gradiente(f: Callable[[np.ndarray], float], x: np.ndarray, h: float) -> np.ndarray:
    return np.array([(f(x + e) - f(x - e)) / (2 * h) for e in np.eye(len(x)) * h])


def polir(f: Callable[[np.ndarray], float], x: np.ndarray, valor: float,
          iteracoes: int = POLIMENTO, h: float = PASSO_HESSIANA) -> tuple[np.ndarray, float]:
    """Passos de Newton amortecidos (Levenberg-Marquardt) a partir do Nelder-Mead.

    O objetivo é mal condicionado — o alarme falso é ~10⁴ vezes mais curvo que o resto
    (medido no talhão sintético, 16/10/2026) — e o simplex para fora do ótimo nessa direção
    estreita; a Hessiana ali não é positiva e a incerteza de Laplace sai errada.
    """
    amortecimento = 1e-3
    hess = None
    for _ in range(iteracoes):
        grad = _gradiente(f, x, h / 4)
        hess = _hessiana(f, x, h)
        aceito = False
        for _ in range(6):
            a = hess + amortecimento * np.diag(np.abs(np.diag(hess)) + 1.0)
            try:
                passo = -np.linalg.solve(a, grad)
            except np.linalg.LinAlgError:
                amortecimento *= 10
                continue
            novo = f(x + passo)
            if novo < valor:
                x, valor, aceito = x + passo, novo, True
                amortecimento = max(amortecimento / 10, 1e-6)
                break
            amortecimento *= 10
        if not aceito or float(np.abs(passo).max()) < 1e-3:
            break
    return x, valor


def laplace(f: Callable[[np.ndarray], float], x: np.ndarray, lo: np.ndarray, hi: np.ndarray,
            sigma_prior: np.ndarray, h: float = PASSO_HESSIANA
            ) -> tuple[np.ndarray, np.ndarray, str]:
    """A covariância no espaço livre, a máscara dos parâmetros com curvatura própria e de onde
    ela veio.

    Só entra na Hessiana quem está a mais de um passo ``h`` dos limites **no ponto final**: a
    diferença finita de um parâmetro encostado lê a penalidade de fora do domínio e estraga a
    matriz inteira (era o que derrubava a Laplace para o prior, e o conjunto da previsão
    sorteava ``β`` num fator 7 — medido no talhão sintético, 16/10/2026). Se a Hessiana não for
    positiva, a diagonal dela (a curvatura de cada parâmetro sozinho, sem as correlações); só
    o parâmetro sem curvatura positiva fica com a variância do prior.
    """
    dentro = (x - h > lo) & (x + h < hi)
    cov = np.diag(sigma_prior**2)
    if not dentro.any():
        return cov, dentro, "prior (nenhum parâmetro dentro dos limites)"
    idx = np.flatnonzero(dentro)

    def parcial(z: np.ndarray) -> float:
        completo = x.copy()
        completo[idx] = z
        return f(completo)

    hess = _hessiana(parcial, x[idx], h)
    hess = (hess + hess.T) / 2
    if np.all(np.linalg.eigvalsh(hess) > 0):
        sub = np.linalg.inv(hess)
        cov[np.ix_(idx, idx)] = (sub + sub.T) / 2
        return cov, dentro, "Hessiana no ótimo"
    diagonal = np.diag(hess)
    for j, i in enumerate(idx):
        if diagonal[j] > 0:
            cov[i, i] = 1.0 / diagonal[j]
        else:
            dentro[i] = False
    return cov, dentro, "diagonal da Hessiana (a matriz inteira não ficou positiva)"

FRACOES_AEREAS = (0.0, 0.3, 0.6, 0.9)
"""Candidatas a ``ω`` (parte da pressão que vai pelo ar): medido em talhão sintético, escolher
pela verossimilhança das passadas acerta o verdadeiro e evita o vento onde ele não existe
(``training/mede_dispersao.py``)."""


def escolher_dispersao(grafo: GrafoLavoura, doenca: str, serie: SerieClimatica,
                       modelo: ModeloAjustado, parametros_dispersao, azimute_rua_graus: float,
                       meia_vida_pares: float | None = None) -> dict:
    """A fração aérea ``ω`` que melhor prevê as passadas do talhão, com os parâmetros ajustados.

    Não entra no Nelder-Mead (a convolução por dia deixaria o ajuste lento demais para a API):
    é seleção de modelo depois do ajuste, com o ``β`` mantido — o núcleo do vento tem a mesma
    massa do de contato, então ``β`` continua significando o mesmo. Sem dia com vento medido
    no período, ``ω = 0`` (e diz por quê).
    """
    if grafo.t < 2:
        return {"fracao_aerea": 0.0, "motivo": "uma passada só: nada a comparar"}
    ultimo = int(grafo.dia_obs.max())
    clima = serie.periodo(grafo.inicio, grafo.inicio + timedelta(days=ultimo + 1))
    com_vento = sum(1 for d in clima if d.tem_vento)
    if com_vento < len(clima) / 2:
        return {"fracao_aerea": 0.0, "dias_com_vento": com_vento, "dias": len(clima),
                "motivo": "menos da metade dos dias com vento medido: só o contato"}
    tabela = {}
    original = grafo.dispersao
    try:
        for omega in FRACOES_AEREAS:
            grafo.dispersao = (replace(parametros_dispersao, fracao_aerea=omega),
                               float(azimute_rua_graus))
            perda, n = perda_um_passo(grafo, doenca, clima, modelo.parametros, meia_vida_pares)
            tabela[omega] = {"perda": round(float(perda), 6), "plantas": n}
    finally:
        grafo.dispersao = original
    melhor = min(tabela, key=lambda w: tabela[w]["perda"])
    return {"fracao_aerea": melhor, "azimute_rua_graus": float(azimute_rua_graus),
            "dias_com_vento": com_vento, "dias": len(clima),
            "por_fracao": {str(w): v for w, v in tabela.items()},
            "parametros": replace(parametros_dispersao, fracao_aerea=melhor).to_dict(),
            "aviso": "D, λ e a fração do vento no dossel são hipótese de projeto (não medidos);"
                     " a escolha de ω é sobre as passadas deste talhão"}


def ajustar(grafo: GrafoLavoura, doenca: str, serie: SerieClimatica,
            prior: ParametrosDoenca | None = None, max_iter: int = 250,
            meia_vida_pares: float | None = None) -> ModeloAjustado:
    """MAP dos parâmetros livres sobre as passadas do talhão, com Laplace em log."""
    prior = prior or parametros_para(doenca)
    if doenca not in grafo.posterior:
        raise KeyError(f"O talhão não tem posterior de {doenca!r}.")
    if grafo.t < 2:
        return ModeloAjustado(prior, diagnostico={
            "identificado": False, "passadas": grafo.t,
            "motivo": "uma passada só: nada a prever — o modelo usa o prior"})
    ultimo = int(grafo.dia_obs.max())
    clima = serie.periodo(grafo.inicio, grafo.inicio + timedelta(days=ultimo + 1))
    base = np.array([para_livre(k, getattr(prior, k)) for k in LIVRES])
    sigma = np.array([SIGMA_LIVRE[k] for k in LIVRES])
    lo = np.array([para_livre(k, LIMITES[k][0]) for k in LIVRES])
    hi = np.array([para_livre(k, LIMITES[k][1]) for k in LIVRES])
    contagem = {"n": 0}

    def com(x: np.ndarray) -> ParametrosDoenca:
        return replace(prior, **{k: de_livre(k, float(v)) for k, v in zip(LIVRES, x,
                                                                         strict=True)})

    def objetivo(x: np.ndarray) -> float:
        if np.any(x < lo) or np.any(x > hi):
            return 1e6
        try:
            parametros = com(x)
        except ValueError:  # alarme falso >= sensibilidade
            return 1e6
        perda, n = perda_um_passo(grafo, doenca, clima, parametros, meia_vida_pares)
        contagem["n"] = n
        if not np.isfinite(perda):
            return 1e6
        return perda * n + 0.5 * float((((x - base) / sigma) ** 2).sum())

    inicial = objetivo(base)
    # Reinício do simplex no melhor ponto (passo menor a cada rodada): o Nelder-Mead para em
    # simplex degenerado longe do ótimo.
    x, valor, iteracoes = nelder_mead(objetivo, base, max_iter=max_iter)
    for rodada in range(1, REINICIOS + 1):
        x2, valor2, it2 = nelder_mead(objetivo, x, passo=0.4 / 2**rodada, max_iter=max_iter)
        iteracoes += it2
        melhorou = valor - valor2
        if valor2 < valor:
            x, valor = x2, valor2
        if melhorou < 1e-4 * max(1.0, abs(valor)):
            break
    # Polimento só nos que ficaram dentro dos limites (o encostado no limite não tem curvatura
    # honesta); a Laplace refaz a conta de quem está dentro no ponto final.
    interior = np.flatnonzero((x - PASSO_HESSIANA > lo) & (x + PASSO_HESSIANA < hi))
    if len(interior):
        fixo = x.copy()

        def parcial(z: np.ndarray) -> float:
            completo = fixo.copy()
            completo[interior] = z
            return objetivo(completo)

        z, valor = polir(parcial, x[interior], valor)
        x = fixo.copy()
        x[interior] = z
    cov, dentro, origem_laplace = laplace(objetivo, x, lo, hi, sigma)
    objetivo(x)  # a contagem de plantas do ponto final
    n = contagem["n"]
    deslocamento = np.abs((x - base) / sigma)
    ajustado = com(x)
    # Ganho **total** (nats, razão de verossimilhança ~20): por planta, um talhão grande e quase
    # todo sadio dilui o ganho — 40.000 pés com 20 focos davam "não identificado" com os dados
    # movendo o ajuste (16/10/2026).
    identificado = bool(n >= 30 and inicial - valor >= GANHO_IDENTIFICADO)
    # O MAP com a covariância de Laplace é a posterior, movam os dados muito ou pouco:
    # "identificado" só diz se moveram. A primeira versão devolvia o prior com a covariância
    # do prior quando o ganho ficava abaixo do limiar, e o conjunto da previsão sorteava β num
    # fator 7 em volta dele — a previsão caiu de 0,111 para 0,022 numa semente cujo ganho foi
    # 0,00099 (talhão sintético, 16/10/2026).
    ajustado = replace(ajustado, origem=(
        f"ajustado às {grafo.t} passadas do talhão {grafo.field_id}" if identificado
        else f"perto do prior: as {grafo.t} passadas quase não moveram o ajuste"))
    return ModeloAjustado(
        ajustado, LIVRES, x, cov,
        {"identificado": identificado, "passadas": grafo.t, "plantas_previstas": n,
         "perda_por_planta_prior": round(inicial / max(1, n), 5),
         "perda_por_planta_ajuste": round(valor / max(1, n), 5),
         "ganho_total_nats": round(float(inicial - valor), 3),
         "iteracoes": iteracoes,
         "desvio_do_prior_em_sigmas": dict(zip(LIVRES, np.round(deslocamento, 2).tolist(),
                                               strict=True)),
         "incerteza_livre_sigma": dict(zip(LIVRES, np.round(np.sqrt(np.diag(cov)), 3)
                                           .tolist(), strict=True)),
         "intervalo_95": {k: [round(de_livre(k, float(x[i] - 1.96 * math.sqrt(cov[i, i]))), 6),
                              round(de_livre(k, float(x[i] + 1.96 * math.sqrt(cov[i, i]))), 6)]
                          for i, k in enumerate(LIVRES)},
         "no_limite": [k for k, d in zip(LIVRES, dentro, strict=True) if not d],
         "laplace": origem_laplace,
         "clima": dict(_fontes(clima))})


def _fontes(dias: Sequence[DiaClimatico]) -> dict[str, int]:
    saida: dict[str, int] = {}
    for d in dias:
        saida[d.fonte] = saida.get(d.fonte, 0) + 1
    return saida
