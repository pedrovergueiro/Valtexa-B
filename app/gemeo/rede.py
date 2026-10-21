"""A rede espaço-temporal do gêmeo: um **resíduo** aprendido sobre a física, nunca no lugar dela.

O modelo epidemiológico (:mod:`app.gemeo.epidemia`) sabe o que a epidemiologia sabe — tempo,
contágio, latência — e não sabe o que é do talhão: a baixada que fica molhada, a bordadura do
vento, a variedade mais suscetível de duas ruas, a sombra. A rede aprende isso das passadas do
próprio talhão, como uma correção no logit do sinal que a física prevê para a passada seguinte:

    logit P(sinal_i) = logit q_i (física)  +  g(x_i, Σ_j w̃_ij x_j)

* ``g`` é uma camada de mensagem pelo grafo da lavoura (a média das vizinhas ponderada pelo
  núcleo de contágio ajustado, ``w̃`` normalizado) seguida de uma camada oculta ``tanh`` — um
  grafo-convolucional de uma camada. Pequena de propósito (cerca de 150 pesos): um talhão tem
  algumas passadas, não milhões de exemplos.
* **Entradas** (:data:`ENTRADAS`): o logit da física, o estado assimilado (``I`` e ``E``), a
  posterior da passada, a média dela nas vizinhas, os recortes que a planta teve e a
  **memória do resíduo** — onde a física errou para baixo nas passadas anteriores, na planta e
  na vizinhança. É ela que carrega o que é do talhão; sem as passadas anteriores não há o que
  aprender.
* **Começa em zero** (a saída é a física pura) e treina com a mesma verossimilhança da
  evidência do ajuste (:func:`~app.gemeo.epidemia.log_verossimilhanca`), com L2.
* **Portão temporal, com seleção**: três candidatas disputam o último par de passadas, que
  nenhuma viu — a física pura, a física **recalibrada** (Platt: ``a . logit q + b``, dois
  números ajustados no par mais recente do treino) e a física com a **rede** (treinada em
  todos os pares do treino). Fica a melhor, se ganhar da física por :data:`GANHO_MINIMO` por
  planta; senão, a física sozinha, e o diagnóstico diz por quê. A escolhida é refeita com o
  que há de mais novo (a recalibração no último par; a rede em todos).

  A recalibração existe porque a medição mostrou que é ela que a linha de base tem: a
  heurística de contágio do produto, recalibrada no par anterior, ganhava do gêmeo no talhão
  com baixada (a estrutura que a física não conhece) e a rede, com ~150 pesos e poucas
  passadas, ganhava num par e perdia no seguinte (talhão sintético, 16/10/2026). Dois números
  não oscilam assim.

Só NumPy: a API não carrega framework de rede para 150 pesos.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.gemeo.clima import DiaClimatico
from app.gemeo.epidemia import (
    Estado,
    Nucleo,
    ParametrosDoenca,
    canal_de,
    filtrar,
    log_verossimilhanca,
    nucleo_de_contagio,
)
from app.gemeo.grafo import GrafoLavoura

ENTRADAS = ("logit_fisica", "logit_esporulando", "logit_latente", "logit_posterior",
            "logit_posterior_vizinhas", "residuo_planta", "residuo_vizinhas", "log_recortes")
OCULTA = 8
GANHO_MINIMO = 0.002
"""Ganho mínimo de log-verossimilhança por planta, no par retido, para a rede ser adotada."""
L2 = 1e-3
PASSOS = 400
TAXA = 0.02
EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def media_vizinhas(nucleo: Nucleo, valor: np.ndarray) -> np.ndarray:
    """``Σ_j w_ij v_j / Σ_j w_ij`` (0 na planta sem vizinha)."""
    soma = nucleo(valor)
    norma = nucleo(np.ones(len(valor)))
    return np.where(norma > 1e-12, soma / np.maximum(norma, 1e-12), 0.0)


@dataclass
class Par:
    """Um par de passadas: o que o gêmeo sabia depois da passada ``k`` e o que a ``k+1`` viu."""

    k: int
    entradas: np.ndarray
    """``(N, len(ENTRADAS))``."""
    fisica: np.ndarray
    """P(sinal) previsto pela física para a passada ``k+1``."""
    razao: np.ndarray
    """Razão de verossimilhança da passada ``k+1``."""
    vista: np.ndarray


@dataclass
class Historico:
    """O filtro rodado sobre as passadas, com o que a rede precisa (calculado uma vez)."""

    pares: list[Par]
    final: np.ndarray
    """Entradas depois da última passada, menos a física (que depende do horizonte):
    ``(N, len(ENTRADAS))`` com a coluna 0 em zero."""
    residuo: np.ndarray
    """A memória do resíduo depois da última passada."""


def _entradas(nucleo: Nucleo, fisica: np.ndarray, estado: Estado,
              posterior: np.ndarray, prior: float, residuo: np.ndarray,
              recortes: np.ndarray) -> np.ndarray:
    post = np.where(np.isfinite(posterior), posterior, prior)
    return np.column_stack([
        _logit(fisica), _logit(estado.I), _logit(estado.E), _logit(post),
        _logit(media_vizinhas(nucleo, post)), residuo,
        media_vizinhas(nucleo, residuo), np.log1p(recortes)])


def historico(grafo: GrafoLavoura, doenca: str, clima: Sequence[DiaClimatico],
              p: ParametrosDoenca) -> Historico:
    """Roda o filtro (:func:`~app.gemeo.epidemia.filtrar`) e monta um :class:`Par` por par de
    passadas seguidas."""
    nucleo = nucleo_de_contagio(grafo, p)
    razao = grafo.razao(doenca)
    post = grafo.posterior[doenca]
    prior = grafo.prior_fusao.get(doenca, 0.5)
    recortes = grafo.n_recortes if grafo.n_recortes is not None else np.zeros((grafo.t, grafo.n))
    residuo_soma = np.zeros(grafo.n)
    residuo_n = np.zeros(grafo.n)
    pares: list[Par] = []
    final = None
    for s in filtrar(grafo, razao, clima, p, nucleo):
        residuo = np.where(residuo_n > 0, residuo_soma / np.maximum(residuo_n, 1), 0.0)
        if s.previsto_esporulando is None or s.vista_proxima is None:
            if s.k == grafo.t - 1:
                final = _entradas(nucleo, np.full(grafo.n, 0.5), s.assimilado, post[s.k],
                                  prior, residuo, recortes[s.k])
                final[:, 0] = 0.0
            continue
        vista = s.vista_proxima
        c = canal_de(p, grafo.fonte(s.k + 1))
        fisica = c.alarme_falso + (c.sensibilidade - c.alarme_falso) * s.previsto_esporulando
        pares.append(Par(s.k, _entradas(nucleo, fisica, s.assimilado, post[s.k], prior,
                                        residuo, recortes[s.k]), fisica, razao[s.k + 1], vista))
        # A memória do resíduo: o sinal esperado dado o que a passada k+1 viu, menos o previsto.
        lam = razao[s.k + 1]
        esperado = fisica * lam / (fisica * lam + 1 - fisica)
        residuo_soma += np.where(vista, esperado - fisica, 0.0)
        residuo_n += vista
    assert final is not None
    residuo = np.where(residuo_n > 0, residuo_soma / np.maximum(residuo_n, 1), 0.0)
    return Historico(pares, final, residuo)


@dataclass
class RedeResidual:
    """Os pesos e a padronização das entradas — ou, em ``tipo="recalibracao"``, só os dois
    números de Platt (``platt``), com o resto vazio."""

    media: np.ndarray
    desvio: np.ndarray
    w: np.ndarray
    b: np.ndarray
    v: np.ndarray
    c: float
    pesos_grafo: tuple[float, float]
    """Os alcances (na rua, entre ruas) com que as médias das vizinhas foram feitas."""
    diagnostico: dict[str, Any] = field(default_factory=dict)
    tipo: str = "rede"
    platt: tuple[float, float] = (1.0, 0.0)

    def _matriz(self, grafo: GrafoLavoura, entradas: np.ndarray) -> np.ndarray:
        z = (entradas - self.media) / self.desvio
        nucleo = Nucleo(grafo.vizinhanca, *self.pesos_grafo)
        vizinhas = np.column_stack([media_vizinhas(nucleo, z[:, j])
                                    for j in range(z.shape[1])])
        return np.hstack([z, vizinhas])

    def correcao(self, grafo: GrafoLavoura, entradas: np.ndarray) -> np.ndarray:
        """O termo que soma no logit da física."""
        if self.tipo == "recalibracao":
            return (self.platt[0] - 1.0) * entradas[:, 0] + self.platt[1]
        h = np.tanh(self._matriz(grafo, entradas) @ self.w + self.b)
        return h @ self.v + self.c

    def corrigir(self, grafo: GrafoLavoura, entradas: np.ndarray, fisica: np.ndarray
                 ) -> np.ndarray:
        e = entradas.copy()
        e[:, 0] = _logit(fisica)
        return _sigmoid(_logit(fisica) + self.correcao(grafo, e))

    def to_dict(self) -> dict[str, Any]:
        return {"media": self.media.tolist(), "desvio": self.desvio.tolist(),
                "w": self.w.tolist(), "b": self.b.tolist(), "v": self.v.tolist(),
                "c": self.c, "pesos_grafo": list(self.pesos_grafo),
                "diagnostico": self.diagnostico, "tipo": self.tipo,
                "platt": list(self.platt)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RedeResidual:
        return cls(np.array(d["media"]), np.array(d["desvio"]), np.array(d["w"]),
                   np.array(d["b"]), np.array(d["v"]), float(d["c"]),
                   (float(d["pesos_grafo"][0]), float(d["pesos_grafo"][1])),
                   dict(d.get("diagnostico") or {}), str(d.get("tipo", "rede")),
                   tuple(float(x) for x in d.get("platt", (1.0, 0.0))))


def _treinar(grafo: GrafoLavoura, pares: Sequence[Par], alcances: tuple[float, float],
             semente: int = 0) -> RedeResidual:
    todas = np.vstack([q.entradas for q in pares])
    media = todas.mean(axis=0)
    desvio = np.where(todas.std(axis=0) > 1e-9, todas.std(axis=0), 1.0)
    rng = np.random.default_rng(semente)
    f = len(ENTRADAS)
    rede = RedeResidual(media, desvio, rng.normal(0, 1 / math.sqrt(2 * f), (2 * f, OCULTA)),
                        np.zeros(OCULTA), np.zeros(OCULTA), 0.0, alcances)
    xs = [rede._matriz(grafo, q.entradas)[q.vista] for q in pares]
    bases = [_logit(q.fisica[q.vista]) for q in pares]
    lams = [q.razao[q.vista] for q in pares]
    x = np.vstack(xs)
    base = np.concatenate(bases)
    lam = np.concatenate(lams)
    n = len(base)
    params = [rede.w, rede.b, rede.v, np.array([rede.c])]
    m1 = [np.zeros_like(t) for t in params]
    m2 = [np.zeros_like(t) for t in params]
    for it in range(1, PASSOS + 1):
        w, b, v, c = params
        h = np.tanh(x @ w + b)
        q = _sigmoid(base + h @ v + c[0])
        g_out = -(lam - 1) * q * (1 - q) / (1 + q * (lam - 1)) / n
        dv = h.T @ g_out + L2 * v
        dc = np.array([g_out.sum()])
        dz = np.outer(g_out, v) * (1 - h**2)
        dw = x.T @ dz + L2 * w
        db = dz.sum(axis=0)
        for i, g in enumerate((dw, db, dv, dc)):
            m1[i] = 0.9 * m1[i] + 0.1 * g
            m2[i] = 0.999 * m2[i] + 0.001 * g**2
            passo_ = TAXA * (m1[i] / (1 - 0.9**it)) / (np.sqrt(m2[i] / (1 - 0.999**it)) + 1e-8)
            params[i] = params[i] - passo_
    rede.w, rede.b, rede.v, rede.c = params[0], params[1], params[2], float(params[3][0])
    return rede


def _recalibrar(par: Par, alcances: tuple[float, float], l2: float = 1e-3,
                passos: int = 300, taxa: float = 0.05) -> RedeResidual:
    """Platt no par: ``logit q' = a . logit q + b``, pela verossimilhança da evidência, com L2
    em volta da identidade (a = 1, b = 0)."""
    z = _logit(par.fisica[par.vista])
    lam = par.razao[par.vista]
    n = len(z)
    theta = np.array([1.0, 0.0])
    m1 = np.zeros(2)
    m2 = np.zeros(2)
    for it in range(1, passos + 1):
        q = _sigmoid(theta[0] * z + theta[1])
        g_out = -(lam - 1) * q * (1 - q) / (1 + q * (lam - 1)) / max(n, 1)
        grad = np.array([float(g_out @ z), float(g_out.sum())]) + l2 * (theta - [1.0, 0.0])
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad**2
        theta = theta - taxa * (m1 / (1 - 0.9**it)) / (np.sqrt(m2 / (1 - 0.999**it)) + 1e-8)
    f = len(ENTRADAS)
    return RedeResidual(np.zeros(f), np.ones(f), np.zeros((2 * f, 0)), np.zeros(0),
                        np.zeros(0), 0.0, alcances, tipo="recalibracao",
                        platt=(float(theta[0]), float(theta[1])))


def _logv(rede: RedeResidual | None, grafo: GrafoLavoura, par: Par) -> float:
    fisica = par.fisica
    q = fisica if rede is None else rede.corrigir(grafo, par.entradas, fisica)
    return float(log_verossimilhanca(q[par.vista], par.razao[par.vista]).mean())


def aplicavel(rede: RedeResidual, horizonte_dias: int) -> bool:
    """A rede aprendeu a corrigir a previsão de **uma passada à frente**: só vale para um
    horizonte entre metade e uma vez e meia o intervalo em que treinou."""
    intervalo = float(rede.diagnostico.get("intervalo_mediano_dias") or 0)
    return intervalo > 0 and 0.5 * intervalo <= horizonte_dias <= 1.5 * intervalo


def ajustar_rede(grafo: GrafoLavoura, hist: Historico, p: ParametrosDoenca,
                 semente: int = 0) -> tuple[RedeResidual | None, dict[str, Any]]:
    """O portão temporal com seleção: a correção adotada (ou ``None``: física pura) e o porquê.
    """
    alcances = (p.alcance_na_rua_m, p.alcance_entre_ruas_m)
    if len(hist.pares) < 2:
        return None, {"adotada": False, "escolha": "fisica",
                      "motivo": "menos de três passadas: sem par para treinar e outro para "
                      "medir — o gêmeo segue só com a física"}
    treino, retido = hist.pares[:-1], hist.pares[-1]
    candidatas = {"recalibracao": _recalibrar(treino[-1], alcances),
                  "rede": _treinar(grafo, treino, alcances, semente)}
    fisica = _logv(None, grafo, retido)
    logv = {nome: _logv(c, grafo, retido) for nome, c in candidatas.items()}
    intervalos = [grafo.dia_da_passada(q.k + 1) - grafo.dia_da_passada(q.k) for q in hist.pares]
    melhor = max(logv, key=logv.get)
    ganho = logv[melhor] - fisica
    diag = {"pares_treino": len(treino), "par_retido": f"passada {retido.k} -> {retido.k + 1}",
            "intervalo_mediano_dias": float(np.median(intervalos)),
            "logv_fisica_retido": round(fisica, 5),
            "logv_recalibracao_retido": round(logv["recalibracao"], 5),
            "logv_rede_retido": round(logv["rede"], 5),
            "ganho_por_planta": round(ganho, 5), "ganho_minimo": GANHO_MINIMO}
    if ganho < GANHO_MINIMO:
        return None, {**diag, "adotada": False, "escolha": "fisica",
                      "motivo": "nenhuma correção ganhou da física no par que não viu"}
    if melhor == "rede":
        final = _treinar(grafo, hist.pares, alcances, semente)
    else:
        final = _recalibrar(hist.pares[-1], alcances)
    final.diagnostico = {**diag, "adotada": True, "escolha": melhor}
    if final.tipo == "recalibracao":
        final.diagnostico["platt"] = [round(x, 4) for x in final.platt]
    return final, final.diagnostico
