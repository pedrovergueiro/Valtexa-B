"""Onde vale mais olhar: prioridade de inspeção por incerteza e risco, por fonte de observação.

O gêmeo (:mod:`app.gemeo.previsao`) diz, por planta, a probabilidade de estar infectada no
dia da visita. Olhar uma planta vale o quanto a observação **muda a decisão** sobre ela — o
valor da informação (VOI) da decisão tratar / não tratar:

* ``D`` = a planta infectada (latente ou esporulando), ``p = P(D)``; a fonte só vê o que
  esporula: ``P(sinal | D) = a + (s - a) . P(I)/P(D)``, ``P(sinal | não D) = a``, com o
  canal ``(s, a)`` da fonte (:class:`~app.gemeo.epidemia.Canal` — o Rover ajustado ao talhão,
  o drone pela tabela);
* com os números do produtor (custo de tratar ``c``, perda por planta ``L``, eficácia ``e``)
  e a consequência ``(1 + R_i)`` — as vizinhas que ela infectaria até o horizonte, pelo
  núcleo de contágio ajustado: ``R_i = β . f . h . Σ_j w_ij S_j``::

      VOI_i = L_i e (E_y[max(0, P(D|y) - t_i)] - max(0, p - t_i)),   t_i = c / (L_i e)

  — zero quando a decisão já está tomada (planta quase certamente sadia ou doente), máximo
  perto do limiar; é a forma certa de juntar risco e incerteza num número só;
* sem os números do produtor, a **informação mútua** ``I(D; sinal)`` em bits, vezes a
  consequência: quanto a observação ensina sobre a planta, pesado pelo que ela pode causar.

**Tarefas.** O Rover anda na entrelinha e vê a rua: a tarefa dele é o **trecho de rua**
(``segmento_m``). O drone cobre área: o **bloco** (``bloco_ruas`` x ``bloco_m``). O valor da
tarefa é a soma das plantas; o custo (tempo, energia, deslocamento) é de quem aloca — o
SaaS, que conhece cada dispositivo (``valtexa/services/enxame.py``).

**Não é ordem de serviço.** É a lista que a alocação usa; a missão continua nascendo na
plataforma, autorizada pela camada de segurança e assinada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.gemeo.epidemia import Canal, ModeloAjustado, Nucleo, nucleo_de_contagio
from app.gemeo.grafo import GrafoLavoura
from app.gemeo.previsao import Previsao

EPS = 1e-9
SEGMENTO_M = 20.0
BLOCO_RUAS = 6
BLOCO_M = 40.0
R_MAXIMO = 10.0
"""Teto da consequência por contágio (a força do campo médio cresce sem fim no papel)."""


@dataclass(frozen=True)
class Economia:
    """Os números do produtor para a decisão por planta (os da prescrição)."""

    custo_tratar: float
    perda_planta: float
    eficacia: float

    def __post_init__(self) -> None:
        if self.custo_tratar <= 0 or self.perda_planta <= 0 or not 0 < self.eficacia <= 1:
            raise ValueError("custo e perda positivos, eficácia em (0, 1].")


def _entropia(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def verossimilhancas(p: np.ndarray, q: np.ndarray, canal: Canal) -> tuple[np.ndarray, np.ndarray]:
    """``P(sinal | D)`` e ``P(sinal | não D)`` por planta (``q`` = P(esporulando) <= ``p``)."""
    fracao_visivel = np.where(p > EPS, np.clip(q / np.maximum(p, EPS), 0.0, 1.0), 0.0)
    com = canal.alarme_falso + (canal.sensibilidade - canal.alarme_falso) * fracao_visivel
    return com, np.full_like(p, canal.alarme_falso)


def informacao_mutua(p: np.ndarray, q: np.ndarray, canal: Canal) -> np.ndarray:
    """``I(D; sinal)`` em bits, por planta."""
    l1, l0 = verossimilhancas(p, q, canal)
    py = p * l1 + (1 - p) * l0
    return np.maximum(_entropia(py) - (p * _entropia(l1) + (1 - p) * _entropia(l0)), 0.0)


def valor_da_informacao(p: np.ndarray, q: np.ndarray, canal: Canal, economia: Economia,
                        consequencia: np.ndarray | float = 1.0) -> np.ndarray:
    """VOI da decisão tratar / não tratar, em dinheiro, por planta."""
    escala = economia.perda_planta * economia.eficacia * np.asarray(consequencia, dtype=float)
    limiar = economia.custo_tratar / np.maximum(escala, EPS)
    l1, l0 = verossimilhancas(p, q, canal)
    py1 = p * l1 + (1 - p) * l0
    pos1 = p * l1 / np.maximum(py1, EPS)
    pos0 = p * (1 - l1) / np.maximum(1 - py1, EPS)
    depois = py1 * np.maximum(pos1 - limiar, 0) + (1 - py1) * np.maximum(pos0 - limiar, 0)
    antes = np.maximum(p - limiar, 0)
    return np.maximum(escala * (depois - antes), 0.0)


def consequencia_por_contagio(nucleo: Nucleo, suscetivel: np.ndarray, beta: float,
                              favorabilidade_media: float, horizonte_dias: float) -> np.ndarray:
    """``1 + R_i``: a planta e as vizinhas que ela infectaria até o horizonte se ficasse doente
    sem tratar (``R_i = β . f . h . Σ_j w_ij S_j``, núcleo simétrico), com teto em
    :data:`R_MAXIMO`."""
    r = beta * favorabilidade_media * horizonte_dias * nucleo(suscetivel)
    return 1.0 + np.clip(r, 0.0, R_MAXIMO)


@dataclass
class Tarefa:
    tipo: str
    """``inspecao`` (trecho de rua, Rover) ou ``sobrevoo`` (bloco, drone)."""
    ruas: list[str]
    x_inicio_m: float
    x_fim_m: float
    plantas: np.ndarray
    valor: float
    risco_medio: float
    risco_maximo: float
    incerteza_bits: float
    dias_sem_ver: int | None
    """Dias desde a última passada que viu alguma planta da tarefa (``None``: nunca vista)."""
    espacamento_entre_ruas_m: float = 3.5
    poligono: list[list[float]] | None = None
    """``[[lon, lat], ...]`` (anti-horário, fechado) quando as plantas têm GPS."""
    motivo: str = ""

    @property
    def comprimento_m(self) -> float:
        """O que o Rover anda: o trecho em cada rua (pelo menos um passo de planta)."""
        return max(self.x_fim_m - self.x_inicio_m, 0.7) * max(1, len(self.ruas))

    @property
    def area_m2(self) -> float:
        """O que o drone cobre: o trecho pela largura das ruas."""
        return (max(self.x_fim_m - self.x_inicio_m, 0.7) * max(1, len(self.ruas))
                * self.espacamento_entre_ruas_m)

    def to_dict(self, grafo: GrafoLavoura, com_plantas: bool = False) -> dict[str, Any]:
        d = {"tipo": self.tipo, "ruas": self.ruas, "x_inicio_m": round(self.x_inicio_m, 2),
             "x_fim_m": round(self.x_fim_m, 2), "comprimento_m": round(self.comprimento_m, 1),
             "area_m2": round(self.area_m2, 1),
             "plantas": int(len(self.plantas)), "valor": round(self.valor, 4),
             "risco_medio": round(self.risco_medio, 4),
             "risco_maximo": round(self.risco_maximo, 4),
             "incerteza_bits": round(self.incerteza_bits, 4),
             "dias_sem_ver": self.dias_sem_ver, "poligono": self.poligono,
             "motivo": self.motivo}
        if com_plantas:
            d["plant_ids"] = [grafo.plant_ids[i] for i in self.plantas]
        return d


@dataclass
class Prioridades:
    field_id: str
    doenca: str
    canal: str
    criterio: str
    """``valor_da_informacao`` (com a economia, em dinheiro) ou ``informacao`` (bits)."""
    data_visita: str
    tarefas: list[Tarefa]
    valor_por_planta: np.ndarray
    avisos: list[str] = field(default_factory=list)

    def to_dict(self, grafo: GrafoLavoura, max_tarefas: int | None = None,
                com_plantas: bool = False) -> dict[str, Any]:
        tarefas = self.tarefas if max_tarefas is None else self.tarefas[:max_tarefas]
        total = float(sum(t.valor for t in self.tarefas))
        return {"field_id": self.field_id, "doenca": self.doenca, "canal": self.canal,
                "criterio": self.criterio, "data_visita": self.data_visita,
                "tarefas_total": len(self.tarefas), "valor_total": round(total, 4),
                "tarefas": [t.to_dict(grafo, com_plantas) for t in tarefas],
                "avisos": self.avisos}


def _poligono(grafo: GrafoLavoura, idx: np.ndarray, folga_m: float) -> list[list[float]] | None:
    """O retângulo envolvente das plantas, com folga, em lon/lat (sem GPS: ``None``)."""
    lat, lon = grafo.lat[idx], grafo.lon[idx]
    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        return None
    lat, lon = lat[ok], lon[ok]
    dlat = folga_m / 111_320.0
    dlon = folga_m / (111_320.0 * max(math.cos(math.radians(float(lat.mean()))), 1e-6))
    a, b = float(lon.min() - dlon), float(lon.max() + dlon)
    c, d = float(lat.min() - dlat), float(lat.max() + dlat)
    return [[a, c], [b, c], [b, d], [a, d], [a, c]]


def _ultimo_dia_visto(grafo: GrafoLavoura) -> np.ndarray:
    return np.where((grafo.dia_obs >= 0).any(axis=0), grafo.dia_obs.max(axis=0), -1)


def agrupar(grafo: GrafoLavoura, tipo: str, segmento_m: float = SEGMENTO_M,
            bloco_ruas: int = BLOCO_RUAS, bloco_m: float = BLOCO_M) -> list[np.ndarray]:
    """As plantas de cada tarefa: trecho de rua (inspeção) ou bloco (sobrevoo)."""
    x0 = float(grafo.x_m.min()) if grafo.n else 0.0
    if tipo == "inspecao":
        chave = grafo.rua * 100_000 + np.floor((grafo.x_m - x0) / segmento_m).astype(np.int64)
    elif tipo == "sobrevoo":
        chave = ((grafo.rua // bloco_ruas) * 100_000
                 + np.floor((grafo.x_m - x0) / bloco_m).astype(np.int64))
    else:
        raise ValueError(f"tipo de tarefa desconhecido: {tipo!r}")
    ordem = np.argsort(chave, kind="stable")
    cortes = np.flatnonzero(np.diff(chave[ordem])) + 1
    return [g for g in np.split(ordem, cortes) if len(g)]


def priorizar(grafo: GrafoLavoura, previsao: Previsao, modelo: ModeloAjustado, canal: Canal,
              *, nome_canal: str, economia: Economia | None = None,
              segmento_m: float = SEGMENTO_M, bloco_ruas: int = BLOCO_RUAS,
              bloco_m: float = BLOCO_M) -> Prioridades:
    """As tarefas do talhão para uma fonte, da mais valiosa para a menos.

    ``previsao`` deve ter o horizonte no dia da visita: o valor é o de olhar **naquele dia**.
    """
    tipo = "sobrevoo" if nome_canal == "drone" else "inspecao"
    p = np.clip(previsao.risco, 0.0, 1.0)
    q = np.minimum(np.clip(previsao.esporulando_no_horizonte, 0.0, 1.0), p)
    par = modelo.parametros
    horizonte = max(1, (previsao.horizonte - previsao.hoje).days)
    fav = float(np.mean([d["favorabilidade"] for d in previsao.diario])) if previsao.diario else 0.0
    consequencia = consequencia_por_contagio(nucleo_de_contagio(grafo, par), 1.0 - p, par.beta,
                                             fav, horizonte)
    avisos = list(previsao.avisos)
    if canal.origem != "ajustado ao talhão":
        avisos.append(f"Canal de observação {nome_canal!r}: {canal.origem}.")
    if economia is None:
        valor = informacao_mutua(p, q, canal) * consequencia
        criterio = "informacao"
        avisos.append("Sem os números do produtor (custo, perda, eficácia): prioridade pela "
                      "informação (bits) pesada pelo contágio, não pelo dinheiro em jogo.")
    else:
        valor = valor_da_informacao(p, q, canal, economia, consequencia)
        criterio = "valor_da_informacao"
    incerteza = _entropia(p)
    ultimo = _ultimo_dia_visto(grafo)
    dia_visita = (previsao.horizonte - grafo.inicio).days
    tarefas = []
    for idx in agrupar(grafo, tipo, segmento_m, bloco_ruas, bloco_m):
        vistos = ultimo[idx]
        dias = None if (vistos < 0).all() else int(dia_visita - vistos[vistos >= 0].max())
        ruas = sorted({int(grafo.rua[i]) for i in idx})
        t = Tarefa(tipo=tipo, ruas=[grafo.ruas[r] for r in ruas],
                   x_inicio_m=float(grafo.x_m[idx].min()), x_fim_m=float(grafo.x_m[idx].max()),
                   plantas=idx, valor=float(valor[idx].sum()),
                   risco_medio=float(p[idx].mean()), risco_maximo=float(p[idx].max()),
                   incerteza_bits=float(incerteza[idx].mean()), dias_sem_ver=dias,
                   espacamento_entre_ruas_m=grafo.espacamento_entre_ruas_m,
                   poligono=_poligono(grafo, idx, grafo.espacamento_entre_ruas_m / 2))
        t.motivo = _motivo(t, criterio)
        tarefas.append(t)
    tarefas.sort(key=lambda t: -t.valor)
    return Prioridades(grafo.field_id, previsao.doenca, nome_canal, criterio,
                       previsao.horizonte.isoformat(), tarefas, valor, avisos)


def _motivo(t: Tarefa, criterio: str) -> str:
    partes = [f"risco médio {t.risco_medio:.0%} (máx. {t.risco_maximo:.0%})",
              f"incerteza {t.incerteza_bits:.2f} bit"]
    partes.append("nunca vista" if t.dias_sem_ver is None else f"vista há {t.dias_sem_ver} dias")
    unidade = "em jogo" if criterio == "valor_da_informacao" else "bits ponderados"
    return f"{'; '.join(partes)}; valor {t.valor:.2f} {unidade}"
