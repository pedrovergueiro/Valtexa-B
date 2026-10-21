"""A previsão do gêmeo: onde a doença vai estar daqui a algumas semanas, com a incerteza dela.

1. **Assimila** todas as passadas fechadas (o filtro de :func:`assimilar`, o mesmo do ajuste):
   o estado de hoje inclui o que o Rover **não** vê — o latente ``E``, a planta que pegou e
   ainda não esporula (20 a 60 dias na ferrugem). É dele que sai a reboleira do mês que vem.
2. **Projeta** dia a dia até o horizonte, com o tempo da série (estação, histórico, previsão
   e, depois dela, climatologia).
3. **Conjunto**: cada membro sorteia os parâmetros da aproximação de Laplace do ajuste
   (:meth:`~app.gemeo.epidemia.ModeloAjustado.amostras`) e perturba o tempo conforme a fonte
   (:func:`perturbar_clima`): dia medido não muda; dia de previsão ganha erro que cresce com a
   antecedência; dia de climatologia, o erro largo de não se saber o tempo. As faixas (p10 e
   p90) são a dispersão do conjunto — incerteza de parâmetro e de tempo, **não** de estrutura
   do modelo (a do campo médio contra a epidemia de verdade está medida em
   ``training/mede_gemeo.py``).

O que sai por planta: P(esporulando) agora, P(infectada — latente ou esporulando) no horizonte
(o **risco**) e P(sinal) do Rover no horizonte. Por talhão: a favorabilidade diária com as
**janelas de infecção** (dias em que o tempo deixa o fungo germinar), a incidência esperada e
os **focos** de agora e do horizonte (grupos ligados de plantas marcadas — a reboleira).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any

import numpy as np

from app.gemeo.clima import DiaClimatico, SerieClimatica
from app.gemeo.epidemia import (
    Estado,
    ModeloAjustado,
    Nucleo,
    ParametrosDoenca,
    Tratamento,
    aplicar_tratamentos,
    dia_da_ultima_assimilacao,
    favorabilidade,
    filtrar,
    nucleo_de_contagio,
    passo,
    probabilidade_de_sinal,
)
from app.gemeo.grafo import GrafoLavoura, componentes
from app.gemeo.rede import RedeResidual, aplicavel, historico

LIMIAR_JANELA = 0.5
"""Favorabilidade a partir da qual o dia é janela de infecção (metade da máxima: temperatura e
molhamento os dois perto do ótimo)."""
LIMIAR_FOCO_AGORA = 0.5
"""P(esporulando) para a planta entrar num foco de agora."""
LIMIAR_RISCO = 0.3
"""P(infectada no horizonte) para a planta entrar num foco previsto."""

# Erro do tempo por fonte: (sigma da temperatura em °C, sigma do molhamento em h, sigma do log da
# chuva). Na previsão, cresce com a antecedência (por dia), até o da climatologia.
ERRO_PREVISAO = (0.5, 1.0, 0.3)
ERRO_PREVISAO_POR_DIA = (0.15, 0.25, 0.05)
ERRO_CLIMATOLOGIA = (2.0, 3.0, 0.8)


def _sem_erro(fonte: str) -> bool:
    return fonte not in ("open-meteo:previsao", "climatologia")


def perturbar_clima(dias: Sequence[DiaClimatico], hoje: date,
                    rng: np.random.Generator) -> list[DiaClimatico]:
    """Um membro do conjunto: cada dia futuro com o erro da fonte dele.

    O erro é sorteado **por dia** mas com o mesmo viés no membro inteiro (metade da variância
    em cada): previsão de tempo erra em bloco — a frente fria que atrasa atrasa todos os dias.
    """
    vies = rng.normal(size=3) / math.sqrt(2)
    saida = []
    for d in dias:
        if _sem_erro(d.fonte):
            saida.append(d)
            continue
        if d.fonte == "climatologia":
            st, sw, sc = ERRO_CLIMATOLOGIA
        else:
            ante = max(0, (d.data - hoje).days)
            st, sw, sc = (min(b + k * ante, c) for b, k, c in
                          zip(ERRO_PREVISAO, ERRO_PREVISAO_POR_DIA, ERRO_CLIMATOLOGIA,
                              strict=True))
        z = vies + rng.normal(size=3) / math.sqrt(2)
        dt = st * z[0]
        saida.append(replace(
            d, t_media=d.t_media + dt, t_min=d.t_min + dt, t_max=d.t_max + dt,
            molhamento_h=float(np.clip(d.molhamento_h + sw * z[1], 0.0, 24.0)),
            chuva_mm=float(d.chuva_mm * math.exp(sc * z[2] - sc**2 / 2))))
    return saida


def assimilar(grafo: GrafoLavoura, doenca: str, clima: Sequence[DiaClimatico],
              p: ParametrosDoenca, nucleo: Nucleo | None = None) -> tuple[Estado, int]:
    """O estado depois da última passada e o dia (desde ``grafo.inicio``) a que ele se refere
    (o filtro de :func:`~app.gemeo.epidemia.filtrar`)."""
    ultimo = None
    for s in filtrar(grafo, grafo.razao(doenca), clima, p, nucleo):
        ultimo = s
    if ultimo is None:
        raise ValueError("Nenhuma passada com planta vista: nada a assimilar.")
    return ultimo.assimilado, ultimo.dia


@dataclass
class Foco:
    plantas: np.ndarray
    ruas: list[str]
    x_inicio_m: float
    x_fim_m: float
    lat: float | None
    lon: float | None
    intensidade: float
    """Média da probabilidade que marcou as plantas (P(esporulando) agora, risco no horizonte)."""
    novo: bool = False
    """No horizonte: nenhuma planta dele está num foco de agora (reboleira que vai surgir)."""

    def to_dict(self, grafo: GrafoLavoura) -> dict[str, Any]:
        return {"plantas": int(len(self.plantas)), "ruas": self.ruas,
                "x_inicio_m": round(self.x_inicio_m, 2), "x_fim_m": round(self.x_fim_m, 2),
                "lat": self.lat, "lon": self.lon, "intensidade": round(self.intensidade, 4),
                "novo": self.novo,
                "plant_ids": [grafo.plant_ids[i] for i in self.plantas]}


def focos(grafo: GrafoLavoura, valor: np.ndarray, limiar: float,
          distancia_max_m: float | None = None) -> list[Foco]:
    """Grupos ligados de plantas com ``valor >= limiar``.

    Ligadas: até ``distancia_max_m`` uma da outra — por padrão 1,1 vez o espaçamento entre
    ruas, o que junta a rua vizinha e até cinco pés seguidos na mesma rua.
    """
    distancia_max_m = distancia_max_m or 1.1 * grafo.espacamento_entre_ruas_m
    saida = []
    for grupo in componentes(grafo, valor >= limiar, distancia_max_m):
        ruas = sorted({int(grafo.rua[i]) for i in grupo})
        lat = grafo.lat[grupo]
        lon = grafo.lon[grupo]
        geo = np.isfinite(lat) & np.isfinite(lon)
        saida.append(Foco(grupo, [grafo.ruas[r] for r in ruas],
                          float(grafo.x_m[grupo].min()), float(grafo.x_m[grupo].max()),
                          float(lat[geo].mean()) if geo.any() else None,
                          float(lon[geo].mean()) if geo.any() else None,
                          float(valor[grupo].mean())))
    return saida


def janelas_de_infeccao(datas: Sequence[date], fav: np.ndarray,
                        limiar: float = LIMIAR_JANELA) -> list[dict[str, Any]]:
    """Trechos de dias seguidos com favorabilidade (média do conjunto) acima do limiar."""
    saida: list[dict[str, Any]] = []
    inicio = None
    for i, f in enumerate([*list(fav), -1.0]):
        if f >= limiar and inicio is None:
            inicio = i
        elif f < limiar and inicio is not None:
            trecho = fav[inicio:i]
            saida.append({"inicio": datas[inicio].isoformat(), "fim": datas[i - 1].isoformat(),
                          "dias": i - inicio, "favorabilidade_max": round(float(trecho.max()), 3)})
            inicio = None
    return saida


@dataclass
class Previsao:
    field_id: str
    doenca: str
    data_base: date
    """O dia da última passada assimilada."""
    hoje: date
    horizonte: date
    membros: int
    parametros: ParametrosDoenca
    identificado: bool
    esporulando_agora: np.ndarray
    latente_agora: np.ndarray
    risco: np.ndarray
    risco_p10: np.ndarray
    risco_p90: np.ndarray
    sinal_no_horizonte: np.ndarray
    esporulando_no_horizonte: np.ndarray
    """P(esporulando) no horizonte (média do conjunto): o que uma fonte pode ver lá."""
    diario: list[dict[str, Any]]
    janelas: list[dict[str, Any]]
    focos_agora: list[Foco]
    focos_horizonte: list[Foco]
    avisos: list[str] = field(default_factory=list)
    sinal_corrigido: np.ndarray | None = None
    """P(sinal) no horizonte com a correção da rede residual (:mod:`app.gemeo.rede`), quando
    ela foi adotada e o horizonte é o de uma passada."""
    rede: dict[str, Any] | None = None

    def resumo(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id, "doenca": self.doenca,
            "data_base": self.data_base.isoformat(), "hoje": self.hoje.isoformat(),
            "horizonte": self.horizonte.isoformat(), "membros": self.membros,
            "parametros_ajustados": self.identificado,
            "origem_parametros": self.parametros.origem,
            "plantas": int(len(self.risco)),
            "esporulando_agora_esperado": round(float(self.esporulando_agora.sum()), 2),
            "latentes_agora_esperado": round(float(self.latente_agora.sum()), 2),
            "infectadas_no_horizonte_esperado": round(float(self.risco.sum()), 2),
            "infectadas_no_horizonte_p10_p90": [
                round(float(self.diario[-1]["infectadas_p10"]), 2),
                round(float(self.diario[-1]["infectadas_p90"]), 2)] if self.diario else None,
            "janelas_de_infeccao": self.janelas,
            "focos_agora": len(self.focos_agora),
            "focos_no_horizonte": len(self.focos_horizonte),
            "focos_novos_no_horizonte": sum(f.novo for f in self.focos_horizonte),
            "rede_residual": self.rede,
            "avisos": self.avisos,
        }

    def to_dict(self, grafo: GrafoLavoura, plantas: str = "em_risco",
                risco_minimo: float = 0.05) -> dict[str, Any]:
        """``plantas``: ``todas``, ``em_risco`` (risco ou P(esporulando) >= ``risco_minimo``) ou
        ``nenhuma`` — num talhão de dezenas de milhares de pés, a lista inteira pesa."""
        saida = {**self.resumo(), "diario": self.diario,
                 "focos_agora_detalhe": [f.to_dict(grafo) for f in self.focos_agora],
                 "focos_horizonte_detalhe": [f.to_dict(grafo) for f in self.focos_horizonte]}
        if plantas == "nenhuma":
            return saida
        if plantas == "todas":
            idx = range(len(self.risco))
        else:
            idx = np.flatnonzero((self.risco >= risco_minimo)
                                 | (self.esporulando_agora >= risco_minimo))
        saida["plantas_detalhe"] = [{
            "plant_id": grafo.plant_ids[i], "row_id": grafo.ruas[int(grafo.rua[i])],
            "x_along_m": round(float(grafo.x_m[i]), 3),
            "lat": None if not np.isfinite(grafo.lat[i]) else float(grafo.lat[i]),
            "lon": None if not np.isfinite(grafo.lon[i]) else float(grafo.lon[i]),
            "esporulando_agora": round(float(self.esporulando_agora[i]), 4),
            "latente_agora": round(float(self.latente_agora[i]), 4),
            "risco": round(float(self.risco[i]), 4),
            "risco_p10": round(float(self.risco_p10[i]), 4),
            "risco_p90": round(float(self.risco_p90[i]), 4),
            "sinal_no_horizonte": round(float(self.sinal_no_horizonte[i]), 4),
            "sinal_corrigido": None if self.sinal_corrigido is None
            else round(float(self.sinal_corrigido[i]), 4)} for i in idx]
        return saida


def prever(grafo: GrafoLavoura, doenca: str, modelo: ModeloAjustado, serie: SerieClimatica, *,
           hoje: date | None = None, horizonte_dias: int = 21, membros: int = 16,
           semente: int = 0, tratamentos: Sequence[Tratamento] = (),
           limiar_risco: float = LIMIAR_RISCO, rede: RedeResidual | None = None) -> Previsao:
    """Assimila as passadas e projeta até ``hoje + horizonte_dias``.

    ``tratamentos`` (dias contados a partir da data da última passada) servem para comparar
    cenários: a prescrição orientada por risco (:mod:`app.gemeo.prescricao_risco`) usa isso para
    medir o que a aplicação evita. ``rede``: a residual adotada pelo portão
    (:func:`app.gemeo.rede.ajustar_rede`), aplicada só ao sinal e só no horizonte de uma passada.
    """
    if horizonte_dias < 1:
        raise ValueError("horizonte_dias precisa ser ao menos 1.")
    rng = np.random.default_rng(semente)
    ultimo = int(grafo.dia_obs.max())
    data_base = grafo.inicio + timedelta(days=dia_da_ultima_assimilacao(grafo))
    hoje = hoje or max(date.today(), data_base)
    if hoje < data_base:
        raise ValueError(f"hoje ({hoje}) antes da última passada ({data_base}).")
    horizonte = hoje + timedelta(days=horizonte_dias)
    fim = horizonte + timedelta(days=1)
    clima = serie.periodo(grafo.inicio, max(fim, grafo.inicio + timedelta(days=ultimo + 1)))
    avisos = []
    fontes = {d.fonte for d in clima[(data_base - grafo.inicio).days:]}
    if "climatologia" in fontes:
        avisos.append("Parte do horizonte usa climatologia (sem previsão de tempo para esses "
                      "dias): a faixa do risco é larga ali, e deve ser.")
    if not modelo.diagnostico.get("identificado", False):
        avisos.append("Parâmetros perto do prior, não do talhão: as passadas ainda quase não "
                      "moveram o ajuste. A previsão vale como ordem de grandeza.")
    amostras = modelo.amostras(membros, rng)
    n = grafo.n
    datas_futuras = [data_base + timedelta(days=k)
                     for k in range((horizonte - data_base).days + 1)]
    riscos = np.zeros((len(amostras), n))
    sinais = np.zeros((len(amostras), n))
    esporulando_fim = np.zeros((len(amostras), n))
    esporulando = np.zeros(n)
    latente = np.zeros(n)
    infectadas = np.zeros((len(amostras), len(datas_futuras)))
    esporulando_dia = np.zeros((len(amostras), len(datas_futuras)))
    favs = np.zeros((len(amostras), len(datas_futuras)))
    for m, p in enumerate(amostras):
        clima_m = perturbar_clima(clima, hoje, rng) if len(amostras) > 1 else list(clima)
        nucleo = nucleo_de_contagio(grafo, p)
        estado, dia = assimilar(grafo, doenca, clima_m, p, nucleo)
        if grafo.inicio + timedelta(days=dia) != data_base:
            raise RuntimeError("O relógio do filtro divergiu de dia_da_ultima_assimilacao.")
        esporulando += estado.I / len(amostras)
        latente += estado.E / len(amostras)
        for k, _data in enumerate(datas_futuras):
            infectadas[m, k] = float((estado.E + estado.I).sum())
            esporulando_dia[m, k] = float(estado.I.sum())
            if k == len(datas_futuras) - 1:
                break
            hoje_sim = clima_m[dia]
            favs[m, k] = favorabilidade(hoje_sim, p)
            estado, protecao = aplicar_tratamentos(estado, tratamentos, k)
            estado = passo(estado, hoje_sim, nucleo, p, protecao)
            dia += 1
        favs[m, -1] = favorabilidade(clima_m[min(dia, len(clima_m) - 1)], p)
        riscos[m] = estado.E + estado.I
        esporulando_fim[m] = estado.I
        sinais[m] = probabilidade_de_sinal(estado, p)
    risco = riscos.mean(axis=0)
    fav_media = favs.mean(axis=0)
    diario = [{"data": d.isoformat(),
               "fonte": clima[(d - grafo.inicio).days].fonte,
               "favorabilidade": round(float(fav_media[k]), 4),
               "favorabilidade_p10": round(float(np.quantile(favs[:, k], 0.1)), 4),
               "favorabilidade_p90": round(float(np.quantile(favs[:, k], 0.9)), 4),
               "esporulando": round(float(esporulando_dia[:, k].mean()), 2),
               "infectadas": round(float(infectadas[:, k].mean()), 2),
               "infectadas_p10": round(float(np.quantile(infectadas[:, k], 0.1)), 2),
               "infectadas_p90": round(float(np.quantile(infectadas[:, k], 0.9)), 2)}
              for k, d in enumerate(datas_futuras)]
    futuro = [k for k, d in enumerate(datas_futuras) if d >= hoje]
    janelas = janelas_de_infeccao([datas_futuras[k] for k in futuro], fav_media[futuro])
    agora = focos(grafo, esporulando, LIMIAR_FOCO_AGORA)
    adiante = focos(grafo, risco, limiar_risco)
    marcadas_agora = np.zeros(n, dtype=bool)
    for f in agora:
        marcadas_agora[f.plantas] = True
    for f in adiante:
        f.novo = not bool(marcadas_agora[f.plantas].any())
    sinal = sinais.mean(axis=0)
    sinal_corrigido, rede_info = None, None
    if rede is not None:
        dias_a_frente = (horizonte - data_base).days
        if aplicavel(rede, dias_a_frente):
            hist = historico(grafo, doenca, clima, modelo.parametros)
            sinal_corrigido = rede.corrigir(grafo, hist.final, sinal)
            rede_info = {**rede.diagnostico, "aplicada": True}
        else:
            rede_info = {**rede.diagnostico, "aplicada": False,
                         "motivo_nao_aplicada": f"horizonte de {dias_a_frente} dias fora do "
                         "de uma passada, em que a rede treinou"}
    return Previsao(
        field_id=grafo.field_id, doenca=doenca, data_base=data_base, hoje=hoje,
        horizonte=horizonte, membros=len(amostras), parametros=modelo.parametros,
        identificado=bool(modelo.diagnostico.get("identificado", False)),
        esporulando_agora=esporulando, latente_agora=latente, risco=risco,
        risco_p10=np.quantile(riscos, 0.1, axis=0), risco_p90=np.quantile(riscos, 0.9, axis=0),
        esporulando_no_horizonte=esporulando_fim.mean(axis=0),
        sinal_no_horizonte=sinal, diario=diario, janelas=janelas,
        focos_agora=agora, focos_horizonte=adiante, avisos=avisos,
        sinal_corrigido=sinal_corrigido, rede=rede_info)
