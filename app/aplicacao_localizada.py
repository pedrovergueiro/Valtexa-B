"""Prescrição variável preditiva: o mapa de risco vira taxa por trecho e PWM por bico.

A prescrição de :mod:`app.prescricao` (e :mod:`app.gemeo.prescricao_risco`) decide planta a
planta se trata, com dose fixa. Para o pulverizador de seções, o que se executa é uma **taxa
por trecho de rua** e, em cada bico, um **ciclo de trabalho de PWM**. Aqui, a partir da
previsão do gêmeo:

1. **mapa de risco por trecho** (``trecho_m`` ao longo da rua): risco previsto no horizonte
   (média e máximo), a **incerteza** (faixa p10 a p90 do conjunto), a **intensidade** de hoje
   (P(esporulando) — onde já há fonte), a **favorabilidade** do tempo (janelas de infecção do
   horizonte), o **histórico de intervenções** (residual de aplicação recente protege) e as
   **restrições operacionais** (trechos excluídos: APP, curso d'água, casa, declive);
2. **taxa por trecho**: zero abaixo do limiar de risco; entre o limiar e o de dose cheia, da
   dose mínima à dose padrão da bula; nunca acima da dose padrão (subir até a máxima de bula é
   decisão do agrônomo, não do mapa). É uma **regra**, não uma curva dose-resposta medida;
3. **PWM por bico**: a vazão que cada bico precisa (``taxa . velocidade . faixa do bico``)
   contra a nominal dá o ciclo; abaixo do mínimo do PWM (gota fora do espectro) ou acima de
   100%, o plano diz a velocidade que resolve dentro dos limites da máquina, ou que o trecho é
   inviável — **nunca** corta o ciclo calado;
4. **verificações determinísticas**: cultura e alvo da bula, registro, carência, faixa de dose,
   e as condições do dia da aplicação (vento, temperatura, umidade — limites de boas práticas,
   configuráveis, não regulação).

O resultado é **proposta**: o agrônomo revisa e aprova, e o Safety Engine do SaaS confere de
novo cada trecho contra a bula e a aprovação (presa ao hash do mapa) antes de liberar.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np

from app.gemeo.clima import DiaClimatico
from app.gemeo.grafo import GrafoLavoura
from app.gemeo.previsao import Previsao


class AplicacaoImpossivelError(ValueError):
    """O mapa não tem como ser montado com o que veio (a mensagem diz por quê)."""


@dataclass(frozen=True)
class ProdutoDaBula:
    nome: str
    registro_mapa: str
    culturas: tuple[str, ...]
    alvos: tuple[str, ...]
    dose_min_l_ha: float
    dose_l_ha: float
    dose_max_l_ha: float
    carencia_dias: int
    residual_dias: int = 14
    vento_max_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.registro_mapa.strip():
            raise AplicacaoImpossivelError(f"{self.nome!r} sem registro no MAPA.")
        if not 0 < self.dose_min_l_ha <= self.dose_l_ha <= self.dose_max_l_ha:
            raise AplicacaoImpossivelError("exige 0 < dose mínima <= dose padrão <= máxima.")


@dataclass(frozen=True)
class Equipamento:
    nome: str
    vazao_nominal_l_min: float
    """Vazão de cada bico a 100% do ciclo, na pressão de trabalho."""
    faixa_por_bico_m: float
    """Largura que cada bico cobre na passada (no café, a copa de um lado dividida pelos bicos
    da seção)."""
    bicos_por_lado: int
    velocidade_m_s: float
    velocidade_min_m_s: float
    velocidade_max_m_s: float
    ciclo_min: float = 0.2
    """Abaixo disto o PWM muda o espectro de gotas (deriva): não se usa."""
    ciclo_max: float = 1.0
    frequencia_hz: float = 10.0

    def __post_init__(self) -> None:
        if self.vazao_nominal_l_min <= 0 or self.faixa_por_bico_m <= 0 or self.bicos_por_lado < 1:
            raise AplicacaoImpossivelError("vazão, faixa e número de bicos precisam ser positivos.")
        if not 0 < self.velocidade_min_m_s <= self.velocidade_m_s <= self.velocidade_max_m_s:
            raise AplicacaoImpossivelError("exige velocidade mínima <= planejada <= máxima.")
        if not 0 < self.ciclo_min < self.ciclo_max <= 1.0:
            raise AplicacaoImpossivelError("exige 0 < ciclo mínimo < ciclo máximo <= 1.")


@dataclass(frozen=True)
class Regras:
    limiar_risco: float = 0.15
    """Risco médio previsto do trecho abaixo do qual não se aplica (escolha do agrônomo)."""
    limiar_dose_cheia: float = 0.5
    incerteza_para_revisar: float = 0.4
    """Faixa p10 a p90 do risco acima disto: o trecho vai marcado para revisão."""
    vento_max_ms: float = 2.8
    """≈ 10 km/h: boa prática para deriva; o produto pode exigir menos."""
    temperatura_max_c: float = 30.0
    umidade_min: float = 55.0

    def __post_init__(self) -> None:
        if not 0 < self.limiar_risco < self.limiar_dose_cheia <= 1:
            raise AplicacaoImpossivelError("exige 0 < limiar_risco < limiar_dose_cheia <= 1.")


@dataclass(frozen=True)
class Exclusao:
    row_id: str
    x_inicio_m: float
    x_fim_m: float
    motivo: str


@dataclass(frozen=True)
class AplicacaoAnterior:
    data: date
    row_ids: tuple[str, ...]
    """Ruas tratadas (vazio = o talhão inteiro)."""
    residual_dias: int


@dataclass
class Zona:
    row_id: str
    x_inicio_m: float
    x_fim_m: float
    plantas: int
    risco_medio: float
    risco_max: float
    risco_p10: float
    risco_p90: float
    intensidade: float
    protegida_ate: str | None
    excluida: str | None
    taxa_l_ha: float = 0.0
    motivo: str = ""
    revisar: bool = False
    bico: dict[str, Any] = field(default_factory=dict)

    @property
    def incerteza(self) -> float:
        return self.risco_p90 - self.risco_p10

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["incerteza"] = round(self.incerteza, 4)
        return d


# ---------------------------------------------------------------------------------------------
# 1. Mapa de risco por trecho
# ---------------------------------------------------------------------------------------------


def mapa_de_risco(grafo: GrafoLavoura, previsao: Previsao, trecho_m: float = 5.0,
                  exclusoes: Sequence[Exclusao] = (),
                  anteriores: Sequence[AplicacaoAnterior] = (),
                  data_aplicacao: date | None = None) -> list[Zona]:
    if trecho_m <= 0:
        raise AplicacaoImpossivelError("trecho_m precisa ser positivo.")
    quando = data_aplicacao or previsao.hoje
    zonas = []
    for r, row_id in enumerate(grafo.ruas):
        na_rua = np.flatnonzero(grafo.rua == r)
        if not len(na_rua):
            continue
        x = grafo.x_m[na_rua]
        inicio = float(np.floor(x.min() / trecho_m) * trecho_m)
        while inicio <= x.max():
            fim = inicio + trecho_m
            idx = na_rua[(x >= inicio) & (x < fim)]
            if len(idx):
                excluida = next((e.motivo for e in exclusoes if e.row_id == row_id
                                 and e.x_inicio_m < fim and e.x_fim_m > inicio), None)
                protegida = None
                for a in anteriores:
                    ate = date.fromordinal(a.data.toordinal() + a.residual_dias)
                    if (not a.row_ids or row_id in a.row_ids) and a.data <= quando <= ate:
                        protegida = max(protegida or ate.isoformat(), ate.isoformat())
                zonas.append(Zona(
                    row_id, round(inicio, 2), round(fim, 2), int(len(idx)),
                    round(float(previsao.risco[idx].mean()), 4),
                    round(float(previsao.risco[idx].max()), 4),
                    round(float(previsao.risco_p10[idx].mean()), 4),
                    round(float(previsao.risco_p90[idx].mean()), 4),
                    round(float(previsao.esporulando_agora[idx].mean()), 4),
                    protegida, excluida))
            inicio = fim
    return zonas


# ---------------------------------------------------------------------------------------------
# 2. Taxa por trecho
# ---------------------------------------------------------------------------------------------


def taxas(zonas: list[Zona], produto: ProdutoDaBula, regras: Regras) -> list[Zona]:
    for z in zonas:
        z.revisar = z.incerteza >= regras.incerteza_para_revisar
        if z.excluida:
            z.taxa_l_ha, z.motivo = 0.0, f"excluído: {z.excluida}"
        elif z.protegida_ate:
            z.taxa_l_ha, z.motivo = 0.0, f"residual da aplicação anterior até {z.protegida_ate}"
        elif z.risco_medio < regras.limiar_risco:
            z.taxa_l_ha, z.motivo = 0.0, (f"risco {z.risco_medio:.2f} abaixo do limiar "
                                          f"{regras.limiar_risco:.2f}")
        else:
            fracao = min(1.0, (z.risco_medio - regras.limiar_risco)
                         / (regras.limiar_dose_cheia - regras.limiar_risco))
            taxa = produto.dose_min_l_ha + fracao * (produto.dose_l_ha - produto.dose_min_l_ha)
            z.taxa_l_ha = round(min(max(taxa, produto.dose_min_l_ha), produto.dose_l_ha), 3)
            z.motivo = (f"risco {z.risco_medio:.2f}: {fracao:.0%} do caminho da dose mínima à "
                        "padrão de bula")
    return zonas


# ---------------------------------------------------------------------------------------------
# 3. PWM por bico
# ---------------------------------------------------------------------------------------------


def vazao_por_bico_l_min(taxa_l_ha: float, velocidade_m_s: float, faixa_m: float) -> float:
    """``L/ha . m/s . m`` → L/min: ``taxa/10.000 m² . v . faixa . 60``."""
    return taxa_l_ha * velocidade_m_s * faixa_m * 0.006


def plano_de_bicos(zonas: list[Zona], eq: Equipamento) -> list[Zona]:
    for z in zonas:
        if z.taxa_l_ha <= 0:
            z.bico = {"ciclo": 0.0, "velocidade_m_s": eq.velocidade_m_s, "viavel": True,
                      "estado": "fechado"}
            continue
        q = vazao_por_bico_l_min(z.taxa_l_ha, eq.velocidade_m_s, eq.faixa_por_bico_m)
        ciclo = q / eq.vazao_nominal_l_min
        velocidade, estado = eq.velocidade_m_s, "ok"
        if ciclo < eq.ciclo_min or ciclo > eq.ciclo_max:
            # O ciclo é proporcional à velocidade (mesma taxa por área): a velocidade que põe
            # o ciclo no limite mais próximo.
            alvo = eq.ciclo_min if ciclo < eq.ciclo_min else eq.ciclo_max
            velocidade = eq.velocidade_m_s * alvo / ciclo
            if eq.velocidade_min_m_s <= velocidade <= eq.velocidade_max_m_s:
                ciclo, estado = alvo, ("acelerar" if alvo == eq.ciclo_min else "desacelerar")
            else:
                velocidade = min(max(velocidade, eq.velocidade_min_m_s), eq.velocidade_max_m_s)
                ciclo = vazao_por_bico_l_min(z.taxa_l_ha, velocidade, eq.faixa_por_bico_m) / (
                    eq.vazao_nominal_l_min)
                estado = "inviavel"
        z.bico = {"ciclo": round(ciclo, 4), "velocidade_m_s": round(velocidade, 3),
                  "vazao_l_min": round(q * velocidade / eq.velocidade_m_s, 4),
                  "viavel": estado != "inviavel", "estado": estado,
                  "frequencia_hz": eq.frequencia_hz}
    return zonas


# ---------------------------------------------------------------------------------------------
# 4. Verificações e o mapa inteiro
# ---------------------------------------------------------------------------------------------


def verificacoes(produto: ProdutoDaBula, cultura: str, doenca: str, zonas: list[Zona],
                 regras: Regras, dia: DiaClimatico | None, dias_ate_colheita: int | None
                 ) -> list[dict[str, Any]]:
    def item(codigo: str, ok: bool, texto: str) -> dict[str, Any]:
        return {"codigo": codigo, "ok": bool(ok), "texto": texto}

    aplicadas = [z for z in zonas if z.taxa_l_ha > 0]
    vento_max = min(regras.vento_max_ms, produto.vento_max_ms or regras.vento_max_ms)
    saida = [
        item("cultura", cultura in produto.culturas,
             f"cultura {cultura!r} {'na' if cultura in produto.culturas else 'fora da'} bula"),
        item("alvo", doenca in produto.alvos,
             f"alvo {doenca!r} {'autorizado' if doenca in produto.alvos else 'não autorizado'}"),
        item("registro", bool(produto.registro_mapa.strip()),
             f"registro MAPA {produto.registro_mapa}"),
        item("faixa_de_dose", all(produto.dose_min_l_ha <= z.taxa_l_ha <= produto.dose_max_l_ha
                                  for z in aplicadas),
             f"todo trecho aplicado entre {produto.dose_min_l_ha} e {produto.dose_max_l_ha} L/ha"),
        item("pwm", all(z.bico.get("viavel", True) for z in zonas),
             "todo trecho com ciclo de PWM dentro dos limites da máquina" if all(
                 z.bico.get("viavel", True) for z in zonas) else
             f"{sum(not z.bico.get('viavel', True) for z in zonas)} trecho(s) inviável(is) "
             "na faixa de velocidade"),
    ]
    if dias_ate_colheita is None:
        saida.append(item("carencia", False, "dias até a colheita não informados"))
    else:
        saida.append(item("carencia", dias_ate_colheita >= produto.carencia_dias,
                          f"{dias_ate_colheita} dias até a colheita; carência "
                          f"{produto.carencia_dias}"))
    if dia is None:
        saida.append(item("tempo", False, "sem previsão do tempo para o dia da aplicação"))
    else:
        vento = dia.vento_ms
        saida += [
            item("vento", vento is not None and vento <= vento_max,
                 "sem vento previsto" if vento is None else
                 f"vento {vento:.1f} m/s (limite {vento_max:.1f})"),
            item("temperatura", dia.t_max <= regras.temperatura_max_c,
                 f"máxima {dia.t_max:.0f} °C (limite {regras.temperatura_max_c:.0f})"),
            item("umidade", dia.ur_media >= regras.umidade_min,
                 f"umidade {dia.ur_media:.0f}% (mínimo {regras.umidade_min:.0f}%)"),
        ]
    return saida


def mapa_de_aplicacao(grafo: GrafoLavoura, previsao: Previsao, produto: ProdutoDaBula,
                      equipamento: Equipamento, regras: Regras = Regras(), *,
                      trecho_m: float = 5.0, exclusoes: Sequence[Exclusao] = (),
                      anteriores: Sequence[AplicacaoAnterior] = (),
                      data_aplicacao: date | None = None, dia: DiaClimatico | None = None,
                      dias_ate_colheita: int | None = None) -> dict[str, Any]:
    zonas = mapa_de_risco(grafo, previsao, trecho_m, exclusoes, anteriores, data_aplicacao)
    zonas = plano_de_bicos(taxas(zonas, produto, regras), equipamento)
    checagens = verificacoes(produto, grafo.crop, previsao.doenca, zonas, regras, dia,
                             dias_ate_colheita)
    largura = equipamento.faixa_por_bico_m * equipamento.bicos_por_lado * 2
    area = {z.row_id + str(z.x_inicio_m): (z.x_fim_m - z.x_inicio_m) * largura / 10_000
            for z in zonas}
    litros = sum(z.taxa_l_ha * area[z.row_id + str(z.x_inicio_m)] for z in zonas)
    cheio = produto.dose_l_ha * sum(area.values())
    corpo = {
        "estado": "proposta",
        "doenca": previsao.doenca, "produto": asdict(produto),
        "equipamento": asdict(equipamento), "regras": asdict(regras),
        "data_aplicacao": (data_aplicacao or previsao.hoje).isoformat(),
        "horizonte": previsao.horizonte.isoformat(), "trecho_m": trecho_m,
        "zonas": [z.to_dict() for z in zonas],
        "totais": {"trechos": len(zonas), "aplicados": sum(z.taxa_l_ha > 0 for z in zonas),
                   "para_revisar": sum(z.revisar for z in zonas),
                   "litros": round(litros, 2), "litros_talhao_inteiro": round(cheio, 2),
                   "economia_fracao": round(1 - litros / cheio, 4) if cheio else None,
                   "taxa_max_l_ha": max((z.taxa_l_ha for z in zonas), default=0.0)},
        "verificacoes": checagens,
        "bloqueios": [c["texto"] for c in checagens if not c["ok"]],
        "avisos": [
            *previsao.avisos,
            "Taxa por regra (limiares do agrônomo), não por curva dose-resposta medida.",
            "Limites de vento, temperatura e umidade são boas práticas configuráveis; a bula "
            "do produto e a regulação vigente prevalecem.",
            "Nada é aplicado sem a aprovação do agrônomo e a liberação do Safety Engine."],
    }
    corpo["sha256"] = hashlib.sha256(json.dumps(
        {k: corpo[k] for k in ("zonas", "produto", "equipamento", "data_aplicacao")},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return corpo
