"""Prescrição variável orientada por risco: tratar onde a doença **vai** estar, antes de ela estar.

A prescrição de :mod:`app.prescricao` decide planta a planta com o ganho marginal (o direto e o
do contágio evitado) e confere a bula e a carência. Aqui ela recebe, em vez da posterior de
hoje, o **risco previsto** pelo gêmeo (:mod:`app.gemeo.previsao`): P(planta infectada — latente
ou esporulando — no horizonte). É a diferença entre pulverizar a reboleira que o Rover já vê e
proteger as plantas que pegaram e ainda não mostram, e as que vão pegar na próxima chuva.

Mais três coisas que só o gêmeo dá:

* **Alcance ajustado**: o núcleo de contágio da prescrição usa os alcances ajustados às
  passadas do talhão (``Contagio(ajustado=True)``) quando o ajuste se moveu; senão, os do
  prior, e o aviso da prescrição continua dizendo que não são medição.
* **Quando aplicar**: o protetor tem de estar na folha **antes** da janela de infecção (o dia
  em que temperatura e molhamento deixam o fungo germinar). A data recomendada é a véspera da
  primeira janela do horizonte; sem janela, a aplicação pode esperar a próxima passada.
* **O que se evita**: os cenários (sem aplicar, a prescrição na data, a prescrição sete dias
  depois, o talhão inteiro) rodados no próprio gêmeo, com a mesma incerteza da previsão. É
  aritmética do modelo sobre os números da bula e do produtor, **não** medição de campo — e o
  relatório diz isso.

Não pulveriza nada: recomenda. Quem aprova é o agrônomo, e quem libera a ação física é a camada
de segurança do SaaS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np

from app.gemeo.clima import SerieClimatica
from app.gemeo.epidemia import ModeloAjustado, Tratamento
from app.gemeo.grafo import GrafoLavoura
from app.gemeo.previsao import Previsao, prever
from app.prescricao import (
    Contagio,
    Economia,
    Prescricao,
    PrescricaoImpossivelError,
    Produto,
    prescrever,
)

RISCO_MINIMO = 0.02
"""Abaixo disso a planta nem entra como candidata (a prescrição é densa: O(n²) em memória)."""
MAX_CANDIDATAS = 1500
ATRASO_DIAS = 7


@dataclass
class PrescricaoPorRisco:
    prescricao: Prescricao
    data_aplicacao: date | None
    motivo_data: str
    cenarios: dict[str, dict[str, Any]]
    candidatas: int
    avisos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"prescricao": self.prescricao.to_dict(),
                "data_aplicacao": None if self.data_aplicacao is None
                else self.data_aplicacao.isoformat(),
                "motivo_data": self.motivo_data, "cenarios": self.cenarios,
                "candidatas": self.candidatas, "avisos": self.avisos}


def data_de_aplicacao(previsao: Previsao) -> tuple[date | None, str]:
    """A véspera da primeira janela de infecção a partir de hoje (ou hoje, se ela já começou)."""
    for janela in previsao.janelas:
        inicio = date.fromisoformat(janela["inicio"])
        if inicio < previsao.hoje:
            continue
        data = max(previsao.hoje, inicio - timedelta(days=1))
        return data, (f"protetor na folha antes da janela de infecção de {janela['inicio']} a "
                      f"{janela['fim']} ({janela['dias']} dia(s), favorabilidade até "
                      f"{janela['favorabilidade_max']})")
    return None, ("sem janela de infecção até o horizonte: a aplicação preventiva pode esperar "
                  "a próxima passada")


def _cenario(grafo: GrafoLavoura, doenca: str, modelo: ModeloAjustado, serie: SerieClimatica,
             previsao: Previsao, tratamentos: list[Tratamento], membros: int,
             semente: int) -> dict[str, Any]:
    horizonte = (previsao.horizonte - previsao.hoje).days
    p = prever(grafo, doenca, modelo, serie, hoje=previsao.hoje, horizonte_dias=horizonte,
               membros=membros, semente=semente, tratamentos=tratamentos)
    fim = p.diario[-1]
    return {"infectadas_no_horizonte": fim["infectadas"],
            "infectadas_p10": fim["infectadas_p10"], "infectadas_p90": fim["infectadas_p90"],
            "esporulando_no_horizonte": fim["esporulando"]}


def prescrever_por_risco(grafo: GrafoLavoura, previsao: Previsao, modelo: ModeloAjustado,
                         serie: SerieClimatica, *, produto: Produto, economia: Economia,
                         residual_dias: int, curativo: float = 0.0,
                         dias_ate_colheita: int | None = None,
                         risco_minimo: float = RISCO_MINIMO,
                         max_candidatas: int = MAX_CANDIDATAS, membros: int = 8,
                         semente: int = 0) -> PrescricaoPorRisco:
    """Prescreve sobre o risco previsto e mede, no gêmeo, o que a prescrição evita.

    ``residual_dias`` (quanto o produto protege depois de aplicado) e ``curativo`` (a fração do
    latente que ele elimina) são da bula ou do ensaio do produtor — como a eficácia, não têm
    padrão aqui.
    """
    doenca = previsao.doenca
    if residual_dias < 1:
        raise PrescricaoImpossivelError("residual_dias precisa ser ao menos 1 (está na bula).")
    if not 0.0 <= curativo <= 1.0:
        raise PrescricaoImpossivelError("curativo fora de [0, 1].")
    avisos: list[str] = list(previsao.avisos)
    ordem = np.argsort(-previsao.risco, kind="stable")
    candidatas = [int(i) for i in ordem if previsao.risco[i] >= risco_minimo]
    if len(candidatas) > max_candidatas:
        avisos.append(f"{len(candidatas)} plantas com risco >= {risco_minimo}: a prescrição "
                      f"considerou as {max_candidatas} de maior risco (aumente max_candidatas "
                      "para ver o resto).")
        candidatas = candidatas[:max_candidatas]
    if not candidatas:
        raise PrescricaoImpossivelError(
            f"nenhuma planta com risco previsto >= {risco_minimo} até {previsao.horizonte}: "
            "nada a prescrever.")
    plantas = [SimpleNamespace(
        plant_id=grafo.plant_ids[i], row_id=grafo.ruas[int(grafo.rua[i])],
        x_along_m=float(grafo.x_m[i]), posteriors={doenca: float(previsao.risco[i])},
        lat=None if not np.isfinite(grafo.lat[i]) else float(grafo.lat[i]),
        lon=None if not np.isfinite(grafo.lon[i]) else float(grafo.lon[i]))
        for i in candidatas]
    p = modelo.parametros
    ajustado = bool(modelo.diagnostico.get("identificado", False))
    contagio = Contagio(alcance_na_rua_m=p.alcance_na_rua_m,
                        alcance_entre_ruas_m=p.alcance_entre_ruas_m, ajustado=ajustado)
    prescricao = prescrever(
        plantas, doenca=doenca, produto=produto, economia=economia, crop=grafo.crop,
        field_id=grafo.field_id, pass_id=grafo.passadas[-1],
        espacamento_entre_ruas_m=grafo.espacamento_entre_ruas_m, contagio=contagio,
        dias_ate_colheita=dias_ate_colheita)
    data, motivo = data_de_aplicacao(previsao)
    tratar = np.zeros(grafo.n, dtype=bool)
    indice = {pid: i for i, pid in enumerate(grafo.plant_ids)}
    for pl in prescricao.plantas:
        if pl.tratar:
            tratar[indice[pl.plant_id]] = True
    dia = (data or previsao.hoje) - previsao.data_base
    dia_k = dia.days

    def aplicacao(mascara: np.ndarray, dia_aplicacao: int) -> list[Tratamento]:
        return [Tratamento(dia=dia_aplicacao, plantas=mascara.astype(float),
                           protecao=produto.eficacia, residual_dias=residual_dias,
                           curativo=curativo)]

    cenarios = {
        "sem_aplicacao": _cenario(grafo, doenca, modelo, serie, previsao, [], membros, semente),
        "prescricao_na_data": _cenario(grafo, doenca, modelo, serie, previsao,
                                       aplicacao(tratar, dia_k), membros, semente),
        f"prescricao_{ATRASO_DIAS}_dias_depois": _cenario(
            grafo, doenca, modelo, serie, previsao, aplicacao(tratar, dia_k + ATRASO_DIAS),
            membros, semente),
        "talhao_inteiro_na_data": _cenario(grafo, doenca, modelo, serie, previsao,
                                           aplicacao(np.ones(grafo.n, dtype=bool), dia_k),
                                           membros, semente),
    }
    cenarios["prescricao_na_data"]["plantas_tratadas"] = int(tratar.sum())
    cenarios[f"prescricao_{ATRASO_DIAS}_dias_depois"]["plantas_tratadas"] = int(tratar.sum())
    cenarios["talhao_inteiro_na_data"]["plantas_tratadas"] = grafo.n
    cenarios["sem_aplicacao"]["plantas_tratadas"] = 0
    avisos.append("Os cenários são o gêmeo rodando os números da bula e do produtor (eficácia, "
                  "residual, curativo) — aritmética do modelo, não medição de campo.")
    if data is not None and data > previsao.horizonte:
        avisos.append("A data recomendada passa do horizonte da previsão.")
    return PrescricaoPorRisco(prescricao, data, motivo, cenarios, len(candidatas), avisos)
