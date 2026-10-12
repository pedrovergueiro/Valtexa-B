"""Prescrição planta a planta: tratar quem precisa, e o que isso custa e poupa.

O resto do sistema responde *o que a planta tem*. Este módulo responde a pergunta
seguinte, que é a que o produtor faz: **em quais plantas vale a pena gastar
defensivo, e quanto eu deixo de gastar por não pulverizar o talhão inteiro?**

A decisão **não** é um limiar sobre a posterior. Uma planta doente que fica sem
tratamento não custa só ela: ela é fonte de inóculo para as vizinhas na semana
seguinte. Quem ignora isso trata pouco demais e perde o talhão, ou trata tudo e
perde dinheiro. Aqui, o valor de tratar uma planta é

    valor = eficácia x perda_por_planta x risco x (1 + R)  -  custo_de_tratar

onde

* ``risco`` é a probabilidade de a planta **estar ou ficar** doente até a próxima
  intervenção: ``p + (1 - p) * q``, com ``p`` vindo da fusão por planta
  (:mod:`app.plant_fusion`) e ``q`` da pressão de infecção das vizinhas;
* ``R`` é o número esperado de vizinhas **suscetíveis** que ela infectaria se
  ficasse sem tratamento.

O ``(1 + R)`` é o módulo inteiro em um símbolo: *tratar uma planta vale a planta
mais tudo o que ela infectaria*. É daí que sai, sozinha, a borda de segurança em
volta da reboleira — nenhuma regra de bolso do tipo "pulverize duas ruas em
volta". A borda nasce do alcance de contágio e do preço do produto, e encolhe ou
cresce quando um dos dois muda.

O núcleo de contágio é **anisotrópico**: o alcance ao longo da rua é maior que o
alcance atravessando a entrelinha, porque folha encosta em folha na fila e não
na entrelinha. É a mesma geometria que o rover já usa para associar recorte a
planta.

De onde vêm os números
----------------------
**Os parâmetros econômicos e agronômicos são do produtor, não nossos.** Preço do
produto, perda por planta doente, eficácia e alcance de contágio entram por
:class:`Economia` e :class:`Produto` e **não têm padrão**: sem eles o módulo
recusa, em vez de inventar. O que sai daqui é aritmética sobre os números que
entraram, e a palavra "economia" no relatório significa isso, não uma medição
nossa.

A prescrição também herda o erro do diagnóstico. No teste de campo (recortes na
escala do rover, de fotos que nenhum treino viu) o modelo de banana acerta 0,829
e o de citros 0,969 — ver ``MODELS.md``. Uma planta diagnosticada errado é
tratada errado, e nenhuma conta aqui conserta isso.

O que este módulo não faz
-------------------------
Não pulveriza nada. Devolve recomendação. Quem aprova é o agrônomo e quem libera
a ação física é a camada de segurança do SaaS, que relê tudo do banco e falha
fechada — ver ``VALTEXA-SAAS/docs/SAFETY.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "Contagio",
    "Economia",
    "PlantaPrescrita",
    "PrescricaoImpossivelError",
    "Prescricao",
    "Produto",
    "PontoDaCurva",
    "prescrever",
]


class PrescricaoImpossivelError(ValueError):
    """Falta dado ou a aplicação é proibida. A mensagem diz qual das duas e por quê."""


# ---------------------------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Produto:
    """O defensivo, como está na bula e na nota fiscal.

    ``registro_mapa``, ``culturas`` e ``carencia_dias`` não são burocracia: a
    prescrição que os ignora é recusada lá na frente pela camada de segurança, e
    é melhor recusar aqui, antes de o agrônomo perder tempo.
    """

    nome: str
    registro_mapa: str
    culturas: tuple[str, ...]
    """Culturas autorizadas na bula, normalizadas ('banana', 'citrus')."""
    doencas: tuple[str, ...]
    """Alvos autorizados na bula, com os nomes das classes do modelo."""
    dose_ml_por_planta: float
    dose_min_ml_por_planta: float
    dose_max_ml_por_planta: float
    preco_por_litro: float
    eficacia: float
    """Fração do dano evitada quando a planta é tratada, de 0 a 1. Vem da bula ou
    do ensaio do produtor; não medimos isso."""
    carencia_dias: int
    """Intervalo mínimo entre a aplicação e a colheita."""

    def __post_init__(self) -> None:
        if not 0.0 < self.eficacia <= 1.0:
            raise PrescricaoImpossivelError(f"eficácia fora de (0, 1]: {self.eficacia}")
        if self.dose_ml_por_planta <= 0 or self.preco_por_litro < 0:
            raise PrescricaoImpossivelError("dose e preço precisam ser positivos.")
        if not self.registro_mapa.strip():
            raise PrescricaoImpossivelError(
                f"{self.nome!r} sem número de registro no MAPA: não dá para prescrever."
            )

    @property
    def custo_por_planta(self) -> float:
        return self.dose_ml_por_planta / 1000.0 * self.preco_por_litro


@dataclass(frozen=True)
class Economia:
    """Os números do produtor. Nenhum tem padrão, e é deliberado."""

    perda_por_planta: float
    """Quanto vale, em dinheiro, perder uma planta para a doença nesta safra."""
    custo_operacional_por_planta: float = 0.0
    """Mão de obra, máquina e combustível rateados por planta tratada."""
    orcamento: float | None = None
    """Teto de gasto. ``None`` = sem teto; trata tudo que se paga."""

    def __post_init__(self) -> None:
        if self.perda_por_planta <= 0:
            raise PrescricaoImpossivelError(
                "perda_por_planta precisa ser positiva: sem ela não há o que comparar com o custo."
            )
        if self.orcamento is not None and self.orcamento < 0:
            raise PrescricaoImpossivelError("orçamento negativo.")


@dataclass(frozen=True)
class Contagio:
    """Alcance do contágio, em metros, ao longo da rua e atravessando a entrelinha.

    Os padrões abaixo são **escolha nossa**, não medição: um alcance da ordem do
    espaçamento entre plantas na fila, e metade disso atravessando a entrelinha.
    Servem para rodar; o número honesto sai do módulo de epidemia, ajustado sobre
    as passadas do próprio talhão, e deve substituí-los assim que existir.
    """

    alcance_na_rua_m: float = 3.0
    alcance_entre_ruas_m: float = 1.5
    ajustado: bool = False
    """``True`` só quando os alcances vieram de ajuste sobre as passadas do talhão."""

    def __post_init__(self) -> None:
        if self.alcance_na_rua_m <= 0 or self.alcance_entre_ruas_m <= 0:
            raise PrescricaoImpossivelError("alcances de contágio precisam ser positivos.")


# ---------------------------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------------------------


@dataclass
class PlantaPrescrita:
    """A decisão numa planta, com o porquê legível."""

    plant_id: str
    row_id: str
    x_along_m: float
    tratar: bool
    ordem: int | None
    """Posição na ordem de escolha: 0 é a planta que mais reduz a perda do talhão."""
    motivo: str
    posterior: float
    """Probabilidade de a planta já estar doente, da fusão por planta."""
    risco_sem_nenhum_tratamento: float
    """Probabilidade de estar ou ficar doente se ninguém no talhão fosse tratado."""
    risco_final: float
    """A mesma probabilidade, já descontado o efeito da prescrição inteira."""
    beneficio_direto: float
    """Dinheiro que o tratamento salva nesta planta."""
    beneficio_contagio: float
    """Dinheiro que ele salva nas outras, por esta deixar de ser fonte de inóculo."""
    valor: float
    """Benefício marginal no momento da escolha, menos o custo. Negativo = não se paga."""
    lat: float | None = None
    lon: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PontoDaCurva:
    """Um ponto da curva risco-custo: o agrônomo escolhe onde parar."""

    plantas_tratadas: int
    custo: float
    litros: float
    doentes_esperados: float
    """Nº esperado de plantas doentes no talhão **depois** deste tanto de tratamento."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prescricao:
    """O resultado: o que tratar, o que custa, o que poupa e o risco que sobra."""

    field_id: str
    crop: str
    pass_id: str | None
    doenca: str
    produto: str
    plantas: list[PlantaPrescrita]
    plantas_no_talhao: int
    plantas_tratadas: int
    litros: float
    custo: float
    litros_talhao_inteiro: float
    custo_talhao_inteiro: float
    doentes_sem_nenhum_tratamento: float
    """Nº esperado de plantas doentes se nada fosse feito. É a base de comparação."""
    doentes_com_a_prescricao: float
    """O mesmo número depois de tratar só as plantas escolhidas."""
    doentes_tratando_tudo: float
    """E se o talhão inteiro fosse pulverizado — o teto do que dá para evitar."""
    curva: list[PontoDaCurva]
    avisos: list[str]
    limitado_por_orcamento: bool
    contagio_ajustado: bool

    @property
    def doencas_evitadas(self) -> float:
        return self.doentes_sem_nenhum_tratamento - self.doentes_com_a_prescricao

    @property
    def fracao_do_ganho_possivel(self) -> float:
        """Quanto do ganho máximo (pulverizar tudo) a prescrição já captura."""
        teto = self.doentes_sem_nenhum_tratamento - self.doentes_tratando_tudo
        return self.doencas_evitadas / teto if teto > 0 else 0.0

    @property
    def litros_poupados(self) -> float:
        return self.litros_talhao_inteiro - self.litros

    @property
    def custo_evitado(self) -> float:
        return self.custo_talhao_inteiro - self.custo

    @property
    def fracao_poupada(self) -> float:
        if self.custo_talhao_inteiro <= 0:
            return 0.0
        return self.custo_evitado / self.custo_talhao_inteiro

    def to_dict(self) -> dict[str, Any]:
        saida = asdict(self)
        saida["litros_poupados"] = self.litros_poupados
        saida["custo_evitado"] = self.custo_evitado
        saida["fracao_poupada"] = self.fracao_poupada
        saida["doencas_evitadas"] = self.doencas_evitadas
        saida["fracao_do_ganho_possivel"] = self.fracao_do_ganho_possivel
        return saida

    def to_geojson(self) -> dict[str, Any]:
        """Um ponto por planta, com ``tratar`` — abre no QGIS e alimenta o pulverizador.

        Sem coordenada em nenhuma planta, devolve uma coleção vazia em vez de
        inventar posição.
        """
        feicoes = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [planta.lon, planta.lat]},
                "properties": {
                    k: v for k, v in planta.to_dict().items() if k not in {"lat", "lon"}
                },
            }
            for planta in self.plantas
            if planta.lat is not None and planta.lon is not None
        ]
        return {
            "type": "FeatureCollection",
            "features": feicoes,
            "properties": {
                "field_id": self.field_id,
                "crop": self.crop,
                "pass_id": self.pass_id,
                "doenca": self.doenca,
                "produto": self.produto,
                "plantas_tratadas": self.plantas_tratadas,
                "litros": self.litros,
                "aviso": "Recomendação. Nenhuma aplicação ocorre sem aprovação do agrônomo.",
            },
        }


# ---------------------------------------------------------------------------------------------
# Núcleo
# ---------------------------------------------------------------------------------------------


def _posicoes(plantas: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas locais (ao longo da rua, atravessando as ruas), em metros.

    A rua entra pelo índice dela, não pelo rótulo: ``row_id`` é texto livre
    ('L12', 'norte-3'), e o que importa para a distância é a ordem.
    """
    ruas = sorted({p.row_id for p in plantas}, key=_ordem_da_rua)
    indice = {rua: i for i, rua in enumerate(ruas)}
    ao_longo = np.array([float(p.x_along_m) for p in plantas], dtype=float)
    atraves = np.array([float(indice[p.row_id]) for p in plantas], dtype=float)
    return ao_longo, atraves


def _ordem_da_rua(row_id: str) -> tuple[int, float, str]:
    try:
        return (0, float(row_id), row_id)
    except ValueError:
        return (1, 0.0, row_id)


def _nucleo(
    ao_longo: np.ndarray, atraves: np.ndarray, espacamento_entre_ruas_m: float, contagio: Contagio
) -> np.ndarray:
    """Peso de contágio entre cada par de plantas, com a diagonal zerada.

    Exponencial separável nos dois eixos: ``exp(-d_rua/a_rua) * exp(-d_entre/a_entre)``.
    Separável porque os dois caminhos de contágio são diferentes — contato de
    folhagem na fila, respingo e vento atravessando a entrelinha — e somar os
    dois num alcance só apagaria justamente a anisotropia que interessa.
    """
    d_longo = np.abs(ao_longo[:, None] - ao_longo[None, :])
    d_atraves = np.abs(atraves[:, None] - atraves[None, :]) * espacamento_entre_ruas_m
    k = np.exp(-d_longo / contagio.alcance_na_rua_m) * np.exp(
        -d_atraves / contagio.alcance_entre_ruas_m
    )
    np.fill_diagonal(k, 0.0)
    return k


def _risco(p: np.ndarray, nucleo: np.ndarray, sobra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risco de cada planta e a pressão que ela sofre, dado o tratamento já decidido.

    ``sobra`` é ``1`` na planta sem tratamento e ``1 - eficácia`` na tratada: é o
    que resta do inóculo dela e o que resta da suscetibilidade dela.

    A pressão é ``1 - exp(-soma_j K_ij * p_j * sobra_j)`` — a forma exponencial
    faz muitas vizinhas doentes saturarem em 1 em vez de estourar.
    """
    pressao = nucleo @ (p * sobra)
    q = 1.0 - np.exp(-pressao)
    return (p + (1.0 - p) * q) * sobra, pressao


def _doentes_esperados(p: np.ndarray, nucleo: np.ndarray, tratadas: np.ndarray, ef: float) -> float:
    sobra = np.where(tratadas, 1.0 - ef, 1.0)
    return float(_risco(p, nucleo, sobra)[0].sum())


def _escolher_gulosamente(
    p: np.ndarray,
    nucleo: np.ndarray,
    eficacia: float,
    perda: float,
    custo_planta: float,
    orcamento: float | None,
    max_passos: int,
) -> _Escolha:
    """Escolhe as plantas uma a uma, sempre a que mais reduz a perda do talhão.

    **Por que guloso e não um limiar.** O benefício de tratar um conjunto de
    plantas não é a soma dos benefícios individuais: tratar a vizinha já reduz a
    pressão sobre esta, e somar os dois contaria o mesmo contágio duas vezes.
    Uma versão anterior deste módulo multiplicava o benefício de cada planta pelo
    número de vizinhas que ela infectaria, e num talhão de 1.000 plantas com duas
    reboleiras mandava tratar as 1.000 — o contágio contado mil vezes. Aqui o
    ganho é **marginal**: recalculado a cada escolha, sobre o que já foi decidido.

    O ganho marginal de tratar ``i`` tem duas partes, as duas em dinheiro:

    * **direto** — a própria planta deixa de adoecer: ``ef * (p + (1-p) q)``;
    * **contágio** — ``i`` deixa de ser fonte, e a pressão cai em todas as
      outras: ``ef * p_i * soma_j K_ji * (1-p_j) * sobra_j * exp(-pressao_j)``,
      que é a derivada de ``1 - exp(-pressao)`` propagada pelo núcleo.

    Custo: uma multiplicação matriz-vetor por planta escolhida, ou seja O(n^2)
    por passo. Para os milhares de plantas de um talhão isso é rápido; é por isso
    que existe ``max_passos``.
    """
    n = len(p)
    tratadas = np.zeros(n, dtype=bool)
    escolhidas: list[int] = []
    # A curva comeca sem nenhum tratamento.
    curva = [_doentes_esperados(p, nucleo, tratadas, eficacia)]
    beneficio_direto = np.zeros(n)
    beneficio_contagio = np.zeros(n)
    gasto = 0.0
    parou_por_orcamento = False

    for passo in range(min(n, max_passos)):
        sobra = np.where(tratadas, 1.0 - eficacia, 1.0)
        risco, pressao = _risco(p, nucleo, sobra)
        q = 1.0 - np.exp(-pressao)

        direto = eficacia * (p + (1.0 - p) * q)
        peso = (1.0 - p) * sobra * np.exp(-pressao)
        contagio = eficacia * p * (nucleo.T @ peso)
        ganho = perda * (direto + contagio)
        ganho[tratadas] = -np.inf

        i = int(np.argmax(ganho))
        # Guarda o benefico de cada planta como ele era antes de qualquer escolha:
        # e o numero que explica a decisao ao agronomo.
        if passo == 0:
            beneficio_direto = perda * direto
            beneficio_contagio = perda * contagio
        if ganho[i] <= custo_planta:
            break
        if orcamento is not None and gasto + custo_planta > orcamento:
            parou_por_orcamento = True
            break
        tratadas[i] = True
        escolhidas.append(i)
        gasto += custo_planta
        curva.append(_doentes_esperados(p, nucleo, tratadas, eficacia))

    return _Escolha(
        tratadas, escolhidas, curva, beneficio_direto, beneficio_contagio, parou_por_orcamento
    )


@dataclass
class _Escolha:
    tratadas: np.ndarray
    escolhidas: list[int]
    curva: list[float]
    beneficio_direto: np.ndarray
    beneficio_contagio: np.ndarray
    parou_por_orcamento: bool


def prescrever(
    plantas: Iterable[Any],
    *,
    doenca: str,
    produto: Produto,
    economia: Economia,
    crop: str,
    field_id: str = "",
    pass_id: str | None = None,
    espacamento_entre_ruas_m: float,
    contagio: Contagio | None = None,
    dias_ate_colheita: int | None = None,
    pontos_na_curva: int = 12,
    max_passos: int = 5000,
) -> Prescricao:
    """Decide, planta a planta, se o tratamento se paga — e diz o que sobra de risco.

    ``plantas`` são :class:`~app.plant_fusion.PlantSummary` (ou qualquer objeto com
    ``plant_id``, ``row_id``, ``x_along_m``, ``posteriors``, ``lat`` e ``lon``).

    Levanta :class:`PrescricaoImpossivelError` quando a aplicação é proibida (bula,
    carência, registro) ou quando falta dado para decidir. Recusar é o
    comportamento certo: uma prescrição com número inventado é pior que nenhuma.
    """
    plantas = list(plantas)
    avisos: list[str] = []
    contagio = contagio or Contagio()

    _conferir_bula(produto, doenca, crop, dias_ate_colheita)
    if not plantas:
        raise PrescricaoImpossivelError("talhão sem plantas: rode uma passada antes de prescrever.")
    if espacamento_entre_ruas_m <= 0:
        raise PrescricaoImpossivelError("espaçamento entre ruas precisa ser positivo.")

    p = np.array([float(pl.posteriors.get(doenca, 0.0)) for pl in plantas], dtype=float)
    if not np.all(np.isfinite(p)):
        raise PrescricaoImpossivelError(f"posterior não finita para {doenca!r}.")
    p = np.clip(p, 0.0, 1.0)

    ao_longo, atraves = _posicoes(plantas)
    if not np.all(np.isfinite(ao_longo)):
        raise PrescricaoImpossivelError("planta sem posição ao longo da rua.")

    nucleo = _nucleo(ao_longo, atraves, espacamento_entre_ruas_m, contagio)
    n = len(plantas)
    ef = produto.eficacia
    custo_planta = produto.custo_por_planta + economia.custo_operacional_por_planta

    nenhum = np.zeros(n, dtype=bool)
    risco_sem_tratar = _risco(p, nucleo, np.ones(n))[0]
    escolha = _escolher_gulosamente(
        p, nucleo, ef, economia.perda_por_planta, custo_planta, economia.orcamento, max_passos
    )
    tratar = escolha.tratadas
    risco_final = _risco(p, nucleo, np.where(tratar, 1.0 - ef, 1.0))[0]
    posicao = {i: k for k, i in enumerate(escolha.escolhidas)}

    if not contagio.ajustado:
        avisos.append(
            "Alcance de contágio não ajustado a este talhão: os valores são o padrão do "
            "módulo, não medição. A borda em volta da reboleira depende deles."
        )
    if not tratar.any():
        avisos.append(
            "Nenhuma planta a tratar: o orçamento não cobre nem a primeira."
            if escolha.parou_por_orcamento
            else f"Nenhuma planta a tratar: tratar a pior planta do talhão pouparia "
            f"{float((escolha.beneficio_direto + escolha.beneficio_contagio).max()):.2f} "
            f"e custaria {custo_planta:.2f}."
        )
    if len(escolha.escolhidas) >= min(n, max_passos):
        avisos.append(
            f"A escolha parou no limite de {max_passos} plantas, não porque deixou de "
            "compensar: aumente max_passos para ver o resto."
        )

    prescritas = [
        PlantaPrescrita(
            plant_id=pl.plant_id,
            row_id=pl.row_id,
            x_along_m=float(pl.x_along_m),
            tratar=bool(tratar[i]),
            ordem=posicao.get(i),
            motivo=_motivo(
                bool(tratar[i]),
                float(p[i]),
                float(risco_sem_tratar[i]),
                float(escolha.beneficio_contagio[i]),
                float(escolha.beneficio_direto[i]),
            ),
            posterior=float(p[i]),
            risco_sem_nenhum_tratamento=float(risco_sem_tratar[i]),
            risco_final=float(risco_final[i]),
            beneficio_direto=float(escolha.beneficio_direto[i]),
            beneficio_contagio=float(escolha.beneficio_contagio[i]),
            valor=float(escolha.beneficio_direto[i] + escolha.beneficio_contagio[i] - custo_planta),
            lat=getattr(pl, "lat", None),
            lon=getattr(pl, "lon", None),
        )
        for i, pl in enumerate(plantas)
    ]

    n_tratadas = int(tratar.sum())
    litro_por_planta = produto.dose_ml_por_planta / 1000.0
    return Prescricao(
        field_id=field_id,
        crop=crop,
        pass_id=pass_id,
        doenca=doenca,
        produto=produto.nome,
        plantas=prescritas,
        plantas_no_talhao=n,
        plantas_tratadas=n_tratadas,
        litros=n_tratadas * litro_por_planta,
        custo=n_tratadas * custo_planta,
        litros_talhao_inteiro=n * litro_por_planta,
        custo_talhao_inteiro=n * custo_planta,
        doentes_sem_nenhum_tratamento=_doentes_esperados(p, nucleo, nenhum, ef),
        doentes_com_a_prescricao=float(risco_final.sum()),
        doentes_tratando_tudo=_doentes_esperados(p, nucleo, ~nenhum, ef),
        curva=_curva(escolha.curva, custo_planta, litro_por_planta, pontos_na_curva),
        avisos=avisos,
        limitado_por_orcamento=escolha.parou_por_orcamento,
        contagio_ajustado=contagio.ajustado,
    )


def _conferir_bula(produto: Produto, doenca: str, crop: str, dias_ate_colheita: int | None) -> None:
    """Recusa o que a bula ou a carência proíbem, antes de qualquer conta."""
    if crop not in produto.culturas:
        raise PrescricaoImpossivelError(
            f"{produto.nome!r} não é autorizado em {crop!r} (bula: {', '.join(produto.culturas)})."
        )
    if doenca not in produto.doencas:
        raise PrescricaoImpossivelError(
            f"{produto.nome!r} não tem {doenca!r} como alvo autorizado "
            f"(bula: {', '.join(produto.doencas)})."
        )
    fora_da_faixa = not (
        produto.dose_min_ml_por_planta
        <= produto.dose_ml_por_planta
        <= produto.dose_max_ml_por_planta
    )
    if fora_da_faixa:
        raise PrescricaoImpossivelError(
            f"dose de {produto.dose_ml_por_planta} mL/planta fora da faixa de bula "
            f"({produto.dose_min_ml_por_planta}-{produto.dose_max_ml_por_planta})."
        )
    if dias_ate_colheita is not None and dias_ate_colheita < produto.carencia_dias:
        raise PrescricaoImpossivelError(
            f"carência de {produto.carencia_dias} dias e colheita em {dias_ate_colheita}: "
            "aplicar agora deixaria resíduo acima do permitido na fruta colhida."
        )


def _curva(
    doentes_por_passo: Sequence[float],
    custo_planta: float,
    litro_por_planta: float,
    pontos: int,
) -> list[PontoDaCurva]:
    """Custo contra doença esperada, tratando as k primeiras plantas da ordem escolhida.

    É a curva que o agrônomo olha para decidir onde parar: cada ponto diz quanto
    ele gasta e quantas plantas doentes ainda espera no talhão depois disso.
    Ela é convexa por construção — o guloso pega primeiro o que rende mais —, e é
    por isso que ela achata: a partir de certo ponto, cada litro a mais compra
    muito pouco.
    """
    n = len(doentes_por_passo) - 1
    pontos = max(pontos, 2)
    ks = sorted({int(round(k)) for k in np.linspace(0, n, min(pontos, n + 1))})
    return [
        PontoDaCurva(
            plantas_tratadas=k,
            custo=k * custo_planta,
            litros=k * litro_por_planta,
            doentes_esperados=float(doentes_por_passo[k]),
        )
        for k in ks
    ]


def _motivo(tratar: bool, p: float, risco: float, contagio: float, direto: float) -> str:
    """Uma frase que o agrônomo lê sem consultar a documentação."""
    if tratar:
        if p >= 0.5:
            return f"doente (posterior {p:.2f})"
        if contagio > direto:
            return f"borda do foco: contém mais contágio ({contagio:.2f}) do que salva aqui"
        return f"risco {risco:.2f} acima do que o tratamento custa"
    if risco < 0.05:
        return "sadia e longe do foco"
    return f"risco {risco:.2f} não paga o tratamento"


def resumo_em_texto(prescricao: Prescricao, moeda: str = "R$") -> str:
    """O parágrafo que vai no relatório do produtor. Só números já calculados."""
    p = prescricao
    linhas = [
        f"Talhão {p.field_id or '-'} ({p.crop}), {p.doenca}, {p.produto}.",
        f"Tratar {p.plantas_tratadas} de {p.plantas_no_talhao} plantas: "
        f"{p.litros:.1f} L, {moeda} {p.custo:,.2f}.",
        f"Talhão inteiro seriam {p.litros_talhao_inteiro:.1f} L e "
        f"{moeda} {p.custo_talhao_inteiro:,.2f}.",
        f"Diferença: {p.litros_poupados:.1f} L e {moeda} {p.custo_evitado:,.2f} "
        f"({p.fracao_poupada:.0%}).",
        f"Doença esperada no talhão: {p.doentes_sem_nenhum_tratamento:.1f} plantas sem "
        f"fazer nada, {p.doentes_com_a_prescricao:.1f} com esta prescrição, "
        f"{p.doentes_tratando_tudo:.1f} pulverizando tudo.",
        f"Ou seja, {p.fracao_do_ganho_possivel:.0%} do ganho de pulverizar o talhão "
        f"inteiro, por {1 - p.fracao_poupada:.0%} do custo.",
    ]
    if p.limitado_por_orcamento:
        linhas.append("O orçamento cortou plantas em que o tratamento ainda se pagaria.")
    linhas.extend(f"Aviso: {a}" for a in p.avisos)
    linhas.append(
        "Recomendação: nenhuma aplicação ocorre sem aprovação do agrônomo e liberação "
        "da camada de segurança."
    )
    return "\n".join(linhas)
