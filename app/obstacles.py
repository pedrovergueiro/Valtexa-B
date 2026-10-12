"""Objetos dentro do talhão: o que o rover tem que desviar, e por onde ele anda.

A cobertura em blocos (:func:`app.terrain.detect_cover`) diz *o que* há em cada
pedaço. Isto aqui junta os pedaços vizinhos e responde a pergunta que decide a
rota: **onde**, em coordenada, está cada coisa — a moita de mata que entrou na
lavoura, a árvore isolada no meio da rua, o carreador que corta o talhão.

## Por que isto muda o planejamento, e não é enfeite

Sem obstáculo, o planejador trata o talhão como uma superfície lisa e manda o
rover em vaivém de ponta a ponta. Com os objetos localizados, três coisas mudam:

* a **área operável** encolhe para o que é lavoura de verdade, e o tempo de
  missão para de mentir para mais;
* o rover pode **desviar** em vez de esbarrar;
* o **carreador** deixa de ser buraco na cobertura e vira caminho — é por ele
  que o robô atravessa o talhão sem pisar em planta.

## O que é medida e o que é palpite

A geometria é exata: área, centro e forma de cada agrupamento saem da grade de
blocos, e a conversão para metros é aritmética.

A **classe** de cada objeto é palpite declarado. Sai da mesma regra de cor e
textura da cobertura, mais a forma do agrupamento: comprido e de solo exposto é
carreador; compacto e escuro é mata. Toda resposta carrega ``heuristica=True`` e
uma ``confianca`` — e nenhuma delas vira cerca virtual sozinha. Quem confirma é
o agrônomo, e é a confirmação dele que forma o dataset de um segmentador
treinado.

## O que isto ainda não separa

Poste, pedra solta, formigueiro e barranco caem todos em ``desconhecido``: são
pequenos demais para a grade e não têm assinatura de cor própria. Aparecem como
objeto — o que já é melhor do que não aparecer —, mas sem nome. Resolver isso
exige resolução de ortomosaico e um detector treinado, não regra de cor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from app.terrain import (
    LAVOURA,
    MATA,
    OUTRO,
    SOLO,
    Cobertura,
    bloco_para,
    detect_cover,
    detect_rows,
)

MATA_INTERNA = "mata"
CARREADOR = "carreador"
SOLO_EXPOSTO = "solo_exposto"
DESCONHECIDO = "desconhecido"

TIPOS = (MATA_INTERNA, CARREADOR, SOLO_EXPOSTO, DESCONHECIDO)

BLOCOS_MINIMOS = 3
"""Agrupamentos menores que isto são descartados.

Um bloco isolado é quase sempre sombra de nuvem, falha de plantio ou ruído da
classificação. Reportá-lo encheria o mapa de objetos que não existem, e um mapa
cheio de falso positivo é ignorado inteiro — inclusive nos pontos certos.
"""

ALONGAMENTO_CARREADOR = 2.5
"""Razão comprimento/largura a partir da qual um agrupamento de solo é caminho.

Carreador é faixa: comprido e estreito. Um terreiro de café, ou um talhão
colhido, é solo exposto também, mas largo — e não serve de caminho para
atravessar a lavoura.
"""


@dataclass(frozen=True, slots=True)
class Objeto:
    """Uma coisa dentro do talhão, localizada."""

    tipo: str
    centro_px: tuple[float, float]
    area_px: float
    largura_px: float
    comprimento_px: float
    blocos: int
    confianca: float
    """Fração dos blocos do agrupamento que concordaram com a classe dominante."""

    contorno_px: list[tuple[float, float]]
    """Retângulo do agrupamento. Grosseiro de propósito: a grade é de blocos, e
    desenhar um contorno fino sobre ela sugeriria uma precisão que não existe."""

    @property
    def alongamento(self) -> float:
        if self.largura_px <= 0:
            return 1.0
        return self.comprimento_px / self.largura_px

    @property
    def atravessavel(self) -> bool:
        """Se o rover pode passar por cima em vez de contornar."""
        return self.tipo in (CARREADOR, SOLO_EXPOSTO)


@dataclass(frozen=True, slots=True)
class Terreno:
    """O talhão repartido: quanto dá para operar e o que atrapalha."""

    cobertura: Cobertura
    objetos: list[Objeto]
    fracao_lavoura: float
    fracao_obstaculo: float
    fracao_caminho: float
    periodo_px: float | None = None
    """Espaçamento das ruas medido na imagem. Nulo: não foi possível medir."""

    resolucao_suficiente: bool = True
    """Se a imagem tem pixel bastante para a leitura valer.

    Falso zera as frações e a lista de objetos. É deliberado: numa imagem em que
    a rua não se resolve, **nada** é lavoura para o classificador, e o resultado
    seria "75% de obstáculo" num talhão que é cafezal inteiro. Esse número
    encolheria a área operável pela metade e o planejamento sairia errado — pior
    do que não responder.
    """

    heuristica: bool = True

    @property
    def descricao(self) -> str:
        if not self.resolucao_suficiente:
            return (
                "Resolução insuficiente para ler o terreno: a rua de plantio não "
                "aparece nesta imagem. Use ortomosaico de drone (3 a 10 cm por pixel)."
            )
        bloqueios = sum(1 for o in self.objetos if not o.atravessavel)
        caminhos = sum(1 for o in self.objetos if o.atravessavel)
        return (
            f"{self.fracao_lavoura:.0%} lavoura, {self.fracao_obstaculo:.0%} obstáculo · "
            f"{bloqueios} bloqueio(s) e {caminhos} caminho(s) localizados"
        )


def _dentro(poligono_px: np.ndarray, x: float, y: float) -> bool:
    """Ponto no polígono, por cruzamentos. Borda conta como dentro."""
    n = len(poligono_px)
    dentro = False
    j = n - 1
    for i in range(n):
        xi, yi = poligono_px[i]
        xj, yj = poligono_px[j]
        if (yi > y) != (yj > y):
            corte = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < corte:
                dentro = not dentro
        j = i
    return dentro


def _agrupa(mascara: np.ndarray) -> list[list[tuple[int, int]]]:
    """Agrupamentos de células vizinhas ligadas, em quatro direções.

    Quatro e não oito: dois blocos que se tocam só na quina não são o mesmo
    obstáculo, são dois — e uni-los engordaria a mata por cima da lavoura.
    """
    nl, nc = mascara.shape
    visto = np.zeros_like(mascara, dtype=bool)
    grupos: list[list[tuple[int, int]]] = []
    for i0 in range(nl):
        for j0 in range(nc):
            if not mascara[i0, j0] or visto[i0, j0]:
                continue
            fila = deque([(i0, j0)])
            visto[i0, j0] = True
            grupo: list[tuple[int, int]] = []
            while fila:
                i, j = fila.popleft()
                grupo.append((i, j))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < nl and 0 <= b < nc and mascara[a, b] and not visto[a, b]:
                        visto[a, b] = True
                        fila.append((a, b))
            grupos.append(grupo)
    return grupos


def _classifica(grupo: list[tuple[int, int]], classes: np.ndarray) -> tuple[str, float]:
    """Tipo do agrupamento pela classe dominante e pela forma."""
    rotulos = [classes[i, j] for i, j in grupo]
    dominante = max(set(rotulos), key=rotulos.count)
    concordancia = rotulos.count(dominante) / len(rotulos)

    linhas = [i for i, _ in grupo]
    colunas = [j for _, j in grupo]
    altura = max(linhas) - min(linhas) + 1
    largura = max(colunas) - min(colunas) + 1
    comprido = max(altura, largura) / max(1, min(altura, largura))

    if dominante == MATA:
        return MATA_INTERNA, concordancia
    if dominante == SOLO:
        return (CARREADOR if comprido >= ALONGAMENTO_CARREADOR else SOLO_EXPOSTO), concordancia
    return DESCONHECIDO, concordancia


def detect_obstacles(
    imagem: np.ndarray,
    poligono_px: np.ndarray | list | None = None,
    *,
    bloco_px: int = 48,
    blocos_minimos: int = BLOCOS_MINIMOS,
) -> Terreno:
    """Classifica o terreno e localiza o que não é lavoura dentro do talhão.

    ``poligono_px`` limita a análise ao talhão; sem ele, a imagem inteira é
    considerada. Passar o contorno importa: sem ele, a mata **de fora** do
    talhão entra na conta e a área operável sai errada para menos.

    ``bloco_px`` é só o palpite inicial: o valor real sai do espaçamento das
    ruas medido na própria imagem.
    """
    # Mede a rua ANTES de repartir o terreno. O tamanho do bloco sai daí: um
    # bloco precisa conter vários períodos para a análise de frequência decidir
    # alguma coisa, e sem isso a lavoura inteira cai em "não sei".
    ruas = detect_rows(imagem)
    if not ruas.confiavel:
        vazia = detect_cover(imagem, bloco_px=bloco_px)
        return Terreno(
            cobertura=vazia,
            objetos=[],
            fracao_lavoura=0.0,
            fracao_obstaculo=0.0,
            fracao_caminho=0.0,
            periodo_px=None,
            resolucao_suficiente=False,
        )

    bloco_px = bloco_para(ruas.periodo_px)
    cobertura = detect_cover(imagem, bloco_px=bloco_px, periodo_px=ruas.periodo_px)
    classes = cobertura.classes
    passo = cobertura.passo_px or bloco_px
    nl, nc = classes.shape

    # Máscara do que está dentro do talhão, avaliada no centro de cada bloco.
    if poligono_px is not None:
        anel = np.asarray(poligono_px, dtype=np.float64)
        dentro = np.zeros((nl, nc), dtype=bool)
        for i in range(nl):
            for j in range(nc):
                cx = j * passo + bloco_px / 2
                cy = i * passo + bloco_px / 2
                dentro[i, j] = _dentro(anel, cx, cy)
    else:
        dentro = np.ones((nl, nc), dtype=bool)

    total = int(dentro.sum())
    if total == 0:
        raise ValueError("O contorno não cobre bloco nenhum da imagem.")

    def fracao(*nomes: str) -> float:
        return sum(float(((classes == nome) & dentro).sum()) for nome in nomes) / total

    nao_lavoura = dentro & (classes != LAVOURA)
    objetos: list[Objeto] = []
    for grupo in _agrupa(nao_lavoura):
        if len(grupo) < blocos_minimos:
            continue
        tipo, confianca = _classifica(grupo, classes)
        linhas = [i for i, _ in grupo]
        colunas = [j for _, j in grupo]
        i0, i1 = min(linhas), max(linhas) + 1
        j0, j1 = min(colunas), max(colunas) + 1
        altura_px = (i1 - i0 - 1) * passo + bloco_px
        largura_px = (j1 - j0 - 1) * passo + bloco_px
        objetos.append(
            Objeto(
                tipo=tipo,
                centro_px=(
                    sum(colunas) / len(colunas) * passo + bloco_px / 2,
                    sum(linhas) / len(linhas) * passo + bloco_px / 2,
                ),
                # A área conta o passo, não a janela: janelas vizinhas se
                # sobrepõem, e somar a janela inteira contaria o mesmo chão
                # quatro vezes.
                area_px=len(grupo) * passo * passo,
                largura_px=float(min(altura_px, largura_px)),
                comprimento_px=float(max(altura_px, largura_px)),
                blocos=len(grupo),
                confianca=round(confianca, 3),
                contorno_px=[
                    (j0 * passo, i0 * passo),
                    (j0 * passo + largura_px, i0 * passo),
                    (j0 * passo + largura_px, i0 * passo + altura_px),
                    (j0 * passo, i0 * passo + altura_px),
                ],
            )
        )

    objetos.sort(key=lambda o: -o.area_px)
    return Terreno(
        cobertura=cobertura,
        objetos=objetos,
        fracao_lavoura=round(fracao(LAVOURA), 4),
        fracao_obstaculo=round(fracao(MATA, OUTRO), 4),
        fracao_caminho=round(fracao(SOLO), 4),
        periodo_px=ruas.periodo_px,
    )


def area_operavel_ha(terreno: Terreno, area_total_ha: float) -> float:
    """Quanto do talhão o rover realmente percorre.

    Lavoura mais caminho: o carreador não tem planta, mas é por onde o robô
    atravessa, e tirá-lo da conta subestimaria o tempo de missão.
    """
    return round(area_total_ha * (terreno.fracao_lavoura + terreno.fracao_caminho), 4)

