"""Encaixe de contorno: o desenho torto do produtor vira o limite real do talhão.

O gesto é o do recorte inteligente de celular. A pessoa contorna a lavoura a
dedo, por cima, sem precisão nenhuma — e o contorno salta para a borda que
existe na imagem: onde o café acaba e começa a mata, o carreador, o pasto.

## Por que raio a partir do centro, e não segmentação livre

Porque o desenho **já diz** onde é dentro e onde é fora, e isso muda o problema.

Uma segmentação livre precisa descobrir o objeto no meio de tudo, erra para
regiões vizinhas parecidas e exige um modelo treinado para saber o que é
"lavoura". Aqui não: o interior do desenho é a amostra do que a pessoa quer, o
lado de fora é a amostra do que ela não quer, e a resposta é **quão longe do
centro** a semelhança acaba, em cada direção.

Isso traz três coisas que importam mais que sofisticação:

* **fica perto do que foi desenhado.** O raio refinado é limitado a uma tolerância
  do raio desenhado, então um contorno torto vira um contorno certo — nunca a
  fazenda inteira nem o vizinho;
* **não inventa buraco no meio.** O resultado é sempre um polígono simples e
  fechado, que é o que o resto da plataforma sabe usar (cerca virtual, cobertura,
  planejamento);
* **funciona sem treino**, em lavoura que nenhum dataset viu.

A suposição é que o talhão é *estrelado* a partir do centro — que toda direção
sai do centro e cruza a borda uma vez só. Vale para talhão comum e vale sempre
para "desenhei um círculo em volta". Não vale para um talhão em U muito fechado,
e :func:`refine_boundary` diz quando desconfia disso.

## O que isto não é

Não é modelo treinado, e não se anuncia como tal. É medida de semelhança de cor
e textura com o interior que a própria pessoa marcou. Onde a lavoura e o vizinho
têm a mesma cara, o encaixe erra — e por isso o resultado volta com
``confianca`` por direção e a pessoa continua podendo arrastar o vértice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RAIOS = 360
"""Direções analisadas, uma por grau. Mais que isso não melhora: o passo radial
já é de um pixel."""

TOLERANCIA_PADRAO = 0.35
"""Quanto o contorno pode se afastar do desenho, como fração do raio desenhado.

35% é generoso para corrigir mão trêmula e continua impedindo que o encaixe
fuja para o talhão vizinho. Quem desenhou grosseiro de propósito aumenta; quem
desenhou com cuidado diminui e trava o resultado perto do traço.
"""

SUAVIZACAO = 9
"""Janela da mediana circular sobre o perfil de raios, em graus.

Uma árvore isolada dentro do talhão faz um raio despencar sozinho. A mediana
tira o espinho sem arredondar o canto — que é o que uma média faria.
"""


@dataclass(frozen=True, slots=True)
class Contorno:
    """O contorno encaixado e o quanto se pode confiar nele."""

    pontos: np.ndarray
    """``(n, 2)`` em pixels, ``(x, y)``, na mesma referência da imagem de entrada."""

    confianca: float
    """Fração das direções em que a borda foi encontrada com transição nítida."""

    area_px: float
    area_desenhada_px: float
    encostou_no_limite: float
    """Fração das direções que pararam na tolerância em vez de numa borda.

    Alto quer dizer que o desenho apertou demais a lavoura (ou que não há borda
    ali). É o sinal para a pessoa refazer o traço mais folgado.
    """

    @property
    def variacao(self) -> float:
        """Quanto a área mudou. Perto de zero: o desenho já estava bom."""
        if self.area_desenhada_px <= 0:
            return 0.0
        return self.area_px / self.area_desenhada_px - 1.0

    @property
    def descricao(self) -> str:
        return (
            f"{len(self.pontos)} vértices, área {self.variacao:+.0%} em relação ao "
            f"desenho, borda nítida em {self.confianca:.0%} das direções"
        )


def _caracteristicas(imagem: np.ndarray) -> np.ndarray:
    """Três números por pixel: verde, brilho e aspereza.

    Verde separa vegetação de solo; brilho separa mata escura de café claro; a
    aspereza — variação local — separa copa grande e irregular de lavoura em
    linha. Juntos dão conta das quatro vizinhanças que cercam um talhão de café:
    mata, carreador, pasto e outra lavoura.
    """
    rgb = imagem.astype(np.float32)
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    brilho = rgb.mean(axis=2)

    # Aspereza: diferença para a média dos vizinhos, sem convolução — só somas
    # deslocadas, que é barato e não traz dependência nova.
    suave = (
        np.pad(brilho, 1, mode="edge")[:-2, 1:-1]
        + np.pad(brilho, 1, mode="edge")[2:, 1:-1]
        + np.pad(brilho, 1, mode="edge")[1:-1, :-2]
        + np.pad(brilho, 1, mode="edge")[1:-1, 2:]
    ) / 4.0
    aspereza = np.abs(brilho - suave)

    pilha = np.dstack([exg, brilho, aspereza])
    # Normaliza cada canal pela própria dispersão, para os três pesarem igual.
    mediana = np.median(pilha.reshape(-1, 3), axis=0)
    escala = np.median(np.abs(pilha.reshape(-1, 3) - mediana), axis=0) * 1.4826
    return (pilha - mediana) / np.maximum(escala, 1e-6)


def _centro(poligono: np.ndarray) -> tuple[float, float]:
    """Centroide da área, não dos vértices.

    A média dos vértices puxa para o lado onde a pessoa clicou mais vezes, e um
    centro deslocado joga todos os raios para o mesmo lado.
    """
    x, y = poligono[:, 0], poligono[:, 1]
    cruz = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cruz.sum() / 2.0
    if abs(area) < 1e-9:
        return float(x.mean()), float(y.mean())
    cx = ((x + np.roll(x, -1)) * cruz).sum() / (6 * area)
    cy = ((y + np.roll(y, -1)) * cruz).sum() / (6 * area)
    return float(cx), float(cy)


def _raio_desenhado(poligono: np.ndarray, centro: tuple[float, float], angulos: np.ndarray) -> np.ndarray:
    """Distância do centro até o traço da pessoa, em cada direção."""
    cx, cy = centro
    vx, vy = poligono[:, 0] - cx, poligono[:, 1] - cy
    ang_v = np.arctan2(vy, vx) % (2 * math.pi)
    raio_v = np.hypot(vx, vy)

    ordem = np.argsort(ang_v)
    ang_ord, raio_ord = ang_v[ordem], raio_v[ordem]
    # Fecha o círculo para interpolar entre o último e o primeiro vértice.
    ang_ciclo = np.concatenate([ang_ord - 2 * math.pi, ang_ord, ang_ord + 2 * math.pi])
    raio_ciclo = np.concatenate([raio_ord, raio_ord, raio_ord])
    return np.interp(angulos, ang_ciclo, raio_ciclo)


def _amostra(caracteristicas: np.ndarray, centro: tuple[float, float],
             angulos: np.ndarray, raios: np.ndarray, fracao: float) -> np.ndarray:
    """Valores das características a uma fração do raio, em cada direção."""
    cx, cy = centro
    altura, largura = caracteristicas.shape[:2]
    xs = np.clip((cx + np.cos(angulos) * raios * fracao).astype(int), 0, largura - 1)
    ys = np.clip((cy + np.sin(angulos) * raios * fracao).astype(int), 0, altura - 1)
    return caracteristicas[ys, xs]


def _mediana_circular(valores: np.ndarray, janela: int) -> np.ndarray:
    """Mediana deslizante que dá a volta: o grau 359 é vizinho do grau 0."""
    metade = janela // 2
    esticado = np.concatenate([valores[-metade:], valores, valores[:metade]])
    vistas = np.lib.stride_tricks.sliding_window_view(esticado, janela)
    return np.median(vistas, axis=-1)


def refine_boundary(
    imagem: np.ndarray,
    poligono: np.ndarray | list,
    *,
    tolerancia: float = TOLERANCIA_PADRAO,
    vertices: int = 48,
) -> Contorno:
    """Encaixa um contorno desenhado a mão no limite que existe na imagem.

    ``poligono`` são os pontos ``(x, y)`` em pixels que a pessoa marcou; três no
    mínimo, e podem estar bem tortos. ``tolerancia`` limita o quanto o resultado
    pode se afastar do traço, como fração do raio.

    Como funciona, em uma frase: o interior do desenho vira a amostra do que a
    pessoa quer, e cada direção é percorrida do centro para fora até a
    semelhança com essa amostra acabar.
    """
    pontos = np.asarray(poligono, dtype=np.float64)
    if pontos.ndim != 2 or pontos.shape[1] != 2 or len(pontos) < 3:
        raise ValueError("O polígono precisa de ao menos três pontos (x, y).")
    if imagem.ndim != 3 or imagem.shape[2] < 3:
        raise ValueError("A imagem precisa ser RGB com três canais.")

    caracteristicas = _caracteristicas(imagem)
    centro = _centro(pontos)
    angulos = np.linspace(0, 2 * math.pi, RAIOS, endpoint=False)
    raio_desenhado = _raio_desenhado(pontos, centro, angulos)
    if raio_desenhado.max() < 4:
        raise ValueError("O desenho é pequeno demais para encaixar.")

    # Referência do interior: o miolo do desenho, longe da borda incerta.
    miolo = np.vstack(
        [_amostra(caracteristicas, centro, angulos, raio_desenhado, f) for f in (0.25, 0.4, 0.55)]
    )
    referencia = np.median(miolo, axis=0)
    dispersao = np.maximum(np.median(np.abs(miolo - referencia), axis=0) * 1.4826, 0.35)
    limiar = 2.5  # desvios robustos: além disso, já não se parece com o interior

    raio_min = raio_desenhado * (1.0 - tolerancia)
    raio_max = raio_desenhado * (1.0 + tolerancia)
    passos = int(np.ceil(raio_max.max()))

    # Para cada direção, caminha do centro para fora e anota onde a semelhança
    # deixa de valer. Vetorizado por passo: todas as direções andam juntas.
    fim = raio_max.copy()
    nitido = np.zeros(RAIOS, dtype=bool)
    ainda_dentro = np.ones(RAIOS, dtype=bool)
    fora_seguidos = np.zeros(RAIOS, dtype=int)
    PACIENCIA = 3  # tolera falha isolada: rua de terra dentro do talhão

    for passo in range(1, passos + 1):
        ativos = ainda_dentro & (passo <= raio_max)
        if not ativos.any():
            break
        raio_atual = np.full(RAIOS, float(passo))
        valores = _amostra(caracteristicas, centro, angulos, raio_atual, 1.0)
        distancia = np.abs(valores - referencia) / dispersao
        parecido = distancia.max(axis=1) < limiar

        fora_seguidos = np.where(parecido, 0, fora_seguidos + 1)
        acabou = ativos & (fora_seguidos >= PACIENCIA) & (raio_atual >= raio_min)
        fim = np.where(acabou, passo - PACIENCIA, fim)
        nitido |= acabou
        ainda_dentro &= ~acabou

    raio_final = _mediana_circular(np.clip(fim, raio_min, raio_max), SUAVIZACAO)

    # Reduz para um número de vértices que uma pessoa consiga arrastar depois.
    #
    # Índices espaçados em vez de fatia com passo: uma fatia devolve
    # ceil(RAIOS/passo) pontos, que raramente é o número pedido — 32 vértices
    # viravam 33. Quem chama a rota pede uma contagem e recebe essa contagem.
    quantos = min(RAIOS, max(8, vertices))
    indices = np.arange(quantos) * RAIOS // quantos
    ang_saida = angulos[indices]
    raio_saida = raio_final[indices]
    cx, cy = centro
    saida = np.column_stack(
        [cx + np.cos(ang_saida) * raio_saida, cy + np.sin(ang_saida) * raio_saida]
    )

    encostou = float((~nitido).mean())
    return Contorno(
        pontos=saida,
        confianca=float(nitido.mean()),
        area_px=_area(saida),
        area_desenhada_px=_area(pontos),
        encostou_no_limite=encostou,
    )


def _area(poligono: np.ndarray) -> float:
    x, y = poligono[:, 0], poligono[:, 1]
    return float(abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0)
