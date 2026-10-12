"""Leitura do terreno numa imagem aérea: ruas, cobertura e obstáculos.

Responde, sobre uma foto de cima, três perguntas que decidem se um talhão é
operável: **para onde correm as ruas**, **qual é o espaçamento entre elas** e
**onde o rover não passa**.

## Por que isto não é um classificador treinado

Porque não precisa ser, e fingir que é seria pior.

Uma lavoura em linha é um **sinal periódico**: faixas claras e escuras que se
repetem a cada 3,5 m. Achar a direção e o período de um padrão periódico é
análise de frequência, não aprendizado de máquina — e tem três vantagens
decisivas sobre um modelo aqui:

* funciona em lavoura que nenhum dataset viu, inclusive na primeira fazenda de
  um cliente novo;
* devolve **força do sinal**, um número que diz o quanto a resposta vale, em vez
  de uma confiança de softmax que não sabe o que não sabe;
* é auditável: dá para plotar o perfil projetado e ver a onda que gerou a
  resposta.

A separação entre lavoura, mata e solo exposto é heurística de cor e textura —
declarada como heurística. Café em linha tem textura periódica; mata tem textura
forte e **sem** período; solo exposto tem pouco verde. Isso distingue os três
casos que importam para a rota, e o agrônomo corrige o que ficou errado. É a
correção dele que vai formar o dataset de um segmentador treinado, quando houver
amostra suficiente — o mesmo caminho do ``aerial-canopy``.

## O que este módulo NÃO faz

Não georreferencia. Entra imagem em pixels e sai resposta em pixels; converter
para metros e para azimute verdadeiro exige a escala e a rotação da imagem, que
vêm do ortomosaico ou de dois pontos conhecidos que alguém marcou. Quem faz essa
conversão é quem chama, e :func:`detect_rows` aceita os parâmetros para isso.
Sem eles, a resposta sai em pixels e diz que está em pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

JANELA_MINIMA = 64
"""Lado mínimo do recorte analisado. Abaixo disso não há resolução para o padrão."""

BINS_ANGULO = 180
"""Resolução angular da busca: 1° por passo. Meia volta basta — uma rua não tem
sentido, só direção."""

BINS_PERFIL = 1024
"""Amostras do perfil projetado.

Grande de propósito. Numa foto de talhão inteiro, a rua de café tem **poucos
pixels** de período: 3,5 m a meio metro por pixel são sete. Com um perfil de 256
amostras sobre 1.400 px, cada amostra cobriria cinco pixels e o padrão sumiria
por amostragem abaixo de Nyquist.

Foi exatamente esse o erro da primeira versão deste módulo: ela reduzia a imagem
para 512 px, encontrava período de 800 px — que é o relevo — e chamava de rua,
com força alta, inclusive dentro da mata.
"""

PERIODO_MIN_PX = 10.0
PERIODO_MAX_PX = 120.0
"""Faixa de período aceita, em pixels.

Restringir a busca é o que separa lavoura de artefato. O piso é 10 px porque
abaixo disso mora o quadriculado de 8 px do JPEG e o aliás da reamostragem: numa
foto comprimida eles aparecem como padrão fortíssimo em 0, 45 e 90 graus
exatos, e a versão anterior os reportava como ruas de 4,3 px — inclusive dentro
da mata. O teto é 120 px porque acima disso não é mais espaçamento de rua e sim
a forma do terreno, a sombra da mata ou o gradiente de iluminação.

**Consequência prática:** a imagem precisa ter a rua com pelo menos 10 px. Para
café a 3,5 m isso exige 35 cm por pixel ou melhor — um ortomosaico de drone
resolve com folga (3 a 10 cm/px); um print de tela ou uma foto de rede social,
não. Quando não dá, o detector responde que não dá.

Quem conhece a escala da imagem passa a faixa e acerta mais — ver
``periodo_min_px`` e ``periodo_max_px`` em :func:`detect_rows`.
"""

RUIDO_FINO_MAXIMO = 0.15
"""Quanto o espectro **acima** da faixa pode se destacar, em relação ao pico de
dentro dela, antes de o pico de dentro ser considerado artefato.

É o portão que separa "rua de 24 px" de "rua de 6 px que esta imagem não
resolve". Uma trama fina demais não desaparece do espectro: ela bate com a
grade de pixels e reaparece como um pico largo e estável em período grande —
um *moiré*. Esse pico é forte, é coerente e sobrevive à redução de escala, de
modo que nem a força nem a consistência entre escalas o denunciam. O que o
denuncia é a energia que ficou lá em cima, no período verdadeiro.

Medido sobre o mesmo corpus da :data:`FORCA_MINIMA`:

* **lavoura de verdade**, de 12 a 40 px de rua: razão de 0,01 a 0,09. O que
  aparece acima da faixa são os harmônicos da própria rua, e harmônico é
  fraco — um oitavo do fundamental, no pior caso medido;
* **trama fina demais**: rua de 6 px dá 0,26; de 8 px, 0,58; de 4 px, 1,35;
* **gradiente de iluminação em 8 bits**: 6,76. A escada de quantização é fina e
  periódica, e era ela o negativo mais próximo do limiar de força — 23,7 contra
  os 56,6 do pior positivo. Aqui ela é rejeitada por um fator de 45.

0,15 fica entre 0,09 e 0,26. É o portão que tornou possível baixar
:data:`FORCA_MINIMA` sem abrir a porta para o falso positivo: sem ele, os dois
lados se encostavam e não havia limiar de força que separasse os dois.
"""

FORCA_MINIMA = 36.0
"""Destaque sobre o fundo local a partir do qual o padrão é aceito como linha.

Um padrão só é aceito como rua quando passa **nos dois** portões: força acima
deste número **e** ruído fino abaixo de :data:`RUIDO_FINO_MAXIMO`. Eles medem
coisas diferentes, e é por isso que são dois — a força diz o quanto o pico se
destaca, e o ruído fino diz se ele é mesmo o fundamental ou o eco de uma trama
que esta imagem não resolve.

Medido sobre 13 positivos e 14 negativos:

* **positivos** (lavoura de 12 a 90 px de rua, com ângulo conhecido): de 53,4 a
  611. O piso é o cafezal cortado por um carreador de 240 px — um terço da
  imagem sem rua nenhuma dilui o sinal periódico. Lavoura limpa passa de 500;
  com ruído de σ=40 no pixel, fica em 114;
* **negativos** (ruído, gradiente, mata sintética de escala 8 a 64 px, pasto,
  solo nu, JPEG blocado, e tramas de 4, 6 e 8 px): de 4,4 a 37,6.

Sozinha, a força **não** separaria os dois: o pior negativo (37,6, a trama de
6 px vista como moiré de 21,7 px) fica acima de boa parte da margem, e não há
limiar entre 37,6 e 53,4 que dê folga para os dois lados. Quem separa é o portão
do ruído fino, que barra os 14 negativos — inclusive esse.

36 fica logo abaixo do pior positivo e segue sendo útil como segunda trava: se
o portão do ruído fino falhar num caso que este corpus não tem, a força ainda
recusa 13 dos 14 negativos por conta própria. É defesa em profundidade, não
redundância.

O erro caro é o falso positivo: dizer "rua" para mata manda o rover para dentro
dela. Recusar devolve "não sei", que é seguro e é o que o console mostra.

Os números saem de ``tests/test_terrain.py`` e ``tests/test_obstacles.py``, que
seguram os dois lados. Mexer aqui sem refazer a medida é apostar.
"""


@dataclass(frozen=True, slots=True)
class Ruas:
    """O que a imagem diz sobre as linhas de plantio."""

    angulo_deg: float
    """Direção das ruas na imagem, 0 = horizontal, sentido anti-horário."""

    periodo_px: float
    """Distância entre ruas vizinhas, em pixels."""

    forca: float
    """Quantas vezes o pico se levanta acima do fundo local do espectro."""

    confiavel: bool
    espacamento_m: float | None = None
    azimute_deg: float | None = None
    """Direção em graus a partir do norte. Só existe com a rotação da imagem."""

    @property
    def descricao(self) -> str:
        if not self.confiavel:
            return (
                f"Sem padrão de linha detectável (força {self.forca:.1f}). "
                f"Pode ser mata, pasto, lavoura sem linha ou resolução insuficiente."
            )
        espaco = (
            f"{self.espacamento_m:.2f} m" if self.espacamento_m else f"{self.periodo_px:.1f} px"
        )
        direcao = (
            f"azimute {self.azimute_deg:.1f} graus"
            if self.azimute_deg is not None
            else f"{self.angulo_deg:.1f} graus na imagem"
        )
        return f"Ruas a cada {espaco}, {direcao} (força {self.forca:.1f})"


def _verde(imagem: np.ndarray) -> np.ndarray:
    """Índice de excesso de verde, normalizado.

    ``2G - R - B`` é o índice clássico de vegetação para câmera RGB comum, que é
    o que um drone de operador terceirizado carrega. Sem banda infravermelha não
    há NDVI, e inventar um a partir do RGB daria um número com nome de medida e
    conteúdo de chute.
    """
    rgb = imagem.astype(np.float32)
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    faixa = np.percentile(exg, [2, 98])
    if faixa[1] - faixa[0] < 1e-6:
        return np.zeros_like(exg)
    return np.clip((exg - faixa[0]) / (faixa[1] - faixa[0]), 0.0, 1.0)


def _realce(campo: np.ndarray, janela: int) -> np.ndarray:
    """Tira a escala grande e deixa só o que varia no tamanho da rua.

    Subtrai a média local numa janela de algumas ruas. O que é uniforme —
    carreador, terreiro, mata fechada, gradiente de iluminação — some; o que se
    repete no espaçamento das ruas fica.

    Existe porque uma faixa larga de terra atravessando o talhão afogava o sinal
    periódico e o detector respondia "sem padrão" num cafezal inteiro. O
    carreador é parte da lavoura real, não exceção.

    A média local sai de soma acumulada: é O(n) e não traz dependência nova.
    """
    janela = max(3, int(janela) | 1)
    metade = janela // 2

    def media_1d(x: np.ndarray, eixo: int) -> np.ndarray:
        # A soma acumulada de N+janela-1 amostras dá N-1 diferenças de janela.
        # O zero na frente devolve a amostra que falta: com ele, soma[i] vira
        # soma[i-1+janela] - soma[i-1], e a saída tem exatamente o tamanho da
        # entrada. Sem esse zero o filtro encolhia a imagem um pixel por eixo.
        pad = [(0, 0), (0, 0)]
        pad[eixo] = (metade, metade)
        esticado = np.pad(x, pad, mode="edge")
        acumulado = np.cumsum(esticado, axis=eixo)
        zero = [(0, 0), (0, 0)]
        zero[eixo] = (1, 0)
        acumulado = np.pad(acumulado, zero, mode="constant")
        fim = [slice(None), slice(None)]
        comeco = [slice(None), slice(None)]
        fim[eixo] = slice(janela, None)
        comeco[eixo] = slice(0, -janela)
        return (acumulado[tuple(fim)] - acumulado[tuple(comeco)]) / janela

    return campo - media_1d(media_1d(campo, 0), 1)


def _perfil(campo: np.ndarray, angulo_rad: float, bins: int = BINS_PERFIL) -> np.ndarray:
    """Média do campo projetada sobre o eixo perpendicular a ``angulo_rad``.

    Somar ao longo das ruas e variar atravessando-as: se as linhas estiverem
    nesse ângulo, o perfil vira uma onda com o período do espaçamento; em
    qualquer outro ângulo a soma borra e a onda some. É a ideia da transformada
    de Radon, que é como se acha linha em imagem desde antes de existir rede
    neural — e aqui vale mais, porque devolve o **porquê**: dá para plotar o
    perfil e ver a onda.
    """
    altura, largura = campo.shape
    ys, xs = np.mgrid[0:altura, 0:largura]
    t = xs * math.cos(angulo_rad) + ys * math.sin(angulo_rad)
    t_min, t_max = float(t.min()), float(t.max())
    if t_max - t_min < 1e-9:
        return np.zeros(bins)
    indice = np.clip(((t - t_min) / (t_max - t_min) * (bins - 1)).astype(np.int32), 0, bins - 1)
    plano = indice.ravel()
    soma = np.bincount(plano, weights=campo.ravel(), minlength=bins)
    contagem = np.bincount(plano, minlength=bins)
    return soma / np.maximum(contagem, 1)


def _matriz_dft(n: int, k_max: int) -> np.ndarray:
    """Matriz da DFT 1D, montada à mão.

    Sem ``numpy.fft`` de propósito: a DLL dele é bloqueada por política de
    aplicativo em algumas máquinas Windows, e uma multiplicação de matriz de
    511x1024 não justifica uma dependência que impede o módulo de subir.
    """
    k = np.arange(1, k_max + 1).reshape(-1, 1)
    t = np.arange(n).reshape(1, -1)
    return np.exp(-2j * np.pi * k * t / n)


def _fundo_local(magnitude: np.ndarray, janela: int = 25) -> np.ndarray:
    """Tendência suave do espectro, para servir de linha de base.

    O espectro de imagem natural cai como 1/f: muita energia nas frequências
    baixas, pouquíssima nas altas. Comparar o pico com a **mediana global**, como
    a primeira versão fazia, dá razão na casa das centenas para qualquer
    ondulação de baixa frequência — e foi por isso que mata e pasto vinham
    marcados como lavoura com força 300.

    O que interessa é o quanto o pico se destaca da **vizinhança dele**. Uma
    mediana deslizante dá essa linha de base sem supor forma nenhuma para a
    curva.
    """
    metade = janela // 2
    esticado = np.pad(magnitude, metade, mode="edge")
    vistas = np.lib.stride_tricks.sliding_window_view(esticado, janela)
    return np.median(vistas, axis=-1)


def _espectro(perfil: np.ndarray, dft: np.ndarray) -> np.ndarray | None:
    """Destaque de cada frequência sobre o fundo local dela.

    O índice ``i`` do vetor corresponde a ``k = i + 1`` repetições ao longo do
    perfil — é a convenção de :func:`_matriz_dft`, que começa em 1 porque a
    componente contínua não interessa.

    Devolve ``None`` quando o perfil é constante: não há espectro de uma linha
    reta, e inventar um daria destaque infinito sobre ruído numérico.
    """
    # Tirar a tendência antes de olhar frequência. Um talhão com um lado mais
    # iluminado que o outro tem rampa no perfil, e rampa vaza energia por todo o
    # espectro — um gradiente liso chegava a destaque 40, quase o de uma lavoura.
    t = np.linspace(-1.0, 1.0, perfil.size)
    coef = np.polyfit(t, perfil, 1)
    x = (perfil - np.polyval(coef, t)) * np.hanning(perfil.size)
    if x.std() < 1e-9:
        return None
    magnitude = np.abs(dft @ x)
    return magnitude / np.maximum(_fundo_local(magnitude), 1e-9)


def _pico(perfil: np.ndarray, dft: np.ndarray, k_min: int, k_max: int) -> tuple[int, float]:
    """Harmônico dominante dentro da faixa. Devolve ``(k, destaque)``.

    ``destaque`` é a razão entre o pico e o fundo local do espectro naquele
    ponto: quantas vezes aquela frequência específica se levanta acima do que as
    vizinhas dela fazem. É um número comparável entre imagens, que a razão sobre
    a mediana global não era.
    """
    destaque = _espectro(perfil, dft)
    if destaque is None:
        return 0, 0.0

    faixa = destaque[k_min - 1 : k_max]
    if faixa.size == 0:
        return 0, 0.0
    desloc = int(np.argmax(faixa))
    return k_min + desloc, float(faixa[desloc])


def _fino(perfil: np.ndarray, dft: np.ndarray, k_max: int) -> float:
    """Maior destaque **acima** da faixa aceita, onde o período é fino demais.

    É a medida que denuncia o padrão que a imagem não resolve. O índice
    ``k_max`` é o último ``k`` da faixa; ``destaque[k_max:]`` começa portanto em
    ``k = k_max + 1``, que já é período menor que ``periodo_min_px``.
    """
    destaque = _espectro(perfil, dft)
    if destaque is None:
        return 0.0
    acima = destaque[k_max:]
    return float(acima.max()) if acima.size else 0.0


def detect_rows(
    imagem: np.ndarray,
    *,
    metros_por_pixel: float | None = None,
    norte_deg: float = 0.0,
    periodo_min_px: float = PERIODO_MIN_PX,
    periodo_max_px: float = PERIODO_MAX_PX,
) -> Ruas:
    """Acha a direção e o espaçamento das ruas de plantio.

    ``imagem`` é um ``ndarray`` ``(altura, largura, 3)`` em RGB.
    ``metros_por_pixel`` e ``norte_deg`` vêm do georreferenciamento; sem eles a
    resposta sai em pixels, no referencial da imagem, e diz isso.

    O método: índice de verde, projeção em 180 direções, e vence o ângulo cujo
    perfil tem o harmônico mais destacado **dentro da faixa de período
    plausível**. A razão entre esse pico e a mediana do espectro é a força, e é
    ela que separa lavoura em linha de mata.
    """
    if imagem.ndim != 3 or imagem.shape[2] < 3:
        raise ValueError("A imagem precisa ser RGB com três canais.")
    if min(imagem.shape[:2]) < JANELA_MINIMA:
        raise ValueError(
            f"Recorte de {imagem.shape[1]}x{imagem.shape[0]} px é pequeno demais; "
            f"o mínimo é {JANELA_MINIMA} px de lado."
        )

    # Realce na escala da rua: o que é uniforme deixa de atrapalhar.
    campo = _realce(_verde(imagem), periodo_max_px * 1.5)
    altura, largura = campo.shape
    dft = _matriz_dft(BINS_PERFIL, BINS_PERFIL // 2 - 1)

    # (forca, k, angulo_rad, span_px, k_max)
    melhor: tuple[float, int, float, float, int] = (0.0, 0, 0.0, 0.0, 0)
    for passo in range(BINS_ANGULO):
        angulo = math.pi * passo / BINS_ANGULO
        span = abs(largura * math.cos(angulo)) + abs(altura * math.sin(angulo))
        if span < periodo_min_px * 4:
            continue
        # k é o número de repetições ao longo do perfil inteiro, então o período
        # em pixels é span/k — e a faixa aceita de período vira uma faixa de k.
        k_min = max(2, int(math.ceil(span / periodo_max_px)))
        k_max = min(BINS_PERFIL // 2 - 1, int(math.floor(span / periodo_min_px)))
        if k_max <= k_min:
            continue

        perfil = _perfil(campo, angulo)
        k, forca = _pico(perfil, dft, k_min, k_max)
        if k and forca > melhor[0]:
            melhor = (forca, k, angulo, span, k_max)

    forca, k, angulo_rad, span, k_max_vencedor = melhor
    if not k:
        return Ruas(0.0, 0.0, 0.0, confiavel=False)

    # O portão do ruído fino, no ângulo que venceu: se há mais estrutura acima
    # da faixa do que dentro dela, o pico de dentro é moiré de um padrão que
    # esta imagem não resolve — ver RUIDO_FINO_MAXIMO.
    fino = _fino(_perfil(campo, angulo_rad), dft, k_max_vencedor)
    razao_fina = fino / max(forca, 1e-9)

    # O perfil varia ATRAVESSANDO as ruas; a direção delas é a perpendicular.
    angulo = (math.degrees(angulo_rad) + 90.0) % 180.0
    periodo_px = span / k
    confiavel = forca >= FORCA_MINIMA and razao_fina <= RUIDO_FINO_MAXIMO

    return Ruas(
        angulo_deg=round(angulo, 2),
        periodo_px=round(periodo_px, 2),
        forca=round(forca, 2),
        confiavel=confiavel,
        espacamento_m=(
            round(periodo_px * metros_por_pixel, 3) if metros_por_pixel and confiavel else None
        ),
        azimute_deg=(round((angulo + norte_deg) % 360.0, 2) if confiavel else None),
    )


# ---------------------------------------------------------------------------
# Cobertura do terreno
# ---------------------------------------------------------------------------

LAVOURA = "lavoura"
MATA = "mata"
SOLO = "solo"
OUTRO = "outro"

CLASSES = (LAVOURA, MATA, SOLO, OUTRO)


@dataclass(frozen=True, slots=True)
class Cobertura:
    """O que há em cada pedaço do terreno, e o quanto disso é obstáculo."""

    classes: np.ndarray
    """Matriz de blocos, cada célula com um dos valores de :data:`CLASSES`."""

    bloco_px: int
    """Lado da janela analisada. Vem do espaçamento das ruas: quatro períodos."""

    fracoes: dict[str, float]
    passo_px: int = 0
    """Distância entre centros de janela. Metade do bloco: as janelas se
    sobrepõem, e é isso que dá resolução para localizar o objeto."""

    heuristica: bool = True
    """Sempre verdadeiro por enquanto: isto é regra de cor e textura, não modelo
    treinado. O campo existe para que a resposta nunca seja lida como medição."""

    @property
    def fracao_obstaculo(self) -> float:
        """Quanto do terreno o rover não atravessa: mata mais o que não se sabe."""
        return self.fracoes.get(MATA, 0.0) + self.fracoes.get(OUTRO, 0.0)


def bloco_para(periodo_px: float, minimo: int = 48, maximo: int = 192) -> int:
    """Lado do bloco que consegue enxergar um padrão de ``periodo_px``.

    Quatro períodos por bloco. Menos que isso e a análise de frequência não tem
    o que medir — um bloco de 32 px sobre ruas de 24 px contém uma onda e um
    terço, e nenhum método de frequência decide nada com isso.

    Foi esse o erro da primeira versão: bloco fixo de 48 px, lavoura de rua
    larga classificada como "não sei", e área operável saindo pela metade.
    """
    return int(max(minimo, min(maximo, round(periodo_px * 4))))


def detect_cover(
    imagem: np.ndarray,
    bloco_px: int = 48,
    *,
    periodo_px: float | None = None,
    passo_px: int | None = None,
) -> Cobertura:
    """Separa lavoura, mata e solo exposto por cor e textura, em blocos.

    A regra, em ordem:

    1. pouco verde → **solo** exposto (carreador, terreiro, lavoura colhida);
    2. verde com textura **periódica** → **lavoura** em linha;
    3. verde escuro, textura forte e sem período → **mata**: copa grande e
       irregular, que é obstáculo para o rover;
    4. o resto → **outro**, que é a resposta honesta para o que a regra não
       separa — pasto alto, sombra de nuvem, palha.

    É heurística declarada, não segmentação treinada. O agrônomo corrige no
    console, e é a correção dele que vira dataset.
    """
    if imagem.ndim != 3 or imagem.shape[2] < 3:
        raise ValueError("A imagem precisa ser RGB com três canais.")

    verde = _verde(imagem)
    altura, largura = verde.shape
    # A janela precisa ser larga para a frequência decidir; o passo precisa ser
    # curto para localizar. Sobrepor resolve os dois: classifica com a janela
    # inteira, mas anda meia janela por vez. Sem isso, uma moita de mata do
    # tamanho de um bloco se dilui na maioria e some do mapa.
    passo = passo_px or max(1, bloco_px // 2)
    if altura < bloco_px or largura < bloco_px:
        raise ValueError("Imagem pequena demais para o tamanho de bloco pedido.")
    nl = max(1, (altura - bloco_px) // passo + 1)
    nc = max(1, (largura - bloco_px) // passo + 1)

    # Primeira passada: mede cada janela. Segunda: decide, com limiares tirados
    # da distribuição da própria imagem.
    #
    # Os limiares são relativos de propósito. Com número fixo sobre um índice
    # normalizado globalmente, a mesma lavoura classificava diferente conforme o
    # que mais estivesse no quadro — uma estrada de terra no canto deslocava a
    # normalização e a lavoura inteira mudava de classe. Limiar relativo é o que
    # torna a leitura estável entre fotos de exposições diferentes.
    medias = np.zeros((nl, nc))
    desvios = np.zeros((nl, nc))
    brilhos = np.zeros((nl, nc))
    periodos = np.zeros((nl, nc))
    for i in range(nl):
        for j in range(nc):
            fatia = (
                slice(i * passo, i * passo + bloco_px),
                slice(j * passo, j * passo + bloco_px),
            )
            janela = verde[fatia]
            medias[i, j] = janela.mean()
            desvios[i, j] = janela.std()
            brilhos[i, j] = imagem[fatia].astype(np.float32).mean()
            periodos[i, j] = _periodicidade(janela, periodo_px)

    verde_tipico = float(np.median(medias))
    escuro_tipico = float(np.median(brilhos))
    aspero_tipico = float(np.median(desvios))

    classes = np.empty((nl, nc), dtype=object)
    for i in range(nl):
        for j in range(nc):
            classes[i, j] = _classifica(
                media=float(medias[i, j]),
                desvio=float(desvios[i, j]),
                brilho=float(brilhos[i, j]),
                periodicidade=float(periodos[i, j]),
                verde_tipico=verde_tipico,
                escuro_tipico=escuro_tipico,
                aspero_tipico=aspero_tipico,
            )

    total = nl * nc
    fracoes = {nome: float((classes == nome).sum()) / total for nome in CLASSES}
    return Cobertura(classes=classes, bloco_px=bloco_px, fracoes=fracoes, passo_px=passo)


def _classifica(
    *,
    media: float,
    desvio: float,
    brilho: float,
    periodicidade: float,
    verde_tipico: float,
    escuro_tipico: float,
    aspero_tipico: float,
) -> str:
    """Decide a classe de uma janela, com limiares relativos à imagem.

    A ordem importa e cada regra só é avaliada quando as anteriores não
    decidiram:

    1. **solo** — pouco verde **e mais claro** que o típico. Carreador,
       terreiro, talhão colhido. O brilho é o que o separa da mata, que também
       tem pouco verde mas é escura;
    2. **lavoura** — padrão que se repete no espaçamento das ruas. É a única
       classe com evidência positiva, e por isso vem antes das outras;
    3. **mata** — mais escura que o típico, com pouco verde ou textura áspera:
       copa grande e irregular, em sombra;
    4. **outro** — o que sobrou. É resposta, não desistência: pasto alto, palha
       e sombra de nuvem caem aqui, e o rover não deve entrar em nenhum deles
       sem alguém olhar.
    """
    pouco_verde = media < verde_tipico * 0.6
    claro = brilho > escuro_tipico * 1.08
    escuro = brilho < escuro_tipico * 0.92

    # Solo exposto e mata têm as duas pouco verde; o que as separa é o brilho.
    # Terra é clara e avermelhada, copa em sombra é escura. Sem esta distinção,
    # a moita de mata vinha classificada como carreador — e o planejador
    # mandaria o rover atravessar por dentro dela.
    if pouco_verde and claro:
        return SOLO
    if periodicidade >= 2.2:
        return LAVOURA
    if escuro and (pouco_verde or desvio > aspero_tipico * 1.05):
        return MATA
    return OUTRO


def _periodicidade(bloco: np.ndarray, periodo_px: float | None = None) -> float:
    """O quanto o bloco tem padrão que se repete, na melhor direção.

    Com ``periodo_px`` conhecido — medido uma vez na imagem inteira — a busca se
    fecha em volta dele, entre 70% e 140%. É a diferença entre procurar uma nota
    sabendo qual é e varrer o teclado inteiro: numa moita de mata qualquer
    frequência tem algum pico, e sem a faixa estreita a mata passa por lavoura.
    """
    campo = bloco - bloco.mean()
    if campo.std() < 1e-6:
        return 0.0
    bins = 128
    dft = _matriz_dft(bins, bins // 2 - 1)
    lado = float(max(campo.shape))

    if periodo_px and periodo_px > 0:
        alvo_min, alvo_max = periodo_px * 0.7, periodo_px * 1.4
    else:
        alvo_min, alvo_max = PERIODO_MIN_PX, PERIODO_MAX_PX
    k_min = max(2, int(math.ceil(lado / alvo_max)))
    k_max = min(bins // 2 - 1, int(math.floor(lado / alvo_min)))
    if k_max <= k_min:
        return 0.0

    melhor = 0.0
    for passo in range(12):
        perfil = _perfil(campo, math.pi * passo / 12, bins=bins)
        _, forca = _pico(perfil, dft, k_min, k_max)
        melhor = max(melhor, forca)
    return melhor
