"""Encaixe nos carreadores: o talhão de café é um polígono cercado de estrada de terra.

O encaixe radial (:mod:`app.boundary`) procura onde a *cor muda* saindo do centro. Num
cafezal isso falha do jeito mais comum possível: o talhão vizinho é o mesmo café, com a
mesma cor e a mesma textura, e o que separa os dois é uma faixa de terra de 4 a 6 m — o
**carreador**. O raio atravessa a faixa sem ver borda e o contorno vaza para o vizinho; e
como o método parte do centro, talhão comprido e estreito (o formato normal de café em
encosta) vira círculo.

Aqui o desenho da pessoa é lido como o que ele é: **um polígono, lado a lado**. Para cada
lado, a IA procura, numa faixa em volta dele, a linha reta que mais parece carreador —
mais clara e mais avermelhada que **os dois lados** (crista de solo exposto). Sem crista,
vale a borda em degrau (café x pasto, café x mata). O lado encosta na linha achada, e os
cantos saem da interseção dos lados vizinhos. É o gesto de quem conhece a lavoura: "o
talhão vai até a estrada".

**Lado sem carreador nem borda nítida não se mexe.** A resposta diz quais lados
encaixaram, quanto cada um andou e com que nitidez — e a pessoa continua podendo arrastar
o vértice. É medida de cor e geometria, não modelo treinado.

Limite conhecido: o contorno fica no **meio** do carreador. Para a rota isso é o certo
(é por ali que o rover manobra na cabeceira); a área de plantio é um pouco menor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ANGULOS_DEG = (-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0)
"""Giro que um lado pode ter em relação ao traço. Quem desenha erra mais a posição que o
ângulo de uma estrada reta; ±6° cobre a mão torta sem deixar o lado virar outro."""

LADO_MINIMO_M = 12.0
"""Lado mais curto que isto (um clique a mais no canto) não é procurado sozinho: ele
liga os vizinhos encaixados."""

NITIDEZ_MINIMA = 3.0
"""Quanto a melhor linha precisa se destacar das outras candidatas (em desvios robustos)
para o lado encaixar. Abaixo disso não há carreador ali — o lado fica como foi desenhado."""

PREFERENCIA_CRISTA = 0.75
"""Peso da borda em degrau contra a crista: com as duas na faixa, vence o carreador."""


@dataclass
class Lado:
    indice: int
    encaixou: bool
    tipo: str | None
    """``carreador`` (crista de solo), ``borda`` (degrau: café x pasto/mata) ou ``None``."""
    deslocamento_px: float
    giro_deg: float
    nitidez: float


@dataclass
class Encaixe:
    pontos: list[tuple[float, float]]
    """O contorno encaixado, em pixels (x, y) da imagem."""
    lados: list[Lado] = field(default_factory=list)

    @property
    def confianca(self) -> float:
        """Fração dos lados (procurados) que encaixaram em algo nítido."""
        procurados = [lado for lado in self.lados if lado.tipo != "curto"]
        if not procurados:
            return 0.0
        return sum(lado.encaixou for lado in procurados) / len(procurados)


def mapa_de_solo(imagem: np.ndarray) -> np.ndarray:
    """Quanto cada pixel parece solo exposto, padronizado (mediana 0, escala robusta).

    Brilho mais "vermelhidão" (R - G): a terra do carreador é clara e puxa para o
    vermelho; a copa do café é escura e verde. Suavizado em 3x3 para a estrada contar como
    faixa e não como pixel.
    """
    rgb = imagem[..., :3].astype(np.float64) / 255.0
    brilho = rgb.mean(axis=2)
    vermelho = (rgb[..., 0] - rgb[..., 1]) / (rgb[..., 0] + rgb[..., 1] + 1e-6)
    solo = _padroniza(brilho) + _padroniza(vermelho)
    return _caixa3(solo)


def _padroniza(campo: np.ndarray) -> np.ndarray:
    mediana = float(np.median(campo))
    q1, q3 = np.percentile(campo, [25, 75])
    escala = float(q3 - q1) or float(campo.std()) or 1.0
    return (campo - mediana) / escala


def _caixa3(campo: np.ndarray) -> np.ndarray:
    preenchido = np.pad(campo, 1, mode="edge")
    soma = np.zeros_like(campo)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            soma += preenchido[dy : dy + campo.shape[0], dx : dx + campo.shape[1]]
    return soma / 9.0


def _amostra(campo: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Valor mais próximo em cada ponto; fora da imagem, NaN."""
    h, w = campo.shape
    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)
    dentro = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    saida = np.full(xs.shape, np.nan)
    saida[dentro] = campo[yi[dentro], xi[dentro]]
    return saida


def _procura_lado(
    solo: np.ndarray, a: np.ndarray, b: np.ndarray, faixa_px: float, meia_estrada_px: float
) -> tuple[float, float, float, str | None]:
    """A melhor linha em volta do lado a→b: (deslocamento, giro, nitidez, tipo)."""
    meio = (a + b) / 2.0
    comprimento = float(np.linalg.norm(b - a))
    ts = np.linspace(-0.4, 0.4, max(16, int(comprimento * 0.8)))  # o miolo: longe dos cantos
    offsets = np.arange(-math.ceil(faixa_px), math.ceil(faixa_px) + 1, 1.0)
    base = math.atan2(b[1] - a[1], b[0] - a[0])
    lado_px = max(2.0, meia_estrada_px * 1.8)

    notas = []  # (nota, deslocamento, giro, tipo)
    for giro in ANGULOS_DEG:
        ang = base + math.radians(giro)
        u = np.array([math.cos(ang), math.sin(ang)])
        n = np.array([-u[1], u[0]])
        # (offsets, ts): pontos ao longo de cada linha candidata
        ao_lado = offsets[:, None]
        ao_longo = (ts * comprimento)[None, :]
        px = meio[0] + ao_lado * n[0] + ao_longo * u[0]
        py = meio[1] + ao_lado * n[1] + ao_longo * u[1]
        linha = np.nanmean(_amostra(solo, px, py), axis=1)
        esquerda = np.nanmean(_amostra(solo, px - n[0] * lado_px, py - n[1] * lado_px), axis=1)
        direita = np.nanmean(_amostra(solo, px + n[0] * lado_px, py + n[1] * lado_px), axis=1)
        crista = np.minimum(linha - esquerda, linha - direita)
        degrau = np.abs(esquerda - direita) * 0.5
        for k, d in enumerate(offsets):
            c, g = crista[k], degrau[k]
            if np.isnan(c) or np.isnan(g):
                continue
            if c >= PREFERENCIA_CRISTA * g:
                notas.append((float(c), float(d), giro, "carreador"))
            else:
                notas.append((float(PREFERENCIA_CRISTA * g), float(d), giro, "borda"))
    if not notas:
        return 0.0, 0.0, 0.0, None
    valores = np.array([nota for nota, *_ in notas])
    mediana = float(np.median(valores))
    escala = float(np.median(np.abs(valores - mediana))) * 1.4826 or 1e-6
    # A pessoa desenhou perto: entre duas estradas parecidas, vence a mais próxima.
    penalidade = 0.15 * escala * np.array([abs(d) for _, d, *_ in notas]) / max(1.0, faixa_px)
    melhor = int(np.argmax(valores - penalidade))
    nota, d, giro, tipo = notas[melhor]
    return d, giro, (nota - mediana) / escala, tipo


def _reta(a: np.ndarray, b: np.ndarray, d: float, giro: float) -> tuple[np.ndarray, np.ndarray]:
    """O lado a→b deslocado ``d`` px na normal e girado ``giro`` graus no meio."""
    meio = (a + b) / 2.0
    ang = math.atan2(b[1] - a[1], b[0] - a[0]) + math.radians(giro)
    u = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-u[1], u[0]])
    return meio + n * d, u


def _cruza(p1: np.ndarray, u1: np.ndarray, p2: np.ndarray, u2: np.ndarray) -> np.ndarray | None:
    det = u1[0] * (-u2[1]) - u1[1] * (-u2[0])
    if abs(det) < math.sin(math.radians(12)):  # quase paralelos: o canto não é confiável
        return None
    t = ((p2[0] - p1[0]) * (-u2[1]) - (p2[1] - p1[1]) * (-u2[0])) / det
    return p1 + u1 * t


def _area(pontos: np.ndarray) -> float:
    x, y = pontos[:, 0], pontos[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def encaixa_nos_carreadores(
    imagem: np.ndarray,
    desenho_px: np.ndarray,
    *,
    metros_por_pixel: float,
    faixa_m: float = 25.0,
    largura_carreador_m: float = 5.0,
) -> Encaixe:
    """Encaixa cada lado do desenho no carreador (ou na borda) mais nítido da faixa.

    ``desenho_px``: vértices (x, y) em pixels, sem repetir o primeiro. ``faixa_m``: o
    quanto um lado pode andar para cada lado.
    """
    pontos = np.asarray(desenho_px, dtype=np.float64)
    if pontos.ndim != 2 or pontos.shape[0] < 3:
        raise ValueError("O desenho precisa de ao menos 3 vértices.")
    if metros_por_pixel <= 0:
        raise ValueError("Resolução inválida.")
    solo = mapa_de_solo(imagem)
    faixa_px = faixa_m / metros_por_pixel
    meia_estrada_px = largura_carreador_m / 2.0 / metros_por_pixel
    n = len(pontos)

    lados: list[Lado] = []
    retas: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        a, b = pontos[i], pontos[(i + 1) % n]
        if float(np.linalg.norm(b - a)) * metros_por_pixel < LADO_MINIMO_M:
            lados.append(Lado(i, False, "curto", 0.0, 0.0, 0.0))
            retas.append(_reta(a, b, 0.0, 0.0))
            continue
        d, giro, nitidez, tipo = _procura_lado(solo, a, b, faixa_px, meia_estrada_px)
        if tipo is not None and nitidez >= NITIDEZ_MINIMA:
            lados.append(Lado(i, True, tipo, d, giro, nitidez))
            retas.append(_reta(a, b, d, giro))
        else:
            lados.append(Lado(i, False, None, 0.0, 0.0, nitidez))
            retas.append(_reta(a, b, 0.0, 0.0))

    novos: list[tuple[float, float]] = []
    for i in range(n):
        antes, depois = retas[i - 1], retas[i]
        canto = _cruza(antes[0], antes[1], depois[0], depois[1])
        original = pontos[i]
        if canto is None or float(np.linalg.norm(canto - original)) > 2.5 * faixa_px:
            # Lados quase paralelos (ou um canto que fugiria longe): o vértice desce no
            # lado de depois, no ponto mais perto do original.
            p, u = depois
            canto = p + u * float(np.dot(original - p, u))
        novos.append((float(canto[0]), float(canto[1])))

    resultado = np.array(novos)
    area0, area1 = _area(pontos), _area(resultado)
    if area0 > 0 and not 0.4 < area1 / area0 < 2.0:
        # Encaixe que dobra ou some com a área não é encaixe: devolve o desenho.
        return Encaixe([tuple(map(float, p)) for p in pontos],
                       [Lado(lado.indice, False, None, 0.0, 0.0, lado.nitidez) for lado in lados])
    return Encaixe(novos, lados)


@dataclass
class Linhas:
    angulo_imagem_deg: float
    """Direção das linhas no referencial da imagem (x para a direita, y para baixo), [0, 180)."""
    espacamento_m: float
    forca: float
    """Pico do espectro sobre a mediana da faixa de espaçamento: listra de verdade passa de 8."""


ESPACAMENTO_M = (2.2, 4.8)
"""Rua de café: de adensado (2,2 m) a mecanizado largo (4,8 m)."""

FORCA_MINIMA_LINHAS = 8.0


def direcao_das_linhas(
    imagem: np.ndarray, poligono_px: np.ndarray, *, metros_por_pixel: float, margem_m: float = 8.0
) -> Linhas | None:
    """Direção e espaçamento das linhas de plantio pelo espectro do miolo do talhão.

    O miolo fica a ``margem_m`` dos lados — a borda do carreador é a listra mais forte da
    imagem e puxaria a direção para ela. O espectro 2D (com janela de Hann) é procurado só
    na faixa de espaçamento de café; o pico dá a normal das listras e o período.

    Serve quando o detector por frequência (:func:`app.terrain.detect_rows`) não resolve a
    rua: em imagem de ~1 m/px a linha de café ainda risca a copa a cada 3 px. ``None``
    quando a resolução não alcança a faixa ou não há listra.
    """
    nyquist_m = 2.0 * metros_por_pixel
    minimo_m = max(ESPACAMENTO_M[0], nyquist_m * 1.05)
    if minimo_m >= ESPACAMENTO_M[1]:
        return None
    rgb = imagem[..., :3].astype(np.float64)
    verde = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    mascara = _mascara(poligono_px, verde.shape) & (
        _distancia_aos_lados(poligono_px, verde.shape) * metros_por_pixel > margem_m)
    if mascara.sum() < 1600:
        return None
    ys, xs = np.nonzero(mascara)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    recorte = verde[y0:y1, x0:x1]
    dentro = mascara[y0:y1, x0:x1]
    campo = np.where(dentro, recorte - recorte[dentro].mean(), 0.0)
    h, w = campo.shape
    campo = campo * np.outer(np.hanning(h), np.hanning(w))
    potencia = np.abs(np.fft.fftshift(np.fft.fft2(campo))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    raio = np.hypot(fx, fy)  # ciclos por pixel
    faixa = (raio >= metros_por_pixel / ESPACAMENTO_M[1]) & (raio <= metros_por_pixel / minimo_m)
    if faixa.sum() < 20:
        return None
    valores = potencia[faixa]
    k = int(np.argmax(np.where(faixa, potencia, -1.0)))
    iy, ix = np.unravel_index(k, potencia.shape)
    pico_fx, pico_fy = float(fx[0, ix]), float(fy[iy, 0])
    forca = float(potencia[iy, ix] / (np.median(valores) or 1e-12))
    normal = math.degrees(math.atan2(pico_fy, pico_fx))
    return Linhas(angulo_imagem_deg=(normal + 90.0) % 180.0,
                  espacamento_m=round(metros_por_pixel / float(raio[iy, ix]), 2),
                  forca=round(forca, 1))


def _distancia_aos_lados(poligono: np.ndarray, forma: tuple[int, int]) -> np.ndarray:
    h, w = forma
    ys, xs = np.mgrid[0:h, 0:w]
    px, py = xs + 0.5, ys + 0.5
    pts = np.asarray(poligono, dtype=np.float64)
    menor = np.full(forma, np.inf)
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        dx, dy = bx - ax, by - ay
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / ((dx * dx + dy * dy) or 1e-12), 0, 1)
        menor = np.minimum(menor, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return menor


def _mascara(poligono: np.ndarray, forma: tuple[int, int]) -> np.ndarray:
    """Pixels dentro do polígono (par-ímpar), sem dependência de imagem."""
    h, w = forma
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs + 0.5
    ys = ys + 0.5
    dentro = np.zeros(forma, dtype=bool)
    pts = np.asarray(poligono, dtype=np.float64)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cruza = ((yi > ys) != (yj > ys)) & (xs < (xj - xi) * (ys - yi) / ((yj - yi) or 1e-12) + xi)
        dentro ^= cruza
        j = i
    return dentro
