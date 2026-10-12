"""Explicação por oclusão: onde o modelo olhou para decidir.

O método é o mais simples que responde à pergunta: cobre-se um pedaço da imagem
com cinza, pergunta-se de novo ao modelo, e mede-se quanto a confiança na classe
caiu. Repetido numa grade, dá o mapa de **quanto cada região sustenta a
resposta** (Zeiler & Fergus, 2014).

## Por que oclusão, e não Grad-CAM

Grad-CAM exige gradiente, e gradiente no Keras 3 é código de backend — um para o
torch, outro para o TensorFlow, outro para o JAX. A oclusão só precisa de
previsões: funciona com qualquer backend, com o dublê dos testes e com qualquer
arquitetura do catálogo, sem uma linha específica de nenhum deles.

E o número que ela devolve tem unidade: "cobrir esta região derruba a confiança
de 0,97 para 0,31". Um mapa de gradiente só diz "aqui é mais quente que ali".

## Para que serve

Para o agrônomo, é ver se o modelo apontou para a lesão. Para quem treina, é o
teste que teria pegado o ``rover-understory`` antes da quarentena: ele acertava
100% lendo a **cor média da sessão de fotografia**, não a doença. Um modelo que
lê lesão perde confiança quando a lesão é coberta; um que lê a luz da foto quase
não perde em lugar nenhum, porque nenhum pedaço sozinho mexe muito na luz. É o que
``max_drop`` mede, e o que ``training.audit_attention`` agrega por classe.

## O custo

``grid²`` passadas pelo modelo, mais uma: 50 na grade padrão de 7. É uma
ferramenta de auditoria e de console, não algo para rodar em todo quadro de voo.

## Quando a oclusão fica cega: evidência redundante — e o RISE

Medido em 26/09/2026 na auditoria do ``leaf-foliar``: em 96% das folhas de
``Common_Rust``, cobrir qualquer célula sozinha mexeu menos de 5 pontos na
confiança — queda mediana de 0,000. Não é atalho (o teste de cor média deu 0%):
é a ferrugem espalhando pústulas pela folha inteira. Tirada uma, as outras
sustentam a resposta, e a oclusão de uma célula por vez não vê nada.

O :func:`rise_map` (Petsiuk, Das & Saenko, 2018) resolve isso cobrindo **metade**
da imagem ao acaso, centenas de vezes. Com metade das lesões escondidas, as que
sobram deixam de bastar, e a contribuição de cada região aparece na média. O
número continua com unidade: quanto a confiança é maior, em média, quando a
célula está visível do que quando está coberta.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

DEFAULT_GRID = 7
"""Células por lado. 7 numa entrada de 224 dá células de 32 px — do tamanho de
uma lesão de ferrugem numa folha que preenche o quadro."""

FILL_VALUE = 0.0
"""Cor da oclusão, na escala ``[-1, 1]`` do modelo: 0 é o cinza médio."""

CHUNK = 32
"""Imagens ocluídas por passada, para limitar a memória em grade fina."""

RISE_SAMPLES = 256
"""Máscaras do RISE. Com metade das células visíveis em cada uma, cada célula
aparece ~128 vezes de cada lado — o bastante para a média de diferença estabilizar."""

RISE_KEEP = 0.5
"""Fração das células visíveis em cada máscara."""

HEAT_COLOR = (220, 38, 38)
HEAT_OPACITY = 0.65

Predictor = Callable[[np.ndarray], np.ndarray]
"""Lote ``(N, H, W, 3)`` na escala do modelo -> probabilidades ``(N, C)``."""


@dataclass(frozen=True)
class OcclusionMap:
    """Quanto cada célula da grade sustenta a classe explicada."""

    class_index: int
    base_probability: float
    """Probabilidade da classe na imagem inteira, sem oclusão."""

    drops: np.ndarray
    """``(grid, grid)``, linha a linha de cima para baixo. Positivo: cobrir a
    célula derruba a classe, ou seja, a célula é evidência **a favor**. Negativo:
    cobrir a célula reforça a classe — ali havia evidência contra."""

    @property
    def grid(self) -> int:
        return int(self.drops.shape[0])

    @property
    def max_drop(self) -> float:
        """A maior queda causada por uma célula sozinha."""
        return float(self.drops.max())

    @property
    def peak(self) -> tuple[int, int]:
        """``(linha, coluna)`` da célula mais decisiva."""
        row, col = np.unravel_index(int(np.argmax(self.drops)), self.drops.shape)
        return int(row), int(col)


def rise_map(
    predict: Predictor,
    image: np.ndarray,
    class_index: int,
    grid: int = DEFAULT_GRID,
    samples: int = RISE_SAMPLES,
    keep: float = RISE_KEEP,
    fill: float = FILL_VALUE,
    base_probability: float | None = None,
    seed: int = 0,
) -> OcclusionMap:
    """Mapeia a evidência cobrindo metade da imagem ao acaso, ``samples`` vezes.

    O valor de cada célula é ``E[p | célula visível] - E[p | célula coberta]``:
    positivo é evidência a favor, na mesma unidade da oclusão. A semente é fixa
    por padrão — o mesmo pedido devolve o mesmo mapa.

    Raises:
        ValueError: se a grade for mais fina que a imagem, a classe não existir
            ou as máscaras não deixarem cada célula dos dois lados.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Esperada uma imagem (H, W, 3); recebido {image.shape}.")
    height, width = image.shape[:2]
    if not 1 <= grid <= min(height, width):
        raise ValueError(f"Grade {grid} incompatível com imagem {width}x{height}.")
    if not 0.0 < keep < 1.0 or samples < 2:
        raise ValueError(f"RISE precisa de 0 < keep < 1 e ao menos 2 máscaras ({keep}, {samples}).")

    rng = np.random.default_rng(seed)
    masks = rng.random((samples, grid, grid)) < keep
    rows, cols = cell_edges(height, grid), cell_edges(width, grid)

    probabilities: list[np.ndarray] = []
    for start in range(0, samples, CHUNK):
        chunk = masks[start : start + CHUNK]
        variants = np.repeat(image[None], len(chunk), axis=0)
        for variant, mask in zip(variants, chunk, strict=True):
            for r, c in zip(*np.nonzero(~mask), strict=True):
                variant[rows[r] : rows[r + 1], cols[c] : cols[c + 1]] = fill
        probabilities.append(_column(predict(variants), class_index))
    scores = np.concatenate(probabilities)

    flat = masks.reshape(samples, -1)
    visible, hidden = flat.sum(axis=0), (~flat).sum(axis=0)
    if (visible == 0).any() or (hidden == 0).any():
        raise ValueError("Poucas máscaras: alguma célula não apareceu dos dois lados.")
    importance = (scores @ flat) / visible - (scores @ ~flat) / hidden

    base = (
        _column(predict(image[None]), class_index)[0]
        if base_probability is None
        else float(np.clip(base_probability, 0.0, 1.0))
    )
    return OcclusionMap(
        class_index=class_index,
        base_probability=float(base),
        drops=importance.reshape(grid, grid),
    )


def cell_edges(size: int, grid: int) -> np.ndarray:
    """Limites das células ao longo de um eixo, cobrindo-o sem sobra nem buraco."""
    return np.linspace(0, size, grid + 1).round().astype(int)


def occlusion_map(
    predict: Predictor,
    image: np.ndarray,
    class_index: int,
    grid: int = DEFAULT_GRID,
    fill: float = FILL_VALUE,
    base_probability: float | None = None,
) -> OcclusionMap:
    """Mapeia a evidência de uma classe numa imagem ``(H, W, 3)`` pré-processada.

    Args:
        base_probability: a probabilidade da classe na imagem inteira, quando
            quem chama já a tem — poupa uma passada. Precisa ter vindo da
            **mesma** imagem pré-processada, senão as quedas medem outra coisa.

    Raises:
        ValueError: se a grade for mais fina que a imagem ou a classe não existir.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Esperada uma imagem (H, W, 3); recebido {image.shape}.")
    height, width = image.shape[:2]
    if not 1 <= grid <= min(height, width):
        raise ValueError(f"Grade {grid} incompatível com imagem {width}x{height}.")

    base = (
        _column(predict(image[None]), class_index)[0]
        if base_probability is None
        else float(np.clip(base_probability, 0.0, 1.0))
    )
    rows, cols = cell_edges(height, grid), cell_edges(width, grid)
    cells = [(r, c) for r in range(grid) for c in range(grid)]

    occluded: list[np.ndarray] = []
    for start in range(0, len(cells), CHUNK):
        chunk = cells[start : start + CHUNK]
        variants = np.repeat(image[None], len(chunk), axis=0)
        for variant, (r, c) in zip(variants, chunk, strict=True):
            variant[rows[r] : rows[r + 1], cols[c] : cols[c + 1]] = fill
        occluded.append(_column(predict(variants), class_index))

    drops = (base - np.concatenate(occluded)).reshape(grid, grid)
    return OcclusionMap(class_index=class_index, base_probability=float(base), drops=drops)


def region_box(row: int, col: int, grid: int, width: int, height: int) -> list[int]:
    """Retângulo ``[x0, y0, x1, y1]`` de uma célula, em pixels da imagem original.

    O pré-processamento estica a imagem para o quadrado sem preservar o aspect
    ratio, então a célula quadrada na entrada do modelo é um retângulo na foto.
    """
    xs, ys = cell_edges(width, grid), cell_edges(height, grid)
    return [int(xs[col]), int(ys[row]), int(xs[col + 1]), int(ys[row + 1])]


def render_overlay(image: Image.Image, occlusion: OcclusionMap, max_side: int = 640) -> bytes:
    """A foto com a evidência pintada por cima, em PNG.

    A intensidade é **absoluta**: vermelho pleno é uma célula que, coberta, zera a
    confiança. Normalizar pelo máximo de cada imagem faria um mapa sem nenhuma
    região decisiva parecer tão quente quanto um com a lesão marcada — e esconder
    exatamente o caso que o mapa existe para mostrar.

    A grade é desenhada em blocos, sem suavizar: interpolar sugeriria uma
    resolução que a oclusão não tem.
    """
    canvas = image.convert("RGB")
    canvas.thumbnail((max_side, max_side))

    scale = max(occlusion.base_probability, 1e-6)
    strength = np.clip(occlusion.drops / scale, 0.0, 1.0) * HEAT_OPACITY
    # Um array 2D de uint8 já vira modo "L"; passar ``mode`` é depreciado no Pillow 11+.
    mask = Image.fromarray((strength * 255).astype(np.uint8)).resize(
        canvas.size, Image.Resampling.NEAREST
    )
    heat = Image.new("RGB", canvas.size, HEAT_COLOR)

    buffer = io.BytesIO()
    Image.composite(heat, canvas, mask).save(buffer, format="PNG")
    return buffer.getvalue()


def _column(probabilities: np.ndarray, class_index: int) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2 or not 0 <= class_index < array.shape[1]:
        raise ValueError(f"Classe {class_index} fora da saída de shape {array.shape}.")
    return np.clip(array[:, class_index], 0.0, 1.0)
