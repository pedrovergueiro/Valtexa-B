"""Imagem de satélite para uma área, montada a partir de tiles.

Existe para uma coisa só: dar pixels ao encaixe de contorno
(:mod:`app.boundary`) quando o produtor desenha sobre o mapa do console e ainda
não há ortomosaico do talhão.

## Licenciamento — ler antes de ligar em produção

O console desenha o mapa com o *World Imagery* da Esri, que é gratuito para
exibição. **Baixar esses tiles no servidor para processar** é outro uso, e as
condições da Esri não o autorizam em geral. Este módulo é a peça técnica; a
decisão comercial não é dele.

Três caminhos, na ordem em que valem:

1. **O ortomosaico do próprio cliente**, entregue pelo operador do voo. É a
   imagem certa — mais recente, mais nítida (3 a 10 cm/px contra 30 a 60 da
   Esri) e sem dúvida de licença. É também a única que resolve rua de café.
2. **Um provedor contratado** (Mapbox, Maxar, Planet, Google Maps Platform), que
   licencia explicitamente o uso programático.
3. **A Esri**, só com contrato que permita.

Por isso :data:`FONTE_PADRAO` aponta para nenhuma e :func:`buscar_area` exige
que o chamador diga a fonte. Não há padrão silencioso.
"""

from __future__ import annotations

import io
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

TILE = 256
"""Lado do tile no esquema XYZ, em pixels. Padrão de fato da web."""

MAX_TILES = 64
"""Teto de tiles por pedido. Acima disso o recorte é grande demais para a
finalidade — e a conta de rede deixa de ser educada com quem serve a imagem."""

FONTE_PADRAO: str | None = None
"""Sem padrão de propósito. Ver a nota de licenciamento no topo do módulo."""


@dataclass(frozen=True, slots=True)
class Recorte:
    """Um pedaço de mundo em pixels, com o que é preciso para voltar a graus."""

    imagem: np.ndarray
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    zoom: int

    @property
    def metros_por_pixel(self) -> float:
        """Resolução no centro do recorte."""
        lat = (self.lat_min + self.lat_max) / 2
        return 156543.03392 * math.cos(math.radians(lat)) / (2**self.zoom)

    def para_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        """Graus para pixel dentro deste recorte."""
        altura, largura = self.imagem.shape[:2]
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * largura
        # A latitude não é linear em Mercator; converter pelo y do esquema.
        y0, y1 = _y_mercator(self.lat_max), _y_mercator(self.lat_min)
        y = (_y_mercator(lat) - y0) / (y1 - y0) * altura
        return x, y

    def para_grau(self, x: float, y: float) -> tuple[float, float]:
        """Pixel deste recorte para graus."""
        altura, largura = self.imagem.shape[:2]
        lon = self.lon_min + x / largura * (self.lon_max - self.lon_min)
        y0, y1 = _y_mercator(self.lat_max), _y_mercator(self.lat_min)
        lat = _lat_de_mercator(y0 + y / altura * (y1 - y0))
        return lon, lat


def _y_mercator(lat: float) -> float:
    """Coordenada y normalizada do Mercator esférico, 0 no topo e 1 embaixo."""
    seno = math.sin(math.radians(max(-85.05, min(85.05, lat))))
    return 0.5 - math.log((1 + seno) / (1 - seno)) / (4 * math.pi)


def _lat_de_mercator(y: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))


def _tile_de(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    return (lon + 180.0) / 360.0 * n, _y_mercator(lat) * n


def zoom_para(lon_min: float, lat_min: float, lon_max: float, lat_max: float,
              lado_alvo: int = 1024) -> int:
    """Maior zoom cujo recorte ainda cabe em ``lado_alvo`` pixels.

    Maior zoom é mais detalhe, e detalhe é o que decide se a borda entre café e
    mata aparece. O teto existe porque o custo cresce com o quadrado.
    """
    for zoom in range(20, 8, -1):
        x0, y0 = _tile_de(lon_min, lat_max, zoom)
        x1, y1 = _tile_de(lon_max, lat_min, zoom)
        largura = abs(x1 - x0) * TILE
        altura = abs(y1 - y0) * TILE
        if max(largura, altura) <= lado_alvo:
            return zoom
    return 9


def buscar_area(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    fonte: str | None = FONTE_PADRAO,
    zoom: int | None = None,
    user_agent: str = "valtexa-b/terrain",
    timeout: float = 20.0,
) -> Recorte:
    """Monta a imagem da área a partir dos tiles de ``fonte``.

    ``fonte`` é um molde XYZ, por exemplo
    ``"https://.../tile/{z}/{y}/{x}"``. É obrigatório: ver a nota de
    licenciamento no topo do módulo.
    """
    if not fonte:
        raise ValueError(
            "Nenhuma fonte de imagem configurada. Aponte para o ortomosaico do "
            "cliente ou para um provedor licenciado — ver app/basemap.py."
        )
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("Retângulo inválido: mínimo precisa ser menor que máximo.")

    z = zoom or zoom_para(lon_min, lat_min, lon_max, lat_max)
    x0, y0 = _tile_de(lon_min, lat_max, z)
    x1, y1 = _tile_de(lon_max, lat_min, z)
    tx0, ty0 = int(math.floor(x0)), int(math.floor(y0))
    tx1, ty1 = int(math.floor(x1)), int(math.floor(y1))
    colunas, linhas = tx1 - tx0 + 1, ty1 - ty0 + 1
    if colunas * linhas > MAX_TILES:
        raise ValueError(
            f"A área pede {colunas * linhas} tiles, acima do teto de {MAX_TILES}. "
            f"Diminua a área ou o zoom."
        )

    mosaico = Image.new("RGB", (colunas * TILE, linhas * TILE))

    def pega(alvo: tuple[int, int]) -> tuple[tuple[int, int], Image.Image | None]:
        tx, ty = alvo
        url = fonte.format(z=z, x=tx, y=ty)
        try:
            pedido = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(pedido, timeout=timeout) as resposta:
                return alvo, Image.open(io.BytesIO(resposta.read())).convert("RGB")
        except Exception:
            # Um tile que falta vira buraco preto, não exceção: o encaixe
            # continua valendo no resto da área, e a região sem imagem aparece
            # como direção sem borda nítida.
            return alvo, None

    alvos = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    with ThreadPoolExecutor(max_workers=8) as piscina:
        for (tx, ty), imagem in piscina.map(pega, alvos):
            if imagem is not None:
                mosaico.paste(imagem, ((tx - tx0) * TILE, (ty - ty0) * TILE))

    # Recorta o mosaico no retângulo pedido.
    esq = int((x0 - tx0) * TILE)
    topo = int((y0 - ty0) * TILE)
    dir_ = int((x1 - tx0) * TILE)
    baixo = int((y1 - ty0) * TILE)
    recorte = mosaico.crop((esq, topo, max(dir_, esq + 1), max(baixo, topo + 1)))

    return Recorte(
        imagem=np.asarray(recorte),
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
        zoom=z,
    )
