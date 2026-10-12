"""Rotas de leitura de terreno: encaixe de contorno e leitura de ruas.

São as rotas que atendem ao gesto do produtor no console: ele contorna a lavoura
a dedo sobre o mapa, e o que volta é o limite que existe na imagem, com as ruas
de plantio lidas por dentro dele.

Vivem aqui, e não em ``app/main.py``, porque não dependem de nenhum modelo
treinado nem dos pesos — sobem mesmo quando o serviço está sem GPU e sem
``models/``. É a diferença entre uma capacidade que funciona no primeiro dia de
um cliente e uma que espera dataset.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, status

from app import basemap, boundary, obstacles, terrain
from app.config import HTTP_422, settings
from app.schemas import (
    CoverInfo,
    ObjectInfo,
    RefineRequest,
    RefineResponse,
    RowsInfo,
    TerrainInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["terreno"])

MARGEM = 0.25
"""Folga em volta do desenho ao buscar a imagem, como fração do tamanho dele.

A borda verdadeira pode estar fora do traço — é justamente o caso de quem
desenhou apertado. Sem folga, o encaixe não teria pixel para onde crescer.
"""


@router.post(
    "/refine-boundary",
    response_model=RefineResponse,
    summary="Encaixa um contorno desenhado a mão no limite real do talhão",
)
def refine_boundary(payload: RefineRequest) -> RefineResponse:
    """Recebe o traço em graus e devolve o contorno encaixado, em graus.

    O gesto é o do recorte inteligente de celular: contorne por cima, sem
    precisão, e o limite salta para onde a lavoura realmente acaba.

    Responde ``422`` quando não há fonte de imagem configurada. É deliberado —
    ver a nota de licenciamento em :mod:`app.basemap`: baixar tile de basemap
    para processar não é o mesmo uso que exibi-lo num mapa.
    """
    fonte = payload.tiles or settings.basemap_tiles
    if not fonte:
        raise HTTPException(
            HTTP_422,
            "Nenhuma fonte de imagem configurada (BASEMAP_TILES). Aponte para o "
            "ortomosaico do cliente ou para um provedor licenciado — ver app/basemap.py.",
        )

    anel = np.asarray(payload.polygon, dtype=np.float64)
    lon_min, lat_min = anel.min(axis=0)
    lon_max, lat_max = anel.max(axis=0)
    folga_lon = (lon_max - lon_min) * MARGEM
    folga_lat = (lat_max - lat_min) * MARGEM
    if folga_lon <= 0 or folga_lat <= 0:
        raise HTTPException(
            HTTP_422, "O desenho não tem área."
        )

    try:
        recorte = basemap.buscar_area(
            lon_min - folga_lon,
            lat_min - folga_lat,
            lon_max + folga_lon,
            lat_max + folga_lat,
            fonte=fonte,
        )
    except ValueError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc
    except Exception as exc:  # rede, tile fora do ar, formato inesperado
        logger.exception("Falha ao buscar a imagem da área")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"Não consegui buscar a imagem: {exc}"
        ) from exc

    em_pixel = np.array([recorte.para_pixel(lon, lat) for lon, lat in anel])
    try:
        contorno = boundary.refine_boundary(
            recorte.imagem, em_pixel, tolerancia=payload.tolerance, vertices=payload.vertices
        )
    except ValueError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc

    em_grau = [list(recorte.para_grau(x, y)) for x, y in contorno.pontos]

    notas: list[str] = []
    if contorno.confianca < 0.5:
        notas.append(
            f"Borda nítida em só {contorno.confianca:.0%} das direções: a imagem não "
            f"mostra um limite claro em boa parte do contorno. Confira antes de salvar."
        )
    if contorno.encostou_no_limite > 0.4:
        notas.append(
            "Boa parte do contorno parou no limite da tolerância, não numa borda. "
            "Desenhe mais folgado ou aumente a tolerância."
        )
    if abs(contorno.variacao) > 0.6:
        notas.append(
            f"A área mudou {contorno.variacao:+.0%} em relação ao seu desenho — "
            f"muito. Vale conferir se o encaixe pegou o talhão certo."
        )

    leitura = _terreno(recorte, em_pixel_encaixado(recorte, em_grau))
    if leitura and not leitura.resolution_ok:
        notas.append(
            "Não deu para ler o terreno nesta imagem: a rua de plantio não se "
            "resolve. O contorno acima continua valendo; obstáculos e área "
            "operável exigem ortomosaico de 3 a 10 cm por pixel."
        )

    return RefineResponse(
        polygon=em_grau,
        confidence=round(contorno.confianca, 3),
        hit_limit=round(contorno.encostou_no_limite, 3),
        area_change=round(contorno.variacao, 4),
        meters_per_pixel=round(recorte.metros_por_pixel, 3),
        zoom=recorte.zoom,
        rows=_ruas(recorte, em_grau),
        terrain=leitura,
        notes=notas,
    )


def em_pixel_encaixado(recorte: basemap.Recorte, anel: list[list[float]]) -> np.ndarray:
    """O contorno já encaixado, de volta em pixels da imagem recortada."""
    return np.array([recorte.para_pixel(lon, lat) for lon, lat in anel])


def _terreno(recorte: basemap.Recorte, anel_px: np.ndarray) -> TerrainInfo | None:
    """Reparte o terreno dentro do contorno e localiza o que não é lavoura.

    Trabalha sobre a imagem inteira do recorte, com o contorno servindo de
    máscara — e não sobre um recorte do polígono. Mascarar criaria bordas retas
    artificiais, que é exatamente o sinal que o detector de rua confundiria com
    linha de plantio.

    Erro de leitura aqui não derruba o encaixe do contorno, que é o que a pessoa
    pediu: devolve ``None`` e a resposta sai sem a parte do terreno.
    """
    try:
        lido = obstacles.detect_obstacles(recorte.imagem, anel_px)
    except ValueError:
        # Contorno fora da imagem, ou imagem pequena demais para um bloco.
        logger.info("Terreno não lido para este contorno", exc_info=True)
        return None

    if not lido.resolucao_suficiente:
        return TerrainInfo(
            resolution_ok=False,
            description=lido.descricao,
            crop_fraction=0.0,
            obstacle_fraction=0.0,
            path_fraction=0.0,
        )

    # A área de um pixel no chão. Vale como aproximação porque o recorte é
    # pequeno perto da escala em que a projeção de Mercator distorce.
    m2_por_pixel = recorte.metros_por_pixel**2

    objetos = [
        ObjectInfo(
            kind=objeto.tipo,
            center=list(recorte.para_grau(*objeto.centro_px)),
            outline=[list(recorte.para_grau(x, y)) for x, y in objeto.contorno_px],
            area_ha=round(objeto.area_px * m2_por_pixel / 10_000.0, 4),
            traversable=objeto.atravessavel,
            confidence=objeto.confianca,
            blocks=objeto.blocos,
        )
        for objeto in lido.objetos
    ]

    return TerrainInfo(
        resolution_ok=True,
        description=lido.descricao,
        crop_fraction=lido.fracao_lavoura,
        obstacle_fraction=lido.fracao_obstaculo,
        path_fraction=lido.fracao_caminho,
        row_spacing_px=lido.periodo_px,
        cover=CoverInfo(
            fractions={k: round(v, 4) for k, v in lido.cobertura.fracoes.items()},
            obstacle_fraction=round(lido.cobertura.fracao_obstaculo, 4),
            heuristic=lido.cobertura.heuristica,
        ),
        objects=objetos,
        heuristic=lido.heuristica,
    )


def _ruas(recorte: basemap.Recorte, anel: list[list[float]]) -> RowsInfo | None:
    """Lê as ruas de plantio dentro do contorno encaixado.

    Recorta o retângulo do contorno em vez do polígono: a leitura é de
    frequência, e um recorte com cantos de fora só acrescenta um pouco de fundo
    — enquanto mascarar o polígono criaria bordas retas artificiais, que são
    exatamente o tipo de sinal que o detector confundiria com rua.
    """
    pontos = np.asarray(anel, dtype=np.float64)
    xs, ys = zip(*[recorte.para_pixel(lon, lat) for lon, lat in pontos], strict=True)
    x0, x1 = int(max(0, min(xs))), int(min(recorte.imagem.shape[1], max(xs)))
    y0, y1 = int(max(0, min(ys))), int(min(recorte.imagem.shape[0], max(ys)))
    if x1 - x0 < terrain.JANELA_MINIMA or y1 - y0 < terrain.JANELA_MINIMA:
        return None

    # O norte da imagem é para cima; o azimute é medido a partir dele.
    lido = terrain.detect_rows(
        recorte.imagem[y0:y1, x0:x1], metros_por_pixel=recorte.metros_por_pixel
    )
    azimute = None
    if lido.confiavel:
        # O ângulo sai no referencial da imagem, com y para baixo; o azimute
        # cresce do norte para o leste.
        azimute = round((90.0 - lido.angulo_deg) % 180.0, 1)

    return RowsInfo(
        detected=lido.confiavel,
        azimuth_deg=azimute,
        spacing_m=lido.espacamento_m,
        strength=lido.forca,
        description=lido.descricao,
    )

