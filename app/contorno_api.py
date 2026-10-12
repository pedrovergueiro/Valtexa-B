"""Rota do encaixe nos carreadores: o contorno do talhão de café lado a lado.

Mesmo pedido e mesma resposta de ``/refine-boundary`` (:mod:`app.terrain_api`), com o
método de :mod:`app.encaixe_carreador` no lugar do encaixe radial — que vaza para o
talhão vizinho quando o vizinho é o mesmo café. A leitura de ruas e de obstáculos é a de
lá, sobre o contorno encaixado aqui; quando a rua não se resolve pela frequência, as
linhas saem do espectro do miolo do talhão.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from fastapi import APIRouter, HTTPException, status

from app import basemap
from app.config import HTTP_422, settings
from app.encaixe_carreador import FORCA_MINIMA_LINHAS, direcao_das_linhas, encaixa_nos_carreadores
from app.schemas import RefineRequest, RefineResponse, RowsInfo
from app.terrain_api import _ruas, _terreno, em_pixel_encaixado

logger = logging.getLogger(__name__)

router = APIRouter(tags=["terreno"])

FOLGA_M = 60.0
"""Imagem em volta do desenho: a faixa de busca de cada lado mais o lado de fora do
carreador, que a comparação de crista precisa ver."""

METROS_POR_GRAU = 111_320.0


def faixa_da_tolerancia(tolerancia: float) -> float:
    """A tolerância do console (fração, 0,05 a 0,6) em metros que um lado pode andar."""
    return float(min(60.0, max(8.0, 8.0 + 60.0 * tolerancia)))


def azimute_da_imagem(angulo_imagem_deg: float) -> float:
    """Ângulo na imagem (x para leste, y para o sul) → azimute a partir do norte, [0, 180)."""
    return round((90.0 + angulo_imagem_deg) % 180.0, 1)


@router.post(
    "/encaixe-carreador",
    response_model=RefineResponse,
    summary="Encaixa o contorno desenhado nos carreadores do talhão (café)",
)
def encaixe_carreador(payload: RefineRequest) -> RefineResponse:
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
    if lon_max - lon_min <= 0 or lat_max - lat_min <= 0:
        raise HTTPException(HTTP_422, "O desenho não tem área.")
    folga_lat = FOLGA_M / METROS_POR_GRAU
    folga_lon = folga_lat / max(0.01, math.cos(math.radians((lat_min + lat_max) / 2)))
    try:
        recorte = basemap.buscar_area(lon_min - folga_lon, lat_min - folga_lat,
                                      lon_max + folga_lon, lat_max + folga_lat, fonte=fonte)
    except ValueError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc
    except Exception as exc:  # rede, tile fora do ar, formato inesperado
        logger.exception("Falha ao buscar a imagem da área")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            f"Não consegui buscar a imagem: {exc}") from exc

    mpp = recorte.metros_por_pixel
    desenho_px = np.array([recorte.para_pixel(lon, lat) for lon, lat in anel])
    try:
        encaixe = encaixa_nos_carreadores(recorte.imagem, desenho_px, metros_por_pixel=mpp,
                                          faixa_m=faixa_da_tolerancia(payload.tolerance))
    except ValueError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc

    em_grau = [list(recorte.para_grau(x, y)) for x, y in encaixe.pontos]
    notas = _notas(encaixe, mpp)
    ruas = _ruas(recorte, em_grau)
    if ruas is None or not ruas.detected:
        linhas = direcao_das_linhas(recorte.imagem, np.array(encaixe.pontos), metros_por_pixel=mpp)
        if linhas is not None and linhas.forca >= FORCA_MINIMA_LINHAS:
            azimute = azimute_da_imagem(linhas.angulo_imagem_deg)
            ruas = RowsInfo(
                detected=True, azimuth_deg=azimute, spacing_m=linhas.espacamento_m,
                strength=linhas.forca,
                description=(f"Ruas lidas no miolo do talhão: {linhas.espacamento_m:.1f} m entre "
                             f"linhas, direção {azimute:.0f}° (força {linhas.forca:.0f})."))
    leitura = _terreno(recorte, em_pixel_encaixado(recorte, em_grau))
    if leitura and not leitura.resolution_ok:
        notas.append("Obstáculos e área operável exigem ortomosaico de 3 a 10 cm por pixel; "
                     "nesta imagem só o contorno e as ruas.")

    area0 = _area(desenho_px)
    area1 = _area(np.array(encaixe.pontos))
    procurados = [lado for lado in encaixe.lados if lado.tipo != "curto"]
    return RefineResponse(
        polygon=em_grau,
        confidence=round(encaixe.confianca, 3),
        hit_limit=round(sum(not lado.encaixou for lado in procurados) / max(1, len(procurados)), 3),
        area_change=round(area1 / area0 - 1.0, 4) if area0 else 0.0,
        meters_per_pixel=round(mpp, 3),
        zoom=recorte.zoom,
        rows=ruas,
        terrain=leitura,
        notes=notas,
    )


def _notas(encaixe, mpp: float) -> list[str]:
    encaixados = [lado for lado in encaixe.lados if lado.encaixou]
    sem = [lado.indice + 1 for lado in encaixe.lados if not lado.encaixou and lado.tipo != "curto"]
    notas = []
    if encaixados:
        partes = [f"lado {lado.indice + 1} no {lado.tipo} ({abs(lado.deslocamento_px) * mpp:.0f} m)"
                  for lado in encaixados]
        notas.append("Encaixou: " + "; ".join(partes) + ".")
    if sem:
        notas.append(f"Sem carreador nem borda nítida no(s) lado(s) {', '.join(map(str, sem))}: "
                     "ficaram como desenhados. Confira e arraste se precisar.")
    return notas


def _area(pontos: np.ndarray) -> float:
    x, y = pontos[:, 0], pontos[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
