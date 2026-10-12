"""Encaixe nos carreadores, sobre um cafezal sintético: talhões vizinhos do mesmo café,
separados por estrada de terra; ruas a 20° com 4 m entre linhas; desenho torto por cima."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app import basemap
from app.encaixe_carreador import direcao_das_linhas, encaixa_nos_carreadores

LADO = 600
MPP = 1.0
CANTOS = np.array([[150.0, 120.0], [450.0, 150.0], [430.0, 470.0], [140.0, 450.0]])
"""O talhão do meio: a linha central dos carreadores que o cercam."""
ANGULO_RUAS = 20.0
PERIODO_PX = 4.0


def _cafezal(rng_seed: int = 3) -> np.ndarray:
    ys, xs = np.mgrid[0:LADO, 0:LADO].astype(np.float64)
    a = math.radians(ANGULO_RUAS)
    # distância à linha de plantio (normal às ruas): a copa escurece e clareia a cada rua
    normal = -xs * math.sin(a) + ys * math.cos(a)
    onda = 0.5 + 0.5 * np.cos(2 * math.pi * normal / PERIODO_PX)
    rng = np.random.default_rng(rng_seed)
    r = 40 + 18 * onda + rng.normal(0, 5, xs.shape)
    g = 70 + 22 * onda + rng.normal(0, 5, xs.shape)
    b = 35 + 10 * onda + rng.normal(0, 5, xs.shape)
    img = np.dstack([r, g, b])
    # carreadores: o contorno do talhão e as estradas dos vizinhos (mesmo café dos dois lados)
    estradas = [(CANTOS[i], CANTOS[(i + 1) % 4]) for i in range(4)]
    estradas += [(np.array([0.0, 110.0]), np.array([LADO, 165.0]))]  # passa rente ao de cima
    for p, q in estradas:
        d = _distancia_ao_segmento(xs, ys, p, q, estender=True)
        terra = d <= 2.5  # 5 m de estrada
        img[terra] = [168, 128, 92]
    return np.clip(img, 0, 255).astype(np.uint8)


def _distancia_ao_segmento(xs, ys, p, q, estender=False):
    dx, dy = q - p
    t = ((xs - p[0]) * dx + (ys - p[1]) * dy) / (dx * dx + dy * dy)
    t = np.clip(t, -0.15, 1.15) if estender else np.clip(t, 0, 1)
    return np.hypot(xs - (p[0] + t * dx), ys - (p[1] + t * dy))


@pytest.fixture(scope="module")
def imagem() -> np.ndarray:
    return _cafezal()


def test_desenho_torto_encosta_nos_carreadores(imagem) -> None:
    torto = CANTOS + np.array([[12, 9], [-10, 14], [-14, -11], [11, -13]])
    resultado = encaixa_nos_carreadores(imagem, torto, metros_por_pixel=MPP, faixa_m=30)
    assert resultado.confianca == 1.0
    assert all(lado.tipo == "carreador" for lado in resultado.lados)
    erro = np.linalg.norm(np.array(resultado.pontos) - CANTOS, axis=1)
    assert erro.max() < 4.0, erro  # cada canto a menos de 4 m da estrada de verdade
    assert np.linalg.norm(torto - CANTOS, axis=1).min() > 13  # e o desenho estava longe


def test_estrada_vizinha_paralela_nao_rouba_o_lado(imagem) -> None:
    """A estrada de cima do vizinho passa a ~10-15 m: o lado vai para a mais perto do traço."""
    torto = CANTOS + np.array([[0, 6], [0, 6], [0, 0], [0, 0]])
    resultado = encaixa_nos_carreadores(imagem, torto, metros_por_pixel=MPP, faixa_m=30)
    erro = np.linalg.norm(np.array(resultado.pontos)[:2] - CANTOS[:2], axis=1)
    assert erro.max() < 4.0


def test_sem_carreador_o_lado_fica_como_desenhado() -> None:
    liso = np.full((300, 300, 3), (50, 80, 40), dtype=np.uint8)
    desenho = np.array([[60.0, 60.0], [240.0, 60.0], [240.0, 240.0], [60.0, 240.0]])
    resultado = encaixa_nos_carreadores(liso, desenho, metros_por_pixel=MPP, faixa_m=20)
    assert resultado.confianca == 0.0
    assert np.allclose(resultado.pontos, desenho)


def test_linhas_do_miolo_dao_direcao_e_espacamento(imagem) -> None:
    linhas = direcao_das_linhas(imagem, CANTOS, metros_por_pixel=MPP)
    assert linhas is not None and linhas.forca > 8
    diferenca = abs((linhas.angulo_imagem_deg - ANGULO_RUAS + 90) % 180 - 90)
    assert diferenca < 3.0
    assert linhas.espacamento_m == pytest.approx(PERIODO_PX * MPP, rel=0.1)


def test_resolucao_que_nao_alcanca_a_rua_nao_inventa_direcao(imagem) -> None:
    assert direcao_das_linhas(imagem, CANTOS, metros_por_pixel=3.0) is None


def test_rota_inteira_em_graus(monkeypatch, imagem) -> None:
    from fastapi.testclient import TestClient

    from app.contorno_api import azimute_da_imagem
    from app.main import app

    recorte = basemap.Recorte(imagem=imagem, lon_min=-45.535, lat_min=-21.398,
                              lon_max=-45.535 + 0.0058, lat_max=-21.398 + 0.0054, zoom=18)
    monkeypatch.setattr(basemap, "buscar_area", lambda *a, **k: recorte)
    torto = CANTOS + np.array([[12, 9], [-10, 14], [-14, -11], [11, -13]])
    poligono = [list(recorte.para_grau(x, y)) for x, y in torto]
    r = TestClient(app).post("/encaixe-carreador", json={
        "polygon": poligono, "tolerance": 0.35, "tiles": "file:///nao-usado/{z}/{x}/{y}"})
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo["confidence"] == 1.0 and corpo["hit_limit"] == 0.0
    cantos_px = np.array([recorte.para_pixel(lon, lat) for lon, lat in corpo["polygon"]])
    assert np.linalg.norm(cantos_px - CANTOS, axis=1).max() < 4.0
    assert corpo["rows"]["detected"] is True
    esperado = azimute_da_imagem(ANGULO_RUAS)
    assert abs((corpo["rows"]["azimuth_deg"] - esperado + 90) % 180 - 90) < 3.0
    assert any("Encaixou" in nota for nota in corpo["notes"])


def test_azimute_da_imagem() -> None:
    from app.contorno_api import azimute_da_imagem

    assert azimute_da_imagem(0.0) == 90.0  # linha leste-oeste
    assert azimute_da_imagem(90.0) == 0.0  # norte-sul
    assert azimute_da_imagem(19.0) == 109.0  # desce para a direita: lés-sudeste
