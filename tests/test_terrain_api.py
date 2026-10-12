"""A rota de encaixe de contorno, ponta a ponta, sem rede.

Os tiles vêm de ``file://``: o módulo de basemap usa ``urllib``, que atende
esse esquema igual ao ``https``. Isso deixa o teste exercitar o caminho inteiro
— buscar imagem, montar mosaico, encaixar contorno, ler ruas, repartir terreno
e converter tudo de volta para graus — sem depender de provedor nenhum e sem
esbarrar na questão de licença que está documentada em :mod:`app.basemap`.

O cafezal sintético tem período de 32 px, que divide 256 — o lado do tile.
Isso é de propósito: com o padrão fechando na borda, o mosaico de tiles
repetidos vira uma lavoura contínua, e não uma grade de emendas que o detector
leria como rua.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

PERIODO = 32
"""Divide 256: o padrão fecha na borda do tile e o mosaico não tem emenda."""


@pytest.fixture(scope="module")
def tiles(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Um tile de cafezal, servido por file://, para qualquer z/x/y."""
    pasta = tmp_path_factory.mktemp("tiles")
    ys, xs = np.mgrid[0:256, 0:256]
    onda = 0.5 + 0.5 * np.cos(2 * math.pi * xs / PERIODO)
    rng = np.random.default_rng(4)
    img = np.dstack(
        [
            70 + 25 * onda + rng.normal(0, 4, (256, 256)),
            150 + 55 * onda + rng.normal(0, 5, (256, 256)),
            55 + 20 * onda + rng.normal(0, 4, (256, 256)),
        ]
    )
    caminho = pasta / "cafe.png"
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(caminho)
    # Um molde sem {z}/{x}/{y} é válido: format() devolve a mesma string, e
    # todos os tiles saem iguais — que é o cafezal contínuo que se quer aqui.
    return Path(caminho).as_uri()


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(app)


DESENHO = [
    [-45.8660, -21.9060],
    [-45.8590, -21.9055],
    [-45.8585, -21.9010],
    [-45.8655, -21.9015],
]


class TestSemFonteDeImagem:
    def test_sem_tiles_configurados_recusa_e_explica(self, cliente: TestClient) -> None:
        """Não há padrão silencioso: ver a nota de licença em app/basemap.py."""
        resposta = cliente.post("/refine-boundary", json={"polygon": DESENHO})
        assert resposta.status_code == 422
        assert "BASEMAP_TILES" in resposta.json()["detail"]

    def test_desenho_sem_area_e_recusado(self, cliente: TestClient, tiles: str) -> None:
        reta = [[-45.86, -21.90], [-45.85, -21.90], [-45.84, -21.90]]
        resposta = cliente.post("/refine-boundary", json={"polygon": reta, "tiles": tiles})
        assert resposta.status_code == 422


class TestEncaixe:
    def test_devolve_contorno_em_graus_perto_do_desenho(
        self, cliente: TestClient, tiles: str
    ) -> None:
        corpo = cliente.post(
            "/refine-boundary", json={"polygon": DESENHO, "tiles": tiles, "vertices": 32}
        ).json()

        assert len(corpo["polygon"]) == 32
        lons = [p[0] for p in corpo["polygon"]]
        lats = [p[1] for p in corpo["polygon"]]
        # O encaixe não pode sair do bairro do desenho: a tolerância é fração
        # do raio, então o resultado fica dentro do retângulo do traço com folga.
        assert -45.88 < min(lons) and max(lons) < -45.84
        assert -21.92 < min(lats) and max(lats) < -21.88
        assert 0.0 <= corpo["confidence"] <= 1.0
        assert corpo["meters_per_pixel"] > 0

    def test_le_as_ruas_do_cafezal(self, cliente: TestClient, tiles: str) -> None:
        """O tile tem rua de 32 px; a resposta tem que enxergá-la."""
        corpo = cliente.post(
            "/refine-boundary", json={"polygon": DESENHO, "tiles": tiles}
        ).json()

        ruas = corpo["rows"]
        assert ruas is not None
        assert ruas["detected"] is True
        assert ruas["spacing_m"] == pytest.approx(
            PERIODO * corpo["meters_per_pixel"], rel=0.25
        )


class TestTerreno:
    """A parte que o planejamento consome: área operável e objetos localizados."""

    def test_reparte_o_terreno_dentro_do_contorno(
        self, cliente: TestClient, tiles: str
    ) -> None:
        corpo = cliente.post(
            "/refine-boundary", json={"polygon": DESENHO, "tiles": tiles}
        ).json()

        terreno = corpo["terrain"]
        assert terreno is not None
        assert terreno["resolution_ok"] is True
        # Cafezal inteiro: quase tudo lavoura, e nada perto de "metade obstáculo".
        assert terreno["crop_fraction"] > 0.8
        assert terreno["obstacle_fraction"] < 0.2
        assert terreno["row_spacing_px"] == pytest.approx(PERIODO, abs=6)

    def test_cafezal_limpo_nao_inventa_objeto(
        self, cliente: TestClient, tiles: str
    ) -> None:
        corpo = cliente.post(
            "/refine-boundary", json={"polygon": DESENHO, "tiles": tiles}
        ).json()
        bloqueios = [o for o in corpo["terrain"]["objects"] if not o["traversable"]]
        assert bloqueios == [], "objeto inventado num cafezal sem obstáculo"

    def test_objeto_sai_em_graus_e_nao_em_pixel(
        self, cliente: TestClient, tiles: str
    ) -> None:
        """Quem consome é o mapa do console, que só fala grau."""
        corpo = cliente.post(
            "/refine-boundary", json={"polygon": DESENHO, "tiles": tiles}
        ).json()
        for objeto in corpo["terrain"]["objects"]:
            lon, lat = objeto["center"]
            assert -46.0 < lon < -45.0, "centro fora do bairro do talhão"
            assert -22.0 < lat < -21.0
            assert len(objeto["outline"]) >= 3
            assert objeto["area_ha"] > 0

    def test_tudo_marcado_como_heuristica(self, cliente: TestClient, tiles: str) -> None:
        """A classe de cada objeto é palpite de cor e textura, nunca medição."""
        corpo = cliente.post(
            "/refine-boundary", json={"polygon": DESENHO, "tiles": tiles}
        ).json()
        assert corpo["terrain"]["heuristic"] is True
        assert corpo["terrain"]["cover"]["heuristic"] is True
