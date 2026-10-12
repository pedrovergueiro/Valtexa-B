"""Objetos dentro do talhão: achar o que atrapalha, e não inventar o que não há.

A cena sintética tem gabarito: uma moita de mata de posição e raio conhecidos,
dentro de um cafezal de espaçamento conhecido. O que se exige é que o objeto
apareça perto de onde foi plantado, e que um cafezal limpo não produza objeto
nenhum.

Os dois limites do método estão fixados em teste de propósito — resolução
insuficiente e objeto menor que a janela. São eles que impedem a conta de área
operável de mentir, e um deles some em silêncio se ninguém o segurar.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.obstacles import CARREADOR, MATA_INTERNA, area_operavel_ha, detect_obstacles

LADO = 768
PERIODO = 24.0
MATA_CENTRO = (560, 200)
MATA_RAIO = 110


def cafezal(
    lado: int = LADO,
    *,
    com_mata: bool = True,
    carreador_px: int = 0,
    seed: int = 5,
) -> np.ndarray:
    """Cafezal em linhas, com uma moita de mata e um carreador opcionais."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:lado, 0:lado]

    linhas = 0.5 + 0.5 * np.cos(
        2 * math.pi * (xs * math.cos(0.3) + ys * math.sin(0.3)) / PERIODO
    )
    img = np.dstack(
        [
            70 + 25 * linhas + rng.normal(0, 6, (lado, lado)),
            150 + 55 * linhas + rng.normal(0, 7, (lado, lado)),
            55 + 20 * linhas + rng.normal(0, 6, (lado, lado)),
        ]
    )

    if com_mata:
        cx, cy = MATA_CENTRO
        mata = (xs - cx) ** 2 + (ys - cy) ** 2 < MATA_RAIO**2
        escuro = np.dstack(
            [
                rng.normal(42, 15, (lado, lado)),
                rng.normal(76, 20, (lado, lado)),
                rng.normal(36, 13, (lado, lado)),
            ]
        )
        img[mata] = escuro[mata]

    if carreador_px:
        faixa = (ys > 500) & (ys < 500 + carreador_px)
        terra = np.dstack(
            [
                rng.normal(155, 10, (lado, lado)),
                rng.normal(115, 10, (lado, lado)),
                rng.normal(88, 9, (lado, lado)),
            ]
        )
        img[faixa] = terra[faixa]

    return np.clip(img, 0, 255).astype(np.uint8)


CONTORNO = [[20, 20], [LADO - 28, 20], [LADO - 28, LADO - 28], [20, LADO - 28]]


class TestAcha:
    def test_localiza_a_moita_de_mata(self) -> None:
        terreno = detect_obstacles(cafezal(), CONTORNO)
        matas = [o for o in terreno.objetos if o.tipo == MATA_INTERNA]

        assert matas, "a moita de mata não foi localizada"
        achada = matas[0]
        erro = math.hypot(
            achada.centro_px[0] - MATA_CENTRO[0], achada.centro_px[1] - MATA_CENTRO[1]
        )
        # A precisão de posição é da ordem de uma janela, e a janela sai do
        # espaçamento das ruas. Exigir mais seria exigir do método o que ele
        # não tem para dar.
        assert erro < terreno.cobertura.bloco_px, f"erro de {erro:.0f} px"
        assert not achada.atravessavel

    def test_a_mata_nao_engole_a_lavoura(self) -> None:
        """A moita ocupa ~7% do talhão. Se o obstáculo vier muito acima disso,
        a classificação está comendo cafezal — e a área operável sai errada."""
        terreno = detect_obstacles(cafezal(), CONTORNO)
        assert terreno.fracao_lavoura > 0.85
        assert terreno.fracao_obstaculo < 0.15

    def test_area_operavel_desconta_o_obstaculo(self) -> None:
        terreno = detect_obstacles(cafezal(), CONTORNO)
        operavel = area_operavel_ha(terreno, 24.2)
        assert 0 < operavel < 24.2
        # A tolerância é o arredondamento declarado de area_operavel_ha:
        # 0,0001 ha, que é um metro quadrado.
        assert operavel == pytest.approx(
            24.2 * (terreno.fracao_lavoura + terreno.fracao_caminho), abs=1e-4
        )

    def test_janela_vem_do_espacamento_das_ruas(self) -> None:
        """Quatro períodos por janela: é o que a análise de frequência precisa."""
        terreno = detect_obstacles(cafezal(), CONTORNO)
        assert terreno.periodo_px == pytest.approx(PERIODO, abs=3)
        assert terreno.cobertura.bloco_px == pytest.approx(PERIODO * 4, abs=8)

    def test_carreador_largo_e_reconhecido_como_caminho(self) -> None:
        """Comprido e de solo exposto: o rover atravessa por ele."""
        terreno = detect_obstacles(cafezal(carreador_px=160), CONTORNO)
        caminhos = [o for o in terreno.objetos if o.atravessavel]
        assert caminhos, "o carreador largo não apareceu"
        assert any(o.tipo == CARREADOR for o in caminhos)
        assert terreno.fracao_caminho > 0.05


class TestNaoInventa:
    def test_cafezal_limpo_nao_produz_objeto(self) -> None:
        terreno = detect_obstacles(cafezal(com_mata=False), CONTORNO)
        assert terreno.objetos == []
        assert terreno.fracao_lavoura > 0.95
        assert terreno.fracao_obstaculo < 0.05

    def test_resolucao_insuficiente_nao_vira_obstaculo(self) -> None:
        """O erro caro: numa imagem em que a rua não se resolve, **nada** é
        lavoura para o classificador. Sem esta recusa, o resultado seria "75% de
        obstáculo" num talhão que é cafezal inteiro — e a área operável cairia
        pela metade sem ninguém desconfiar.
        """
        # Ruas de 3 px: abaixo do piso de resolução do detector.
        fino = cafezal(lado=400, com_mata=False)
        apertado = fino[::8, ::8]  # reduz a imagem: a rua some
        terreno = detect_obstacles(np.repeat(np.repeat(apertado, 2, 0), 2, 1))

        assert terreno.resolucao_suficiente is False
        assert terreno.objetos == []
        assert terreno.fracao_obstaculo == 0.0
        assert "ortomosaico" in terreno.descricao

    def test_objeto_menor_que_a_janela_nao_e_localizado(self) -> None:
        """Limite declarado: um carreador de 6 m sob uma janela de 14 m não aparece.

        Está em teste para ser um limite **conhecido**, e não uma falha que
        alguém descobre no campo. Achar faixa estreita pede detector de feição
        linear, que é outro algoritmo.
        """
        terreno = detect_obstacles(cafezal(com_mata=False, carreador_px=32), CONTORNO)
        assert not [o for o in terreno.objetos if o.tipo == CARREADOR]

    def test_contorno_limita_a_analise_ao_talhao(self) -> None:
        """Sem o contorno, a mata de fora entraria na conta do talhão."""
        canto = [[20, 20], [300, 20], [300, 300], [20, 300]]
        com_limite = detect_obstacles(cafezal(), canto)
        sem_limite = detect_obstacles(cafezal())
        assert com_limite.fracao_obstaculo < sem_limite.fracao_obstaculo


class TestRecusas:
    def test_contorno_fora_da_imagem_e_erro(self) -> None:
        with pytest.raises(ValueError, match="bloco nenhum"):
            detect_obstacles(cafezal(), [[5000, 5000], [5100, 5000], [5100, 5100]])

    def test_imagem_sem_cor_e_erro(self) -> None:
        with pytest.raises(ValueError, match="três canais"):
            detect_obstacles(np.zeros((400, 400), dtype=np.uint8))


def test_tudo_sai_marcado_como_heuristica() -> None:
    """A classe de cada objeto é palpite de cor e textura, nunca medição."""
    terreno = detect_obstacles(cafezal(), CONTORNO)
    assert terreno.heuristica is True
    assert terreno.cobertura.heuristica is True
    assert all(0.0 <= o.confianca <= 1.0 for o in terreno.objetos)
