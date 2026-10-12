"""Encaixe de contorno: o desenho torto tem que virar o limite certo.

A verificação é a mesma ideia do detector de ruas — **plantar e colher**. A cena
sintética tem um talhão de raio conhecido; o desenho é gerado torto de
propósito; e o que se exige é que o erro radial **caia**, não que fique bonito.

Os testes de recusa valem tanto quanto os de acerto. Um encaixe que sempre
devolve alguma coisa é pior que nenhum: numa imagem sem borda ele entregaria um
limite inventado com cara de medido, e é esse polígono que viraria cerca virtual.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.boundary import refine_boundary


def cena(lado: int = 600, raio: int = 180, seed: int = 5) -> np.ndarray:
    """Talhão de café — verde claro e listrado — cercado de mata escura."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:lado, 0:lado]
    centro = lado / 2
    dentro = (xs - centro) ** 2 + (ys - centro) ** 2 < raio**2

    img = np.dstack(
        [
            rng.normal(45, 14, (lado, lado)),
            rng.normal(80, 18, (lado, lado)),
            rng.normal(38, 12, (lado, lado)),
        ]
    )
    linhas = 0.5 + 0.5 * np.cos(
        2 * math.pi * (xs * math.cos(0.4) + ys * math.sin(0.4)) / 24
    )
    cafe = np.dstack(
        [
            70 + 25 * linhas + rng.normal(0, 6, (lado, lado)),
            150 + 55 * linhas + rng.normal(0, 7, (lado, lado)),
            55 + 20 * linhas + rng.normal(0, 6, (lado, lado)),
        ]
    )
    img[dentro] = cafe[dentro]
    return np.clip(img, 0, 255).astype(np.uint8)


def desenho_torto(
    centro: tuple[float, float], raio: float, n: int = 9, bagunca: float = 0.30, seed: int = 11
) -> np.ndarray:
    """O que uma pessoa marca a dedo: poucos pontos e raio irregular."""
    rng = np.random.default_rng(seed)
    cx, cy = centro
    ang = np.sort(rng.uniform(0, 2 * math.pi, n))
    raios = raio * (1 + rng.uniform(-bagunca, bagunca, n))
    return np.column_stack([cx + np.cos(ang) * raios, cy + np.sin(ang) * raios])


def erro_radial(pontos: np.ndarray, centro: tuple[float, float], raio: float) -> float:
    cx, cy = centro
    return float(np.abs(np.hypot(pontos[:, 0] - cx, pontos[:, 1] - cy) - raio).mean())


CENTRO = (300.0, 300.0)
RAIO = 180.0


class TestEncaixa:
    @pytest.mark.parametrize("bagunca", [0.15, 0.30, 0.45])
    def test_o_erro_do_desenho_cai(self, bagunca: float) -> None:
        """O que se exige não é perfeição: é que o encaixe melhore o traço."""
        rascunho = desenho_torto(CENTRO, RAIO, bagunca=bagunca)
        antes = erro_radial(rascunho, CENTRO, RAIO)

        contorno = refine_boundary(cena(), rascunho)
        depois = erro_radial(contorno.pontos, CENTRO, RAIO)

        assert depois < antes / 3, f"antes {antes:.1f}px, depois {depois:.1f}px"
        assert depois < 6.0

    def test_borda_nitida_e_reconhecida_como_nitida(self) -> None:
        contorno = refine_boundary(cena(), desenho_torto(CENTRO, RAIO, bagunca=0.2))
        assert contorno.confianca > 0.9
        assert contorno.encostou_no_limite < 0.1

    def test_funciona_com_desenho_por_dentro_e_por_fora(self) -> None:
        """Apertado ou folgado, o encaixe converge para a mesma borda."""
        apertado = refine_boundary(cena(), desenho_torto(CENTRO, RAIO * 0.80, n=10, seed=4))
        folgado = refine_boundary(cena(), desenho_torto(CENTRO, RAIO * 1.20, n=10, seed=4))

        assert erro_radial(apertado.pontos, CENTRO, RAIO) < 8.0
        assert erro_radial(folgado.pontos, CENTRO, RAIO) < 8.0

    def test_devolve_poligono_simples_e_fechado(self) -> None:
        """A cerca virtual e o planejamento só sabem lidar com anel simples."""
        contorno = refine_boundary(cena(), desenho_torto(CENTRO, RAIO))
        assert len(contorno.pontos) >= 8
        assert contorno.pontos.shape[1] == 2
        assert contorno.area_px > 0
        # Sem vértice repetido no fecho: o anel fecha implicitamente.
        assert not np.allclose(contorno.pontos[0], contorno.pontos[-1])


class TestNaoFoge:
    def test_tolerancia_apertada_prende_o_resultado_no_traco(self) -> None:
        """Quem desenhou com cuidado não quer a IA reinterpretando o traço."""
        rascunho = desenho_torto(CENTRO, RAIO, bagunca=0.20)
        preso = refine_boundary(cena(), rascunho, tolerancia=0.03)

        cx, cy = 300.0, 300.0
        raio_preso = np.hypot(preso.pontos[:, 0] - cx, preso.pontos[:, 1] - cy)
        raio_traco = np.hypot(rascunho[:, 0] - cx, rascunho[:, 1] - cy)
        assert raio_preso.min() > raio_traco.min() * 0.90
        assert raio_preso.max() < raio_traco.max() * 1.10

    def test_nao_engole_o_talhao_vizinho(self) -> None:
        """Duas lavouras iguais lado a lado: o encaixe fica na que foi marcada."""
        base = cena(lado=700, raio=140)
        vizinho = cena(lado=700, raio=140)
        # Cola a segunda lavoura deslocada, criando duas manchas parecidas.
        base[:, 350:] = np.roll(vizinho, 0, axis=0)[:, :350]

        contorno = refine_boundary(base, desenho_torto((350.0, 350.0), 140.0, bagunca=0.2))
        largura = contorno.pontos[:, 0].max() - contorno.pontos[:, 0].min()
        assert largura < 140 * 2 * 1.5


class TestRecusa:
    def test_campo_uniforme_avisa_que_nao_achou_borda(self) -> None:
        """Sem borda, a resposta honesta é 'parei onde você mandou'."""
        rng = np.random.default_rng(3)
        liso = np.clip(
            np.dstack(
                [
                    rng.normal(70, 8, (500, 500)),
                    rng.normal(140, 8, (500, 500)),
                    rng.normal(60, 8, (500, 500)),
                ]
            ),
            0,
            255,
        ).astype(np.uint8)

        contorno = refine_boundary(liso, desenho_torto((250.0, 250.0), 150.0, bagunca=0.2))
        assert contorno.confianca < 0.2
        assert contorno.encostou_no_limite > 0.8

    def test_poucos_pontos_e_erro(self) -> None:
        with pytest.raises(ValueError, match="três pontos"):
            refine_boundary(cena(), [[10.0, 10.0], [20.0, 20.0]])

    def test_imagem_sem_cor_e_erro(self) -> None:
        with pytest.raises(ValueError, match="três canais"):
            refine_boundary(np.zeros((300, 300), dtype=np.uint8), desenho_torto(CENTRO, RAIO))

    def test_desenho_minusculo_e_erro(self) -> None:
        with pytest.raises(ValueError, match="pequeno demais"):
            refine_boundary(cena(), [[300.0, 300.0], [301.0, 300.0], [300.0, 301.0]])


def test_variacao_de_area_e_informada() -> None:
    """O número que a pessoa confere: o quanto a IA mexeu no que ela desenhou."""
    contorno = refine_boundary(cena(), desenho_torto(CENTRO, RAIO, bagunca=0.3))
    assert contorno.area_desenhada_px > 0
    assert contorno.variacao == pytest.approx(
        contorno.area_px / contorno.area_desenhada_px - 1.0
    )
    assert "área" in contorno.descricao
