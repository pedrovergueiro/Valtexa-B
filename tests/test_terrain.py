"""Leitura do terreno: o detector acha a rua que existe e recusa a que não existe.

A estratégia é a única que serve aqui: **plantar e colher**. Geramos uma lavoura
sintética com período e ângulo conhecidos e exigimos que o detector devolva
aqueles números. Um detector que não recupera o que foi plantado não vai acertar
numa foto de verdade — e, ao contrário de uma foto, o sintético tem gabarito.

Do outro lado, os casos negativos: ruído, gradiente e padrão fora da faixa de
resolução. São eles que protegem contra o erro caro — dizer "rua" para mata e
mandar o rover para dentro dela.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.terrain import (
    FORCA_MINIMA,
    LAVOURA,
    PERIODO_MIN_PX,
    RUIDO_FINO_MAXIMO,
    SOLO,
    detect_cover,
    detect_rows,
)


def lavoura(lado: int = 600, periodo: float = 24.0, angulo_deg: float = 30.0,
            ruido: float = 0.25, seed: int = 3) -> np.ndarray:
    """Linhas de plantio com período e ângulo conhecidos, sobre solo, com ruído."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:lado, 0:lado]
    a = math.radians(angulo_deg)
    # A onda varia ATRAVESSANDO as ruas: o eixo da variação é o perpendicular.
    t = xs * math.cos(a + math.pi / 2) + ys * math.sin(a + math.pi / 2)
    onda = 0.5 + 0.5 * np.cos(2 * math.pi * t / periodo)
    campo = np.clip(onda + rng.normal(0, ruido, onda.shape), 0, 1)
    verde = (60 + 150 * campo).astype(np.uint8)
    solo = (90 - 40 * campo).astype(np.uint8)
    return np.dstack([solo, verde, solo // 2]).astype(np.uint8)


def distancia_angular(a: float, b: float) -> float:
    """Direção é módulo 180: 179° e 1° distam 2°, não 178°."""
    bruta = abs(a - b) % 180.0
    return min(bruta, 180.0 - bruta)


class TestAchaOQueFoiPlantado:
    @pytest.mark.parametrize(
        ("periodo", "angulo"),
        [(24.0, 30.0), (18.0, 0.0), (32.0, 75.0), (14.0, 120.0), (40.0, 155.0)],
    )
    def test_recupera_periodo_e_angulo(self, periodo: float, angulo: float) -> None:
        lido = detect_rows(lavoura(periodo=periodo, angulo_deg=angulo))

        assert lido.confiavel, f"força {lido.forca} não passou de {FORCA_MINIMA}"
        assert lido.periodo_px == pytest.approx(periodo, abs=1.5)
        assert distancia_angular(lido.angulo_deg, angulo) < 4.0

    def test_converte_para_metros_quando_a_escala_e_conhecida(self) -> None:
        """Sem escala a resposta sai em pixels, e diz que está em pixels."""
        img = lavoura(periodo=30.0, angulo_deg=45.0)

        em_pixels = detect_rows(img)
        assert em_pixels.espacamento_m is None
        assert "px" in em_pixels.descricao

        # 10 cm por pixel: 30 px viram 3 m.
        em_metros = detect_rows(img, metros_por_pixel=0.10)
        assert em_metros.espacamento_m == pytest.approx(3.0, abs=0.2)
        assert " m," in em_metros.descricao

    def test_azimute_so_existe_com_a_rotacao_da_imagem(self) -> None:
        img = lavoura(periodo=24.0, angulo_deg=20.0)
        sem_norte = detect_rows(img)
        com_norte = detect_rows(img, norte_deg=90.0)

        assert distancia_angular(com_norte.azimute_deg, sem_norte.angulo_deg + 90.0) < 2.0

    def test_ruido_forte_nao_apaga_a_lavoura(self) -> None:
        """Foto de drone tem sombra, falha e mato na rua. O padrão tem que sobreviver."""
        lido = detect_rows(lavoura(periodo=26.0, angulo_deg=60.0, ruido=0.45))
        assert lido.confiavel
        assert lido.periodo_px == pytest.approx(26.0, abs=2.0)


class TestRecusaOQueNaoEhRua:
    """Os negativos. É aqui que mora o erro que manda o rover para dentro da mata."""

    def test_ruido_puro_nao_vira_rua(self) -> None:
        rng = np.random.default_rng(7)
        ruido = rng.normal(128, 40, (600, 600, 3)).clip(0, 255).astype(np.uint8)
        lido = detect_rows(ruido)
        assert not lido.confiavel

    def test_gradiente_de_iluminacao_nao_vira_rua(self) -> None:
        """Um lado do talhão mais claro que o outro é rampa, não periodicidade."""
        ys, xs = np.mgrid[0:600, 0:600]
        grad = np.dstack(
            [(xs / 3).astype(np.uint8), (100 + ys / 6).astype(np.uint8), (xs / 6).astype(np.uint8)]
        )
        assert not detect_rows(grad).confiavel

    def test_verde_uniforme_nao_vira_rua(self) -> None:
        pasto = np.dstack(
            [np.full((400, 400), 70, np.uint8), np.full((400, 400), 130, np.uint8),
             np.full((400, 400), 60, np.uint8)]
        )
        assert not detect_rows(pasto).confiavel

    def test_rua_fina_demais_para_a_resolucao_e_recusada(self) -> None:
        """O caso da foto comprimida: rua de 6 px, abaixo do piso de resolução.

        É a resposta certa e é desconfortável: a imagem **tem** lavoura, mas não
        tem pixel suficiente para provar. Dizer 'não sei' é melhor que devolver
        o quadriculado de 8 px do JPEG como espaçamento de plantio, que foi o
        que uma versão anterior deste módulo fazia.
        """
        fina = lavoura(periodo=PERIODO_MIN_PX - 4, angulo_deg=30.0, ruido=0.1)
        assert not detect_rows(fina).confiavel

    def test_recorte_pequeno_demais_e_erro_e_nao_palpite(self) -> None:
        with pytest.raises(ValueError, match="pequeno demais"):
            detect_rows(lavoura(lado=40))

    def test_imagem_em_tons_de_cinza_e_recusada(self) -> None:
        with pytest.raises(ValueError, match="três canais"):
            detect_rows(np.zeros((200, 200), dtype=np.uint8))


class TestPortaoDoRuidoFino:
    """O portão que separa rua de verdade de moiré de trama fina.

    Uma trama fina demais não some do espectro: ela bate com a grade de pixels e
    volta como um pico largo, forte e **estável** num período grande. Força não o
    denuncia — o moiré de uma rua de 6 px chega a 37, mais do que um cafezal
    cortado por carreador, que fica em 53. Consistência entre escalas também não:
    reduzir a imagem pela metade preserva o moiré tão bem quanto preserva a rua.

    O que o denuncia é a energia que ficou **acima** da faixa, no período
    verdadeiro. Estes testes seguram essa medida; sem ela, não existe limiar de
    força que separe os dois lados, e o detector volta a chamar de lavoura o que
    é gradiente de iluminação.
    """

    def test_trama_fina_e_barrada_mesmo_com_forca_alta(self) -> None:
        """Rua de 6 px: aparece como moiré de 21,7 px com força acima do piso.

        É o caso que obriga o portão a existir. Se alguém remover o
        RUIDO_FINO_MAXIMO, este teste falha — e falha dizendo por quê.
        """
        fina = lavoura(periodo=6.0, angulo_deg=30.0, ruido=0.1)
        lido = detect_rows(fina)

        assert lido.forca > FORCA_MINIMA, (
            "o corpus mudou: este caso valia como teste porque a força SOZINHA "
            "não o rejeitava"
        )
        assert not lido.confiavel, "o moiré passou por rua"

    def test_lavoura_de_verdade_tem_pouca_energia_acima_da_faixa(self) -> None:
        """O harmônico de uma rua real é fraco: um oitavo do fundamental, medido.

        É essa folga que deixa o portão fechar sem cortar lavoura legítima.
        """
        for periodo in (12.0, 18.0, 24.0, 40.0):
            lido = detect_rows(lavoura(periodo=periodo, angulo_deg=30.0))
            assert lido.confiavel, f"rua de {periodo} px foi barrada"

    def test_o_portao_tem_folga_dos_dois_lados(self) -> None:
        """A margem medida: 0,09 no pior positivo, 0,26 no melhor negativo."""
        assert 0.09 < RUIDO_FINO_MAXIMO < 0.26


class TestCobertura:
    def test_separa_lavoura_de_solo_exposto(self) -> None:
        """Metade plantada, metade terra: a divisão tem que aparecer."""
        plantado = lavoura(lado=384, periodo=24.0, angulo_deg=0.0, ruido=0.15)
        terra = np.dstack(
            [
                np.full((384, 384), 150, np.uint8),
                np.full((384, 384), 110, np.uint8),
                np.full((384, 384), 85, np.uint8),
            ]
        )
        junto = np.hstack([plantado, terra])

        cobertura = detect_cover(junto, bloco_px=48)
        assert cobertura.heuristica is True
        assert cobertura.fracoes[SOLO] > 0.3
        assert cobertura.fracoes[LAVOURA] > 0.2

        # A metade esquerda é a plantada; a direita, terra.
        metade = cobertura.classes.shape[1] // 2
        assert (cobertura.classes[:, :metade] == LAVOURA).mean() > 0.5
        assert (cobertura.classes[:, metade:] == SOLO).mean() > 0.8

    def test_fracoes_somam_um(self) -> None:
        cobertura = detect_cover(lavoura(lado=384), bloco_px=48)
        assert sum(cobertura.fracoes.values()) == pytest.approx(1.0)

    def test_obstaculo_conta_mata_e_o_que_nao_se_sabe(self) -> None:
        cobertura = detect_cover(lavoura(lado=384), bloco_px=48)
        assert cobertura.fracao_obstaculo == pytest.approx(
            cobertura.fracoes["mata"] + cobertura.fracoes["outro"]
        )

    def test_bloco_maior_que_a_imagem_e_erro(self) -> None:
        with pytest.raises(ValueError, match="pequena demais"):
            detect_cover(lavoura(lado=100), bloco_px=256)
