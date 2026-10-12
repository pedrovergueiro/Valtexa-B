"""A prescrição planta a planta: o que ela decide, o que recusa e o que promete."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.prescricao import (
    Contagio,
    Economia,
    PrescricaoImpossivelError,
    Produto,
    prescrever,
    resumo_em_texto,
)


@dataclass
class PlantaFalsa:
    """O mínimo que ``prescrever`` lê de um ``PlantSummary``."""

    plant_id: str
    row_id: str
    x_along_m: float
    posteriors: dict[str, float]
    lat: float | None = None
    lon: float | None = None


CANKER = Produto(
    nome="Oxicloreto de cobre",
    registro_mapa="00123",
    culturas=("citrus",),
    doencas=("Canker",),
    dose_ml_por_planta=200.0,
    dose_min_ml_por_planta=150.0,
    dose_max_ml_por_planta=300.0,
    preco_por_litro=40.0,
    eficacia=0.7,
    carencia_dias=7,
)
CONTAS = Economia(perda_por_planta=180.0, custo_operacional_por_planta=1.0)


def pomar(doentes: set[tuple[int, int]], ruas: int = 5, por_rua: int = 12) -> list[PlantaFalsa]:
    """Pomar retangular, 4 m entre plantas na rua; ``doentes`` são posições (rua, planta)."""
    return [
        PlantaFalsa(
            plant_id=f"{rua}-{i}",
            row_id=str(rua),
            x_along_m=i * 4.0,
            posteriors={"Canker": 0.95 if (rua, i) in doentes else 0.01},
        )
        for rua in range(ruas)
        for i in range(por_rua)
    ]


def prescricao(plantas, **extra):
    base = {
        "doenca": "Canker",
        "produto": CANKER,
        "economia": CONTAS,
        "crop": "citrus",
        "espacamento_entre_ruas_m": 6.0,
    }
    return prescrever(plantas, **{**base, **extra})


class TestADecisao:
    def test_a_planta_doente_e_tratada_e_a_distante_nao(self) -> None:
        r = prescricao(pomar({(2, 5)}))
        por_id = {p.plant_id: p for p in r.plantas}

        assert por_id["2-5"].tratar
        assert not por_id["0-0"].tratar
        assert por_id["0-0"].motivo == "sadia e longe do foco"

    def test_a_borda_em_volta_do_foco_nasce_sozinha(self) -> None:
        """Ninguém codificou 'trate duas plantas em volta': vem do alcance e do preço."""
        r = prescricao(pomar({(2, 5)}))
        tratadas = {p.plant_id for p in r.plantas if p.tratar}

        assert "2-5" in tratadas
        # Vizinhas na mesma rua entram; a planta sadia no outro canto, nao.
        assert {"2-4", "2-6"} <= tratadas
        assert "0-11" not in tratadas

    def test_alcance_maior_na_rua_faz_a_borda_crescer_na_rua(self) -> None:
        curto = prescricao(pomar({(2, 5)}), contagio=Contagio(1.0, 1.0))
        longo = prescricao(pomar({(2, 5)}), contagio=Contagio(12.0, 1.0))

        def na_rua_2(r):
            return sum(1 for p in r.plantas if p.tratar and p.row_id == "2")

        assert na_rua_2(longo) > na_rua_2(curto)

    def test_produto_mais_caro_trata_menos(self) -> None:
        """A borda encolhe quando o tratamento fica caro. É a conta, não um limiar."""
        import dataclasses

        caro = dataclasses.replace(CANKER, preco_por_litro=400.0)
        barato = prescricao(pomar({(2, 5)}))
        salgado = prescricao(pomar({(2, 5)}), produto=caro)

        assert salgado.plantas_tratadas < barato.plantas_tratadas

    def test_talhao_sadio_nao_gera_tratamento(self) -> None:
        r = prescricao(pomar(set()))

        assert r.plantas_tratadas == 0
        assert r.custo == 0.0
        assert any("Nenhuma planta a tratar" in a for a in r.avisos)
        # O aviso diz o porque com numero, para o agronomo nao achar que travou.
        assert any("custaria" in a for a in r.avisos)

    def test_duas_reboleiras_sao_tratadas_as_duas(self) -> None:
        r = prescricao(pomar({(0, 1), (4, 10)}))
        tratadas = {p.plant_id for p in r.plantas if p.tratar}

        assert "0-1" in tratadas
        assert "4-10" in tratadas

    def test_nao_trata_o_talhao_inteiro_por_causa_de_um_foco_pequeno(self) -> None:
        """A regressão que motivou o guloso.

        A primeira versão multiplicava o benefício de cada planta pelo número de
        vizinhas que ela infectaria. Isso conta o mesmo contágio uma vez por
        planta: num talhão de 1.000 com dois focos, ela mandava tratar as 1.000.
        Com ganho marginal, tratar a vizinha já desconta a pressão sobre esta.
        """
        grande = [
            PlantaFalsa(
                plant_id=f"{rua}-{i}",
                row_id=str(rua),
                x_along_m=i * 4.0,
                posteriors={"Canker": 0.95 if (rua, i) in {(4, 12), (15, 38)} else 0.001},
            )
            for rua in range(20)
            for i in range(50)
        ]
        r = prescricao(grande)

        assert r.plantas_no_talhao == 1000
        assert r.plantas_tratadas < 200
        # E mesmo tratando pouco, captura a maior parte do ganho de pulverizar tudo.
        assert r.fracao_do_ganho_possivel > 0.5


class TestAsContas:
    def test_o_que_poupa_e_a_diferenca_para_o_talhao_inteiro(self) -> None:
        r = prescricao(pomar({(2, 5)}))

        assert r.plantas_no_talhao == 60
        assert r.litros_talhao_inteiro == pytest.approx(60 * 0.2)
        assert r.litros == pytest.approx(r.plantas_tratadas * 0.2)
        assert r.custo_evitado == pytest.approx(r.custo_talhao_inteiro - r.custo)
        assert 0.0 < r.fracao_poupada < 1.0

    def test_o_orcamento_corta_e_o_relatorio_avisa(self) -> None:
        cheio = prescricao(pomar({(2, 5)}))
        curto = prescricao(
            pomar({(2, 5)}),
            economia=Economia(
                perda_por_planta=180.0,
                custo_operacional_por_planta=1.0,
                orcamento=2 * (CANKER.custo_por_planta + 1.0),
            ),
        )

        assert curto.plantas_tratadas == 2
        assert curto.plantas_tratadas < cheio.plantas_tratadas
        assert curto.limitado_por_orcamento

    def test_a_curva_comeca_sem_tratar_e_so_melhora(self) -> None:
        r = prescricao(pomar({(2, 5)}))
        curva = r.curva

        assert curva[0].plantas_tratadas == 0
        assert curva[0].custo == 0.0
        assert curva[0].doentes_esperados == pytest.approx(r.doentes_sem_nenhum_tratamento)
        assert curva[-1].plantas_tratadas == r.plantas_tratadas
        assert curva[-1].doentes_esperados == pytest.approx(r.doentes_com_a_prescricao)
        # Tratar mais nunca aumenta a doenca esperada, e cada planta rende menos que
        # a anterior: o guloso pega primeiro o que mais compensa.
        esperados = [ponto.doentes_esperados for ponto in curva]
        assert esperados == sorted(esperados, reverse=True)

    def test_cada_planta_rende_menos_que_a_anterior(self) -> None:
        """Retorno decrescente: é o que faz a curva achatar e dá onde parar."""
        r = prescricao(pomar({(2, 5)}), pontos_na_curva=100)
        ganhos = [
            a.doentes_esperados - b.doentes_esperados
            for a, b in zip(r.curva, r.curva[1:], strict=False)
        ]

        assert all(g >= -1e-9 for g in ganhos)
        assert ganhos[0] > ganhos[-1]

    def test_a_prescricao_fica_entre_nao_fazer_nada_e_pulverizar_tudo(self) -> None:
        r = prescricao(pomar({(2, 5)}))

        assert r.doentes_tratando_tudo < r.doentes_com_a_prescricao
        assert r.doentes_com_a_prescricao < r.doentes_sem_nenhum_tratamento
        assert 0.0 < r.fracao_do_ganho_possivel <= 1.0

    def test_o_risco_final_de_cada_planta_soma_o_total(self) -> None:
        r = prescricao(pomar({(2, 5)}))
        de_fora = sum(p.risco_final for p in r.plantas)

        assert r.doentes_com_a_prescricao == pytest.approx(de_fora)


class TestOQueEleRecusa:
    def test_produto_sem_registro_no_mapa(self) -> None:
        with pytest.raises(PrescricaoImpossivelError, match="registro no MAPA"):
            Produto(
                nome="X",
                registro_mapa="  ",
                culturas=("citrus",),
                doencas=("Canker",),
                dose_ml_por_planta=200.0,
                dose_min_ml_por_planta=150.0,
                dose_max_ml_por_planta=300.0,
                preco_por_litro=40.0,
                eficacia=0.7,
                carencia_dias=7,
            )

    def test_cultura_fora_da_bula(self) -> None:
        with pytest.raises(PrescricaoImpossivelError, match="não é autorizado em 'banana'"):
            prescricao(pomar({(2, 5)}), crop="banana")

    def test_alvo_fora_da_bula(self) -> None:
        with pytest.raises(PrescricaoImpossivelError, match="alvo autorizado"):
            prescricao(pomar({(2, 5)}), doenca="Greening")

    def test_colheita_dentro_da_carencia(self) -> None:
        with pytest.raises(PrescricaoImpossivelError, match="carência de 7 dias"):
            prescricao(pomar({(2, 5)}), dias_ate_colheita=3)

    def test_dose_fora_da_faixa_de_bula(self) -> None:
        import dataclasses

        with pytest.raises(PrescricaoImpossivelError, match="fora da faixa de bula"):
            prescricao(
                pomar({(2, 5)}), produto=dataclasses.replace(CANKER, dose_ml_por_planta=900.0)
            )

    def test_sem_perda_por_planta_nao_ha_o_que_comparar(self) -> None:
        with pytest.raises(PrescricaoImpossivelError, match="perda_por_planta"):
            Economia(perda_por_planta=0.0)

    def test_talhao_sem_plantas(self) -> None:
        with pytest.raises(PrescricaoImpossivelError, match="sem plantas"):
            prescricao([])

    def test_posterior_nao_finita(self) -> None:
        plantas = pomar(set())
        plantas[0].posteriors["Canker"] = float("nan")

        with pytest.raises(PrescricaoImpossivelError, match="não finita"):
            prescricao(plantas)


class TestOQueEleEntrega:
    def test_o_geojson_traz_so_quem_tem_coordenada(self) -> None:
        plantas = pomar({(2, 5)})
        plantas[0].lat, plantas[0].lon = -21.9, -45.8
        geo = prescricao(plantas).to_geojson()

        assert geo["type"] == "FeatureCollection"
        assert len(geo["features"]) == 1
        assert geo["features"][0]["geometry"]["coordinates"] == [-45.8, -21.9]
        assert "tratar" in geo["features"][0]["properties"]
        assert "agrônomo" in geo["properties"]["aviso"]

    def test_o_resumo_diz_o_risco_e_que_e_recomendacao(self) -> None:
        texto = resumo_em_texto(prescricao(pomar({(2, 5)}), field_id="T07"))

        assert "T07" in texto
        assert "sem fazer nada" in texto
        assert "pulverizando tudo" in texto
        assert "aprovação do agrônomo" in texto

    def test_avisa_quando_o_contagio_nao_foi_ajustado(self) -> None:
        r = prescricao(pomar({(2, 5)}))

        assert not r.contagio_ajustado
        assert any("não ajustado" in a for a in r.avisos)

    def test_contagio_ajustado_nao_gera_o_aviso(self) -> None:
        r = prescricao(pomar({(2, 5)}), contagio=Contagio(3.0, 1.5, ajustado=True))

        assert r.contagio_ajustado
        assert not any("não ajustado" in a for a in r.avisos)
