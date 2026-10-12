"""Talhão: quadros do rover viram plantas, mapa, evolução e prevalência."""

from __future__ import annotations

import json

import pytest

from app.fields_api import fields, training_prior_path
from app.registry import get_spec, registry
from tests.conftest import FakeKerasModel, make_image_bytes
from tests.test_plant_fusion import calibration

BANANA = "banana-foliar"
CLASSES = get_spec(BANANA).class_names


def _attach() -> None:
    """Dublê que responde Sigatoka (a última classe) em todo recorte."""
    registry.get(BANANA).attach(FakeKerasModel(num_classes=len(CLASSES)), CLASSES)


@pytest.fixture
def field_dir(tmp_path):
    fields.reset(tmp_path / "talhoes")
    yield tmp_path / "talhoes"
    fields.reset(None)


def _frame(client, **params):
    query = {
        "model": BANANA,
        "camera_height_mm": 1000,
        "distance_mm": 270,
        "field_id": "T01",
        "row_id": "L1",
        "side": "left",
        "x_m": 10.0,
        "pass_id": "P1",
        **params,
    }
    # O cliente HTTP mandaria None como texto vazio: parâmetro nulo = parâmetro ausente.
    query = {key: value for key, value in query.items() if value is not None}
    return client.post(
        "/predict/rover-frame",
        params=query,
        files={"file": ("q.jpg", make_image_bytes(size=(1000, 750)), "image/jpeg")},
    )


class TestRoverFeedsTheField:
    def test_a_frame_lands_on_a_plant(self, client, field_dir):
        _attach()
        body = _frame(client).json()

        assert body["plants"], "o quadro tem de alimentar ao menos uma planta"
        assert client.get("/fields").json() == {"fields": ["T01"]}

    def test_needs_the_pose_with_the_field(self, client, field_dir):
        _attach()
        response = _frame(client, row_id=None, x_m=None)
        assert response.status_code == 422
        assert "row_id" in response.json()["detail"] and "x_m" in response.json()["detail"]

    def test_is_off_without_fields_dir(self, client):
        fields.reset(None)
        _attach()
        assert _frame(client).status_code == 503

    def test_the_plant_is_dated_by_the_capture_not_the_upload(self, client, field_dir):
        # O rover sem enlace sincroniza horas depois: o histórico da planta
        # (e a AUDPC) tem de usar o instante da foto.
        _attach()
        _frame(client, captured_at="2026-09-20T13:45:00")

        plant = client.get("/fields/T01/plants").json()[0]
        assert plant["time"].startswith("2026-09-20T13:45:00")

    def test_a_nan_position_is_refused(self, client, field_dir):
        _attach()
        assert _frame(client, x_m="nan").status_code == 422

    def test_without_field_the_frame_is_answered_alone(self, client, field_dir):
        _attach()
        body = _frame(client, field_id=None).json()
        assert body["plants"] is None
        assert client.get("/fields").json() == {"fields": []}

    def test_a_frame_for_a_closed_pass_is_a_conflict(self, client, field_dir):
        _attach()
        _frame(client)
        client.post("/fields/T01/passes/P1/close")

        response = _frame(client)

        assert response.status_code == 409
        assert "fechada" in response.json()["detail"]


class TestPlantsAndMap:
    def test_plants_carry_a_fused_diagnosis(self, client, field_dir):
        _attach()
        _frame(client)
        _frame(client, camera_height_mm=450, camera_id="esq-450")

        plants = client.get("/fields/T01/plants").json()

        assert plants
        assert {"status", "posteriors", "incidence", "n_cameras"} <= set(plants[0])
        assert plants[0]["primary_disease"] == "Sigatoka"

    def test_map_is_geojson(self, client, field_dir):
        _attach()
        _frame(client, lat=-22.25, lon=-45.70, heading_deg=90)

        response = client.get("/fields/T01/map")

        assert response.headers["content-type"].startswith("application/geo+json")
        geojson = response.json()
        assert geojson["type"] == "FeatureCollection"
        assert geojson["features"][0]["geometry"]["type"] == "Point"

    def test_closing_a_pass_reports_changes(self, client, field_dir):
        _attach()
        _frame(client)
        body = client.post("/fields/T01/passes/P1/close").json()
        assert body["plants"] and "changes" in body

    def test_unknown_field_is_404(self, client, field_dir):
        assert client.get("/fields/nada/plants").status_code == 404


class TestFieldPrevalence:
    def test_needs_the_training_prior(self, client, field_dir):
        _attach()
        _frame(client)
        assert client.get("/fields/T01/prevalence", params={"model": BANANA}).status_code == 409

    def test_a_field_with_few_tiles_is_not_called_unstable(
        self, client, field_dir, tmp_path, monkeypatch
    ):
        """Sem recortes suficientes ainda vale o prior do treino: nem confiável nem instável."""
        from app.config import settings

        monkeypatch.setattr(settings, "models_dir", tmp_path)
        prior = {name: 1 / len(CLASSES) for name in CLASSES}
        training_prior_path(get_spec(BANANA)).write_text(json.dumps({"prior": prior}))
        _attach()
        _frame(client, field_id="T-POUCO")  # cria o talhão; um quadro pequeno não basta

        answer = client.get("/fields/NAO-VISTO/prevalence", params={"model": BANANA}).json()

        assert answer["divergence"] is None
        assert answer["reliable"] is None
        assert "Poucos recortes" in answer["note"]
        assert answer["predicted_shares"] is None

    def test_follows_the_field_with_the_training_prior(
        self, client, field_dir, tmp_path, monkeypatch
    ):
        from app.config import settings

        monkeypatch.setattr(settings, "models_dir", tmp_path)
        prior = {name: 1 / len(CLASSES) for name in CLASSES}
        training_prior_path(get_spec(BANANA)).write_text(json.dumps({"prior": prior}))
        _attach()

        body = _frame(client).json()
        answer = client.get("/fields/T01/prevalence", params={"model": BANANA}).json()

        assert set(body["field_prevalence"]) == set(CLASSES)
        assert answer["training_prior"]["Sigatoka"] == pytest.approx(0.2)
        assert sum(answer["field_prior"].values()) == pytest.approx(1.0)
        # Dublê confiante: EM e contagem concordam, a estimativa é dada como confiável.
        assert answer["divergence"] is not None
        assert answer["reliable"] is True
        assert answer["note"] is None
        # E a fração de PLANTAS vem junto, a mesma da rota de incidência.
        assert answer["plant_incidence"] == client.get("/fields/T01/incidence").json()


def _install_calibration(tmp_path, monkeypatch, model_key: str = BANANA):
    from app.config import settings

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    cal = calibration(model_key=model_key)
    get_spec(BANANA).fusion_path.write_text(
        json.dumps({"version": 1, "calibration": cal.to_dict()}), encoding="utf-8"
    )
    return cal


class TestFieldIncidence:
    def test_counts_plants_without_calibration(self, client, field_dir):
        _attach()
        _frame(client)

        body = client.get("/fields/T01/incidence").json()

        assert body["n_plants"] >= 1
        assert body["method"] == "aparente"
        assert body["incidence"] is None
        assert body["apparent"] is not None
        assert body["calibration_id"] is None

    def test_uses_the_model_calibration(self, client, field_dir, tmp_path, monkeypatch):
        cal = _install_calibration(tmp_path, monkeypatch)
        _attach()
        _frame(client)

        body = client.get("/fields/T01/incidence").json()
        plants = client.get("/fields/T01/plants").json()

        assert body["method"] == "rogan_gladen"
        assert body["calibration_id"] == cal.id
        assert body["ci95"][0] <= body["incidence"] <= body["ci95"][1]
        assert {plant["calibration_id"] for plant in plants} == {cal.id}

    def test_a_calibration_for_another_model_is_ignored(
        self, client, field_dir, tmp_path, monkeypatch
    ):
        _install_calibration(tmp_path, monkeypatch, model_key="citrus-foliar")
        _attach()
        _frame(client)

        assert client.get("/fields/T01/incidence").json()["calibration_id"] is None

    def test_unknown_field_or_pass_is_404(self, client, field_dir):
        assert client.get("/fields/nada/incidence").status_code == 404
        _attach()
        _frame(client)
        assert client.get("/fields/T01/incidence", params={"pass_id": "X"}).status_code == 404


class TestSigatokaInTheFrame:
    def test_banana_frames_carry_the_severity(self, client, field_dir):
        _attach()
        severity = _frame(client, field_id=None).json()["sigatoka"]

        assert severity["scale_validated"] is False
        assert severity["n_crops_total"] > 0
        assert severity["dsi"] is None or 0 <= severity["dsi"] <= 100


PRODUTO = {
    "nome": "Mancozebe",
    "registro_mapa": "00987",
    "culturas": ["banana"],
    "doencas": ["Sigatoka"],
    "dose_ml_por_planta": 120.0,
    "dose_min_ml_por_planta": 100.0,
    "dose_max_ml_por_planta": 200.0,
    "preco_por_litro": 38.0,
    "eficacia": 0.7,
    "carencia_dias": 14,
}
PEDIDO = {
    "doenca": "Sigatoka",
    "produto": PRODUTO,
    "economia": {"perda_por_planta": 150.0, "custo_operacional_por_planta": 1.0},
}


class TestPrescricao:
    """A prescrição por planta pela rota HTTP."""

    def _plantas(self, client):
        _attach()
        for x in (2.0, 4.0, 6.0):
            _frame(client, x_m=x)

    def test_devolve_o_que_tratar_e_o_que_poupa(self, client, field_dir):
        self._plantas(client)

        body = client.post("/fields/T01/prescription", json=PEDIDO).json()

        assert body["plantas_tratadas"] <= body["plantas_no_talhao"]
        assert body["doentes_com_a_prescricao"] <= body["doentes_sem_nenhum_tratamento"]
        assert "aprovação do agrônomo" in body["resumo"]
        assert body["custo_talhao_inteiro"] >= body["custo"]

    def test_o_geojson_abre_num_sig(self, client, field_dir):
        self._plantas(client)

        response = client.post("/fields/T01/prescription?formato=geojson", json=PEDIDO)

        assert response.headers["content-type"].startswith("application/geo+json")
        assert response.json()["type"] == "FeatureCollection"

    def test_recusa_produto_fora_da_bula_da_cultura(self, client, field_dir):
        self._plantas(client)
        pedido = {**PEDIDO, "produto": {**PRODUTO, "culturas": ["citrus"]}}

        response = client.post("/fields/T01/prescription", json=pedido)

        assert response.status_code == 422
        assert "não é autorizado em 'banana'" in response.json()["detail"]

    def test_recusa_aplicar_dentro_da_carencia(self, client, field_dir):
        self._plantas(client)

        response = client.post("/fields/T01/prescription", json={**PEDIDO, "dias_ate_colheita": 3})

        assert response.status_code == 422
        assert "carência de 14 dias" in response.json()["detail"]

    def test_recusa_preco_nan(self, client, field_dir):
        """NaN passa por toda comparação como falso: um preço NaN faria a conta inteira
        virar NaN sem nada estourar. O corpo vai cru porque o encoder padrão recusa NaN."""
        self._plantas(client)
        pedido = {**PEDIDO, "produto": {**PRODUTO, "preco_por_litro": float("nan")}}

        response = client.post(
            "/fields/T01/prescription",
            content=json.dumps(pedido),
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422

    def test_talhao_desconhecido(self, client, field_dir):
        assert client.post("/fields/NAO-EXISTE/prescription", json=PEDIDO).status_code == 404
