"""``POST /fields/{id}/frames``: o quadro que o Rover já classificou entra na fusão por planta."""

from __future__ import annotations

import json

import pytest

from app.fields_api import fields
from app.registry import get_spec
from tests.test_plant_fusion import calibration

BANANA = "banana-foliar"
CLASSES = get_spec(BANANA).class_names


@pytest.fixture
def field_dir(tmp_path):
    fields.reset(tmp_path / "talhoes")
    yield tmp_path / "talhoes"
    fields.reset(None)


def _tile(x0: float, doenca: float) -> dict:
    probs = dict.fromkeys(CLASSES, 0.0)
    probs["Sigatoka"], probs["Healthy"] = doenca, 1.0 - doenca
    return {"box": [x0, 400, x0 + 220, 620], "predicted_class": max(probs, key=probs.get),
            "probabilities": probs, "confidence": max(probs.values()), "height_mm": 1000.0,
            "zones": ["folhas_baixas"]}


def _quadro(frame_id: str = "r-1", **extra) -> dict:
    return {"row_id": "L03", "side": "left", "camera_height_mm": 1000.0, "distance_mm": 270.0,
            "x_m": 12.5, "pass_id": "m-1", "timestamp": "2026-10-07T12:00:00+00:00",
            "frame_id": frame_id, "camera_id": "esq-1000", "image_width_px": 1600,
            "hfov_deg": 127.0, "lat": -22.9, "lon": -47.06,
            "tiles": [_tile(600, 0.97), _tile(700, 0.95)], **extra}


def test_quadro_da_borda_vira_planta_e_reenvio_nao_soma(client, field_dir) -> None:
    r = client.post("/fields/T01/frames", params={"model": BANANA}, json=_quadro())
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo["plants"] and not corpo["duplicated"] and corpo["calibration"] is None
    antes = client.get("/fields/T01/plants").json()

    again = client.post("/fields/T01/frames", params={"model": BANANA}, json=_quadro())
    assert again.json() == {"duplicated": True, "plants": [], "calibration": None}
    assert client.get("/fields/T01/plants").json() == antes


def test_usa_a_calibracao_da_borda_nao_a_da_nuvem(client, field_dir, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    nuvem = calibration(model_key=BANANA)
    get_spec(BANANA).fusion_path.write_text(
        json.dumps({"version": 1, "calibration": nuvem.to_dict()}), encoding="utf-8")
    r = client.post("/fields/T01/frames", params={"model": BANANA}, json=_quadro())
    assert r.json()["calibration"] is None  # só a .fusion.json da nuvem: não serve à borda

    fields.reset(field_dir)
    borda = calibration(model_key=BANANA)
    get_spec(BANANA).edge_fusion_path.write_text(
        json.dumps({"version": 1, "calibration": borda.to_dict()}), encoding="utf-8")
    r = client.post("/fields/T02/frames", params={"model": BANANA}, json=_quadro("r-2"))
    assert r.json()["calibration"] == borda.id
    assert {p["calibration_id"] for p in client.get("/fields/T02/plants").json()} == {borda.id}


def test_passada_fechada_e_quadro_invalido(client, field_dir) -> None:
    assert client.post("/fields/T01/frames", params={"model": BANANA},
                       json=_quadro()).status_code == 200
    assert client.post("/fields/T01/passes/m-1/close").status_code == 200
    tarde = client.post("/fields/T01/frames", params={"model": BANANA}, json=_quadro("r-9"))
    assert tarde.status_code == 409 and "fechada" in tarde.json()["detail"]

    assert client.post("/fields/T01/frames", params={"model": BANANA},
                       json=_quadro("r-3", side="cima")).status_code == 422
    assert client.post("/fields/T01/frames", params={"model": "nada"},
                       json=_quadro("r-4")).status_code == 422
    sem_id = {k: v for k, v in _quadro().items() if k != "frame_id"}
    assert client.post("/fields/T01/frames", params={"model": BANANA},
                       json=sem_id).status_code == 422


def test_desligado_sem_fields_dir(client) -> None:
    fields.reset(None)
    assert client.post("/fields/T01/frames", params={"model": BANANA},
                       json=_quadro()).status_code == 503


def test_quadro_de_cafe_vira_trecho_da_linha(client, field_dir) -> None:
    """O café é a cultura do produto: a fusão aceita o talhão de café (pé a ~0,7 m)."""
    classes = get_spec("coffee-foliar").class_names

    def tile(x0: float, ferrugem: float) -> dict:
        probs = dict.fromkeys(classes, 0.0)
        probs["Rust"], probs["Healthy"] = ferrugem, 1.0 - ferrugem
        return {"box": [x0, 400, x0 + 220, 620], "predicted_class": max(probs, key=probs.get),
                "probabilities": probs, "confidence": max(probs.values()),
                "height_mm": 1100.0, "zones": ["terco_medio"]}

    quadro = {**_quadro("c-1"), "tiles": [tile(100, 0.97), tile(1300, 0.05)]}
    r = client.post("/fields/C01/frames", params={"model": "coffee-foliar"}, json=quadro)
    assert r.status_code == 200, r.text
    # Recortes a ~0,6 m um do outro num quadro de ~1 m: no cafezal são pés diferentes.
    assert len(r.json()["plants"]) == 2
    plantas = {p["plant_id"]: p for p in client.get("/fields/C01/plants").json()}
    assert len(plantas) == 2
    assert {p["crop"] for p in plantas.values()} == {"coffee"}
    com, sem = sorted(plantas.values(), key=lambda p: -p["posteriors"]["Rust"])
    assert com["primary_disease"] == "Rust"
    assert com["posteriors"]["Rust"] > sem["posteriors"]["Rust"]
    assert com["zones_seen"] == ["terco_medio"]


def test_espacamento_do_talhao_separa_os_pes(client, field_dir) -> None:
    """Cafezal adensado a 0,5 m: recortes a ~0,25 m viram pés diferentes (a 0,7 m, o mesmo)."""
    classes = get_spec("coffee-foliar").class_names
    probs = dict.fromkeys(classes, 0.0)
    probs["Healthy"] = 1.0

    def tile(x0: float) -> dict:
        return {"box": [x0, 400, x0 + 220, 620], "predicted_class": "Healthy",
                "probabilities": probs, "confidence": 1.0}

    quadro = {**_quadro("e-1"), "tiles": [tile(500), tile(870)]}  # ~0,25 m de distância
    padrao = client.post("/fields/E1/frames", params={"model": "coffee-foliar"}, json=quadro)
    adensado = client.post("/fields/E2/frames", params={"model": "coffee-foliar"},
                           json={**quadro, "plant_spacing_m": 0.5})
    assert len(padrao.json()["plants"]) == 1  # 0,7 m: os dois recortes caem no mesmo pé
    assert len(adensado.json()["plants"]) == 2  # 0,5 m: são pés vizinhos
