"""Nível 2, 5.3: a lesão no referencial da planta, reencontrada de uma passada para outra."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.plant_fusion import PlantFusionStore, lesion_tracks
from tests.test_plant_fusion import frame, tile


def _com_geometria(t: dict[str, Any], y_mm: float, z_mm: float,
                   inclinacao: float | None = 20.0) -> dict[str, Any]:
    """O recorte com o ponto 3D do Rover: ``y`` para baixo da câmera, ``z`` para dentro."""
    return {**t, "geometria": {"centro_mm": [0.0, y_mm, z_mm], "inclinacao_graus": inclinacao,
                               "superficie": "plana" if inclinacao is not None else "irregular",
                               "face": "indeterminada"}}


def _passada(store: PlantFusionStore, pass_id: str, ts: str,
             lesoes: list[tuple[float, float, float]], sadios: int = 0) -> None:
    """Cada lesão (ao longo da rua, y, z) num quadro da câmera de 1 m; ``sadios`` quadros a mais."""
    for i, (x, y, z) in enumerate(lesoes):
        store.ingest(frame(0.0, [_com_geometria(tile(x, "left", "Sigatoka"), y, z)],
                           plant_id="B1", pass_id=pass_id, timestamp=ts,
                           frame_id=f"{pass_id}-{i}"))
    for j in range(sadios):
        store.ingest(frame(0.0, [tile(0.0, "left")], plant_id="B1", pass_id=pass_id,
                           timestamp=ts, frame_id=f"{pass_id}-s{j}"))
    store.close_pass("talhao-a", pass_id)


def _evidencias(store: PlantFusionStore, planta: str) -> dict[str, Any]:
    return store._require("talhao-a").plants[planta].passes


def test_lesao_no_referencial_da_planta(tmp_path: Path) -> None:
    store = PlantFusionStore(tmp_path)
    # 0,2 m abaixo da câmera de 1 m (altura 0,8 m), 5 cm além da distância da planta (27 cm).
    _passada(store, "p1", "2026-09-01T10:00:00Z", [(0.0, 200.0, 320.0)], sadios=2)
    ev = _evidencias(store, "B1")["p1"]
    assert len(ev.lesions) == 1  # o recorte sadio não vira lesão
    lesao = ev.lesions[0]
    assert lesao["disease"] == "Sigatoka" and lesao["llr"] > 0
    assert lesao["height_m"] == pytest.approx(0.8)
    assert lesao["canopy_depth_m"] == pytest.approx(0.05)
    assert lesao["face"] == "indeterminada" and lesao["tilt_deg"] == 20.0
    # Sobrevive à gravação do estado.
    de_novo = PlantFusionStore(tmp_path)
    assert _evidencias(de_novo, "B1")["p1"].lesions == ev.lesions


def test_sem_geometria_fica_a_altura_do_recorte(tmp_path: Path) -> None:
    store = PlantFusionStore(tmp_path)
    store.ingest(frame(0.0, [tile(0.0, "left", "Sigatoka")], plant_id="B1", frame_id="f"))
    lesao = _evidencias(store, "B1")["p1"].lesions[0]
    assert lesao["height_m"] == pytest.approx(1.5) and lesao["canopy_depth_m"] is None
    assert lesao["face"] is None


def test_trilhas_entre_passadas(tmp_path: Path) -> None:
    store = PlantFusionStore(tmp_path)
    # p1: duas lesões, 30 cm uma da outra ao longo da rua.
    _passada(store, "p1", "2026-09-01T10:00:00Z", [(0.0, 200.0, 320.0), (0.3, 200.0, 320.0)])
    # p2: a primeira de novo (2 cm de erro), a segunda sumiu, uma nova embaixo.
    _passada(store, "p2", "2026-09-08T10:00:00Z", [(0.02, 210.0, 330.0), (0.0, 600.0, 300.0)])
    r = store.lesions("talhao-a", "B1")
    assert r["passes"] == ["p1", "p2"]
    estados = sorted((t["state"], t["passes"]) for t in r["tracks"])
    assert estados == [("nao_revista", ["p1"]), ("nova", ["p2"]), ("persistente", ["p1", "p2"])]
    persistente = next(t for t in r["tracks"] if t["state"] == "persistente")
    assert persistente["position"]["height_m"] == pytest.approx(0.79)
    assert persistente["offset_from_center_m"] == pytest.approx(
        persistente["position"]["x_m"] - r["center_m"], abs=1e-3)
    with pytest.raises(KeyError):
        store.lesions("talhao-a", "nao-existe")


def test_mesma_passada_varias_vistas_sao_uma_lesao() -> None:
    obs = [{"disease": "Rust", "llr": 1.0 + i / 10, "x_m": 0.01 * i, "height_m": 0.8,
            "canopy_depth_m": 0.05, "tilt_deg": 10.0 * i, "face": "indeterminada"}
           for i in range(3)]
    outra_doenca = {**obs[0], "disease": "Leaf_Miner_Damage"}
    trilhas = lesion_tracks([("p1", [*obs, outra_doenca])], center_m=0.0)
    assert len(trilhas) == 2
    rust = next(t for t in trilhas if t["disease"] == "Rust")
    assert rust["views"] == 3 and rust["tilt_deg"] == pytest.approx(10.0)
    assert rust["state"] == "nova" and rust["face"] == "indeterminada"


def test_doenca_diferente_nao_casa_entre_passadas() -> None:
    base = {"llr": 1.0, "x_m": 0.0, "height_m": 0.8, "canopy_depth_m": 0.05}
    trilhas = lesion_tracks([("p1", [{**base, "disease": "Rust"}]),
                             ("p2", [{**base, "disease": "Leaf_Miner_Damage"}])], 0.0)
    assert sorted(t["state"] for t in trilhas) == ["nao_revista", "nova"]


def test_rota_de_lesoes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import fields_api

    store = PlantFusionStore(tmp_path)
    _passada(store, "p1", "2026-09-01T10:00:00Z", [(0.0, 200.0, 320.0)])
    monkeypatch.setattr(fields_api, "_require_fields", lambda: store)
    monkeypatch.setattr(fields_api, "_known", lambda s, f: None)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(fields_api.router)
    cliente = TestClient(app)
    prefixo = fields_api.router.prefix
    r = cliente.get(f"{prefixo}/talhao-a/plants/B1/lesoes")
    assert r.status_code == 200 and r.json()["tracks"][0]["state"] == "nova"
    assert cliente.get(f"{prefixo}/talhao-a/plants/X9/lesoes").status_code == 404


def test_geometria_torta_e_ignorada(tmp_path: Path) -> None:
    store = PlantFusionStore(tmp_path)
    for i, g in enumerate([{"centro_mm": [0, "x", 300]}, {"centro_mm": [0, 0]}, "lixo",
                           {"centro_mm": [0, 0, float("nan")]},
                           {"centro_mm": [0, 200, 320], "inclinacao_graus": 400, "face": 3}]):
        store.ingest(frame(0.0, [{**tile(0.0, "left", "Sigatoka"), "geometria": g}],
                           plant_id="B1", frame_id=f"f{i}"))
    lesoes = _evidencias(store, "B1")["p1"].lesions
    assert [lesao["canopy_depth_m"] for lesao in lesoes] == [None] * 4 + [pytest.approx(0.05)]
    assert lesoes[-1]["tilt_deg"] is None and lesoes[-1]["face"] is None
