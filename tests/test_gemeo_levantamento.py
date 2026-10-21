"""Levantamento aéreo importado (7.2, etapa 1): regiões suspeitas do voo viram evidência por
planta no canal do drone, e mudam as prioridades do gêmeo. Dados simulados."""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from app.gemeo import levantamento
from app.gemeo.epidemia import CANAIS
from tests.test_gemeo_api import INICIO, _dias, _passada, field_dir  # noqa: F401


def _ret(lat0: float, lat1: float, lon0: float = -45.0001, lon1: float = -44.9999):
    return [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]]


def test_llr_pelo_canal_e_o_que_nao_foi_visto() -> None:
    canal = CANAIS["drone"]
    ids = ["a", "b", "c", "d"]
    lat = np.array([0.5, 0.5, 2.5, np.nan])
    lon = np.array([0.5, 1.5, 0.5, 0.5])
    cobertura = [[0, 0], [2, 0], [2, 1], [0, 1]]
    evid, conta = levantamento.evidencia(ids, lat, lon, cobertura, [[[0, 0], [1, 0], [1, 1],
                                                                     [0, 1]]], canal)
    assert evid == {"a": round(math.log(0.45 / 0.06), 6),
                    "b": round(math.log(0.55 / 0.94), 6)}
    assert conta == {"cobertas": 2, "sinalizadas": 1, "sem_gps": 1, "fora_da_cobertura": 1}


def test_levantamento_pela_api_muda_as_prioridades(client, field_dir) -> None:  # noqa: F811
    for k in range(2):
        _passada(client, k)
    client.post("/fields/C1/clima", json={"dias": _dias(INICIO.date(), 40)})
    hoje = (INICIO + timedelta(days=20)).date().isoformat()
    base = {"doenca": "Rust", "dias_ate_visita": 2, "membros": 1, "hoje": hoje,
            "segmento_m": 4}
    antes = client.get("/fields/C1/gemeo/prioridades", params=base).json()
    voo = {"pass_id": "voo-1", "time": (INICIO + timedelta(days=20)).isoformat(),
           "doenca": "Rust", "cobertura": _ret(-21.3, -21.0),
           "anomalias": [_ret(-21.19995, -21.19985)]}  # x de 5 a 15 m: o foco e além
    assert client.post("/fields/C1/gemeo/levantamento",
                       json={**voo, "fonte": "rover"}).status_code == 422
    assert client.post("/fields/C1/gemeo/levantamento",
                       json={**voo, "doenca": "Gripe"}).status_code == 422
    assert client.post("/fields/C1/gemeo/levantamento",
                       json={**voo, "cobertura": _ret(10, 11)}).status_code == 422
    r = client.post("/fields/C1/gemeo/levantamento", json=voo)
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo["cobertas"] > corpo["sinalizadas"] > 0 and "suposição" in corpo["canal"]["origem"]
    assert "não é específico" in corpo["aviso"]
    assert [p["fonte"] for p in client.get("/fields/C1/gemeo").json()["passadas"]] == \
        ["rover", "rover", "drone"]
    depois = client.get("/fields/C1/gemeo/prioridades", params=base).json()
    assert depois["tarefas"] != antes["tarefas"]  # a evidência do voo mexeu no valor
    assert client.post("/fields/C1/gemeo/levantamento", json=voo).json()["substituiu"]
