"""O gêmeo pela API, pelo caminho real: quadros de café do Rover em três passadas fechadas,
tempo da estação, ajuste, previsão e prescrição por risco."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.config import settings
from app.fields_api import fields
from app.registry import get_spec

CAFE = "coffee-foliar"
CLASSES = get_spec(CAFE).class_names
INICIO = datetime(2026, 3, 2, 12, tzinfo=timezone.utc)
FOCO = {("L01", 4.0), ("L01", 6.0), ("L02", 4.0)}


@pytest.fixture
def field_dir(tmp_path, monkeypatch):
    # Sem os artefatos instalados: a fusão roda com os valores de partida em qualquer máquina
    # (com a calibração do café do Windows o teste passava; sem ela, no WSL, não).
    monkeypatch.setattr(settings, "models_dir", tmp_path / "modelos")
    fields.reset(tmp_path / "talhoes")
    yield tmp_path / "talhoes"
    fields.reset(None)


def _tile(x0: float, ferrugem: float) -> dict:
    probs = dict.fromkeys(CLASSES, 0.0)
    probs["Rust"], probs["Healthy"] = ferrugem, 1.0 - ferrugem
    return {"box": [x0, 400, x0 + 220, 620], "predicted_class": max(probs, key=probs.get),
            "probabilities": probs, "confidence": max(probs.values()), "height_mm": 1100.0,
            "zones": ["terco_medio"]}


def _passada(client, k: int) -> None:
    """Três ruas, um quadro a cada 2 m; o foco cresce de uma passada para a outra."""
    quando = INICIO + timedelta(days=14 * k)
    for rua in ("L01", "L02", "L03"):
        for i, x in enumerate((2.0, 4.0, 6.0, 8.0, 10.0)):
            doente = (rua, x) in FOCO and (k >= 1 or (rua, x) == ("L01", 4.0))
            quadro = {"row_id": rua, "side": "left", "camera_height_mm": 1000.0,
                      "distance_mm": 270.0, "x_m": x, "pass_id": f"m-{k}",
                      "timestamp": (quando + timedelta(minutes=i)).isoformat(),
                      "frame_id": f"{rua}-{x}-{k}", "camera_id": "esq", "image_width_px": 1600,
                      "hfov_deg": 127.0, "lat": -21.2 + x * 1e-5, "lon": -45.0,
                      "tiles": [_tile(x0, 0.97 if doente else 0.03) for x0 in (600, 650, 700)]}
            r = client.post("/fields/C1/frames", params={"model": CAFE}, json=quadro)
            assert r.status_code == 200, r.text
    assert client.post(f"/fields/C1/passes/m-{k}/close").status_code == 200


def _dias(inicio: date, n: int) -> list[dict]:
    return [{"data": (inicio + timedelta(days=d)).isoformat(), "t_media": 21.0, "t_min": 16.0,
             "t_max": 27.0, "ur_media": 85.0, "chuva_mm": 6.0 if d % 3 == 0 else 0.0,
             "molhamento_h": 10.0 if d % 3 == 0 else 4.0} for d in range(n)]


PRODUTO = {"nome": "Fungicida X", "registro_mapa": "12345", "culturas": ["coffee"],
           "doencas": ["Rust"], "dose_ml_por_planta": 50, "dose_min_ml_por_planta": 40,
           "dose_max_ml_por_planta": 60, "preco_por_litro": 80.0, "eficacia": 0.85,
           "carencia_dias": 30}


def test_gemeo_ponta_a_ponta(client, field_dir) -> None:
    for k in range(3):
        _passada(client, k)
    # Sem tempo, o gêmeo não roda — e diz o que falta.
    sem = client.get("/fields/C1/gemeo/previsao", params={"doenca": "Rust"})
    assert sem.status_code == 409 and "clima" in sem.json()["detail"]

    r = client.post("/fields/C1/clima", json={"dias": _dias(INICIO.date(), 40)})
    assert r.status_code == 200 and r.json()["por_fonte"] == {"estacao": 40}
    assert client.get("/fields/C1/clima").json()["dias"] == 40

    resumo = client.get("/fields/C1/gemeo").json()
    assert resumo["passadas"] and len(resumo["passadas"]) == 3
    assert resumo["modelos"]["Rust"] is None

    aj = client.post("/fields/C1/gemeo/ajuste", params={"doenca": "Rust", "max_iter": 40})
    assert aj.status_code == 200, aj.text
    assert set(aj.json()["diagnostico"]) >= {"identificado", "intervalo_95", "laplace"}
    assert "adotada" in aj.json()["rede"]
    assert client.get("/fields/C1/gemeo").json()["modelos"]["Rust"]["passadas"] == [
        "m-0", "m-1", "m-2"]

    hoje = (INICIO + timedelta(days=28)).date().isoformat()
    prev = client.get("/fields/C1/gemeo/previsao",
                      params={"doenca": "Rust", "horizonte_dias": 10, "membros": 2,
                              "hoje": hoje, "plantas": "todas"})
    assert prev.status_code == 200, prev.text
    corpo = prev.json()
    assert corpo["hoje"] == hoje and len(corpo["plantas_detalhe"]) == corpo["plantas"]
    risco = {p["plant_id"]: p["risco"] for p in corpo["plantas_detalhe"]}
    rua = {p["plant_id"]: (p["row_id"], p["x_along_m"]) for p in corpo["plantas_detalhe"]}
    no_foco = [risco[i] for i, (r_, x) in rua.items() if r_ == "L01" and x < 7]
    longe = [risco[i] for i, (r_, x) in rua.items() if r_ == "L03" and x > 7]
    assert max(no_foco) > max(longe)
    # Passou do fim da série: climatologia, e o aviso diz.
    longo = client.get("/fields/C1/gemeo/previsao",
                       params={"doenca": "Rust", "horizonte_dias": 30, "membros": 1,
                               "hoje": hoje, "plantas": "nenhuma"}).json()
    assert any("climatologia" in a for a in longo["avisos"])

    presc = client.post("/fields/C1/gemeo/prescricao", json={
        "doenca": "Rust", "produto": PRODUTO, "economia": {"perda_por_planta": 40.0},
        "residual_dias": 21, "horizonte_dias": 10, "membros": 2, "hoje": hoje})
    assert presc.status_code == 200, presc.text
    p = presc.json()
    assert set(p["cenarios"]) == {"sem_aplicacao", "prescricao_na_data",
                                  "prescricao_7_dias_depois", "talhao_inteiro_na_data"}
    assert "resumo" in p and p["previsao"]["doenca"] == "Rust"
    geo = client.post("/fields/C1/gemeo/prescricao", params={"formato": "geojson"}, json={
        "doenca": "Rust", "produto": PRODUTO, "economia": {"perda_por_planta": 40.0},
        "residual_dias": 21, "horizonte_dias": 10, "membros": 1, "hoje": hoje})
    assert geo.json()["type"] == "FeatureCollection"


def test_gemeo_recusas(client, field_dir, monkeypatch) -> None:
    _passada(client, 0)
    client.post("/fields/C1/clima", json={"dias": _dias(INICIO.date(), 10)})
    assert client.get("/fields/C1/gemeo/previsao",
                      params={"doenca": "Gripe"}).status_code == 422
    assert client.get("/fields/NADA/gemeo").status_code == 404
    # A API não chama serviço de fora por padrão.
    monkeypatch.setattr(settings, "clima_open_meteo", False)
    r = client.post("/fields/C1/clima", json={"buscar_open_meteo": True})
    assert r.status_code == 409 and "CLIMA_OPEN_METEO" in r.json()["detail"]
    assert client.post("/fields/C1/clima", json={}).status_code == 422
    invertido = _dias(INICIO.date(), 1)
    invertido[0]["t_min"] = 30.0
    assert client.post("/fields/C1/clima", json={"dias": invertido}).status_code == 422
    # Bula de outra cultura: recusa antes de qualquer conta.
    banana = {**PRODUTO, "culturas": ["banana"]}
    r = client.post("/fields/C1/gemeo/prescricao", json={
        "doenca": "Rust", "produto": banana, "economia": {"perda_por_planta": 40.0},
        "residual_dias": 21, "membros": 1, "hoje": INICIO.date().isoformat()})
    assert r.status_code == 422


def test_open_meteo_com_a_flag_e_buscador_falso(client, field_dir, monkeypatch) -> None:
    from app.gemeo import clima

    _passada(client, 0)
    monkeypatch.setattr(settings, "clima_open_meteo", True)
    chamadas = []

    def falso(lat, lon, *, inicio, hoje, dias_previsao):
        chamadas.append((round(lat, 3), round(lon, 3), inicio, dias_previsao))
        return [clima.DiaClimatico(inicio + timedelta(days=d), 21, 16, 27, 80, 1, 6,
                                   fonte="open-meteo:historico") for d in range(5)]

    monkeypatch.setattr(clima, "open_meteo", falso)
    r = client.post("/fields/C1/clima", json={"buscar_open_meteo": True, "dias_previsao": 7})
    assert r.status_code == 200, r.text
    assert r.json()["por_fonte"] == {"open-meteo:historico": 5}
    assert chamadas and chamadas[0][0] == -21.2 and chamadas[0][3] == 7


def test_desligado_sem_fields_dir(client) -> None:
    fields.reset(None)
    assert client.get("/fields/C1/gemeo").status_code == 503


def test_evidencia_do_drone_e_prioridades(client, field_dir) -> None:
    for k in range(2):
        _passada(client, k)
    client.post("/fields/C1/clima", json={"dias": _dias(INICIO.date(), 40)})
    plantas = {p["plant_id"]: p for p in client.get("/fields/C1/plants").json()}
    foco = next(pid for pid, p in plantas.items()
                if p["row_id"] == "L03" and p["x_along_m"] > 7)
    voo = {"fonte": "drone", "pass_id": "voo-1",
           "time": (INICIO + timedelta(days=20)).isoformat(),
           "evidencia": {"Rust": {foco: 4.0, "nao-existe": 1.0}}}
    # O Rover não entra por aqui (contaria duas vezes); fonte sem canal também não.
    assert client.post("/fields/C1/gemeo/observacoes",
                       json={**voo, "fonte": "rover"}).status_code == 422
    assert client.post("/fields/C1/gemeo/observacoes",
                       json={**voo, "fonte": "satelite"}).status_code == 422
    assert client.post("/fields/C1/gemeo/observacoes",
                       json={**voo, "evidencia": {"Rust": {foco: 99.0}}}).status_code == 422
    assert client.post("/fields/C1/gemeo/observacoes",
                       json={**voo, "evidencia": {"Gripe": {foco: 1.0}}}).status_code == 422
    r = client.post("/fields/C1/gemeo/observacoes", json=voo)
    assert r.status_code == 200, r.text
    assert r.json()["plantas_desconhecidas"] == 1 and not r.json()["substituiu"]
    assert client.post("/fields/C1/gemeo/observacoes", json=voo).json()["substituiu"]
    resumo = client.get("/fields/C1/gemeo").json()
    assert [p["fonte"] for p in resumo["passadas"]] == ["rover", "rover", "drone"]

    hoje = (INICIO + timedelta(days=20)).date().isoformat()
    base = {"doenca": "Rust", "dias_ate_visita": 2, "membros": 1, "hoje": hoje,
            "segmento_m": 4}
    pr = client.get("/fields/C1/gemeo/prioridades", params=base)
    assert pr.status_code == 200, pr.text
    corpo = pr.json()
    assert corpo["criterio"] == "informacao" and corpo["tarefas"]
    valores = [t["valor"] for t in corpo["tarefas"]]
    assert valores == sorted(valores, reverse=True)
    com_eco = client.get("/fields/C1/gemeo/prioridades", params={
        **base, "custo_tratar": 2, "perda_planta": 15, "eficacia": 0.85, "com_plantas": True})
    assert com_eco.json()["criterio"] == "valor_da_informacao"
    assert com_eco.json()["tarefas"][0]["plant_ids"]
    drone = client.get("/fields/C1/gemeo/prioridades", params={**base, "canal": "drone"})
    assert drone.json()["tarefas"][0]["tipo"] == "sobrevoo"
    assert client.get("/fields/C1/gemeo/prioridades",
                      params={**base, "custo_tratar": 2}).status_code == 422
    assert client.get("/fields/C1/gemeo/prioridades",
                      params={**base, "canal": "satelite"}).status_code == 422


def test_ajuste_com_dispersao_pelo_vento(client, field_dir) -> None:
    for k in range(3):
        _passada(client, k)
    dias = [{**d, "vento_ms": 3.0, "vento_u_ms": 3.0, "vento_v_ms": 0.0}
            for d in _dias(INICIO.date(), 40)]
    assert client.post("/fields/C1/clima", json={"dias": dias}).status_code == 200
    sem_azimute = client.post("/fields/C1/gemeo/ajuste",
                              params={"doenca": "Rust", "max_iter": 30, "dispersao": True})
    assert sem_azimute.status_code == 422 and "azimute" in sem_azimute.text
    aj = client.post("/fields/C1/gemeo/ajuste", params={
        "doenca": "Rust", "max_iter": 30, "dispersao": True, "azimute_rua_graus": 0})
    assert aj.status_code == 200, aj.text
    escolha = aj.json()["dispersao"]
    assert escolha["fracao_aerea"] in (0.0, 0.3, 0.6, 0.9)
    assert escolha["dias_com_vento"] == escolha["dias"] > 0
    hoje = (INICIO + timedelta(days=28)).date().isoformat()
    prev = client.get("/fields/C1/gemeo/previsao", params={
        "doenca": "Rust", "horizonte_dias": 7, "membros": 2, "hoje": hoje})
    assert prev.status_code == 200, prev.text
    # Sem pedir, o ajuste não mexe com vento.
    padrao = client.post("/fields/C1/gemeo/ajuste", params={"doenca": "Rust", "max_iter": 30})
    assert padrao.json()["dispersao"] is None
