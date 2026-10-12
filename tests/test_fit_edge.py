"""``training.fit_edge``: os limiares da Edge AI do Rover e a fusão dos recortes dele."""

from __future__ import annotations

import json

import numpy as np

from app.registry import get_spec
from training import fit_edge
from training.fit_fusion import load_plants, read_labels

BANANA = "banana-foliar"
LABELS = list(get_spec(BANANA).class_names)
SADIO = LABELS.index("Healthy")
SIGATOKA = LABELS.index("Sigatoka")


def _probs(doenca: list[float]) -> list[list[float]]:
    linhas = []
    for p in doenca:
        linha = [0.0] * len(LABELS)
        linha[SIGATOKA], linha[SADIO] = p, 1.0 - p
        linhas.append(linha)
    return linhas


def _frames(tmp_path, n=20, seed=0):
    """Doentes com um recorte alto entre baixos; sadias com recortes baixos."""
    rng = np.random.default_rng(seed)
    caminho = tmp_path / "edge.jsonl"
    with caminho.open("w", encoding="utf-8") as fh:
        for i in range(n):
            for rotulo in ("Sigatoka", "Healthy"):
                for altura in (450.0, 1000.0, 1540.0):
                    baixos = list(rng.uniform(0.0, 0.6, 12))
                    if rotulo == "Sigatoka" and altura == 1000.0:
                        baixos[0] = float(rng.uniform(0.9, 1.0))
                    fh.write(json.dumps({"plant": f"{rotulo}-{i}", "label": rotulo,
                                         "group": "campo", "camera_height_mm": altura,
                                         "probs": _probs(baixos)}) + "\n")
    return caminho


def test_evento_e_categorias() -> None:
    quadro = {"doenca": np.array([0.2, 0.95, 0.91])}
    assert fit_edge.frame_event(quadro, 0.9, 2) and not fit_edge.frame_event(quadro, 0.92, 2)
    assert not fit_edge.frame_event(None, 0.5, 1)
    cats = fit_edge.categories(["Healthy", "Leafminer", "Mites", "Greening"])
    assert cats == {"Healthy": "NORMAL", "Leafminer": "POSSIVEL_PRAGA",
                    "Mites": "POSSIVEL_PRAGA", "Greening": "POSSIVEL_DOENCA"}


def test_fit_separa_e_grava_perfil_e_categorias(tmp_path) -> None:
    frames = _frames(tmp_path)
    out = tmp_path / "m.edge.json"
    fit_edge.main(["fit", "--model", BANANA, "--frames", str(frames), "--out", str(out)])
    perfil = json.loads(out.read_text(encoding="utf-8"))
    assert 0.6 <= perfil["event_threshold"] <= 0.9 and perfil["min_event_tiles"] == 1
    assert perfil["near_miss_threshold"] <= perfil["event_threshold"]
    assert perfil["measured"]["fora_da_amostra"]["plantas_doentes_pegas"] == "20/20"
    assert perfil["measured"]["fora_da_amostra"]["quadros_sadios_evento"] == "0/60"
    ev = perfil["evidence"]
    assert set(ev["llr_tables"]) == set(LABELS) - {"Healthy"}
    assert len(ev["llr_tables"]["Sigatoka"]) == len(ev["bin_edges"]) - 1
    assert ev["near_miss_evidence"] <= ev["event_evidence"]
    fora = perfil["measured"]["evidencia_fora_da_amostra"]
    # 20 plantas por classe e uma doença só: as tabelas saem ruidosas; o número que vale é
    # o dos dados reais (MODELS.md). Aqui confere-se a mecânica e a ordem de grandeza.
    assert int(fora["plantas_doentes_pegas"].split("/")[0]) >= 12
    assert fora["quadros_sadios_evento"].endswith("/60")
    cats = json.loads((tmp_path / "m.categories.json").read_text(encoding="utf-8"))
    assert cats["Healthy"] == "NORMAL" and cats["Sigatoka"] == "POSSIVEL_DOENCA"


def test_quadros_da_borda_no_formato_da_fusao(tmp_path) -> None:
    linhas = fit_edge.to_fusion_lines([_frames(tmp_path, n=2)], LABELS)
    assert len(linhas) == 12 and all(t["in_scope"] is None for t in linhas[0]["tiles"])
    convertido = tmp_path / "f.jsonl"
    convertido.write_text("".join(json.dumps(x) + "\n" for x in linhas), encoding="utf-8")
    plantas = load_plants([convertido], ("Sigatoka",))
    assert len(plantas) == 4 and all(len(p.frames) == 3 for p in plantas)
    doente = next(p for p in plantas if p.label == "Sigatoka")
    assert max(f.probs.max() for f in doente.frames) >= 0.9


def test_recalibra_com_o_registro_de_campo(tmp_path) -> None:
    """Café em campo: o registro da IA (uma linha por quadro e planta) + rótulo do agrônomo."""
    import csv

    from app.registry import get_spec

    classes = list(get_spec("coffee-foliar").class_names)
    rng = np.random.default_rng(1)
    registro, rotulos = tmp_path / "registro.jsonl", tmp_path / "rotulos.csv"

    def recortes(doente: bool) -> list[dict]:
        saida = []
        for i in range(12):
            p = dict.fromkeys(classes, 0.0)
            ferrugem = float(rng.uniform(0.9, 1.0)) if doente and i == 0 else float(
                rng.uniform(0.0, 0.5))
            p["Rust"], p["Healthy"] = ferrugem, 1.0 - ferrugem
            saida.append({"box": [0, 0, 1, 1], "probabilities": p})
        return saida

    with registro.open("w", encoding="utf-8") as fh, rotulos.open("w", newline="") as fr:
        w = csv.writer(fr)
        w.writerow(["field_id", "plant_id", "label"])
        for n in range(24):
            doente = n % 2 == 0
            w.writerow(["T01", f"L1-P{n:04d}", "Rust" if doente else "Healthy"])
            for q in range(3):
                fh.write(json.dumps({"field_id": "T01", "pass_id": "m-1", "plant": f"L1-P{n:04d}",
                                     "frame": f"q{n}-{q}", "camera_height_mm": 1000.0,
                                     "tiles": recortes(doente and q == 1)}) + "\n")
        # Um quadro que tocou dois pés (no cafezal é o comum) e um pé sem rótulo.
        for planta in ("L1-P0000", "L1-P9999"):
            fh.write(json.dumps({"field_id": "T01", "pass_id": "m-1", "plant": planta,
                                 "frame": "q-duplo", "camera_height_mm": 1000.0,
                                 "tiles": recortes(False)[:6]}) + "\n")

    linhas = fit_edge.registro_to_frames([registro], read_labels(rotulos), classes)
    duplo = [x for x in linhas if x["plant"].startswith("T01|m-1|L1-P0000")]
    assert len(duplo) == 4  # 3 quadros dele + o quadro duplo, inteiro (12 recortes)
    assert any(len(x["probs"]) == 12 for x in duplo) and not any(
        "P9999" in x["plant"] for x in linhas)

    out = tmp_path / "c.edge.json"
    fit_edge.main(["fit", "--model", "coffee-foliar", "--frames", str(registro),
                   "--labels", str(rotulos), "--out", str(out)])
    perfil = json.loads(out.read_text(encoding="utf-8"))
    assert perfil["crop"] == "coffee" and "evidence" in perfil
    assert perfil["measured"]["plantas"] == {"doentes": 12, "sadias": 12}


def test_mosaico_do_cafe_por_incidencia(tmp_path, monkeypatch) -> None:
    """Folha doente rara num quadro de folhas sadias: quanto mais folhas doentes, mais pega."""
    import hashlib

    from app.config import settings

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    (tmp_path / "framing_vision.onnx").write_bytes(b"torre de visao")  # o perfil leva o hash
    labels = ["Cercospora_Leaf_Spot", "Healthy", "Leaf_Miner_Damage", "Phoma_Leaf_Spot", "Rust"]
    rng = np.random.default_rng(0)

    def fotos(classe: str, n: int) -> np.ndarray:
        p = np.full((n, 5), 0.01)
        p[:, labels.index(classe)] = rng.uniform(0.85, 0.96, n)
        return p / p.sum(axis=1, keepdims=True)

    probs = {"Healthy": fotos("Healthy", 200), "Rust": fotos("Rust", 100)}
    pools = {c: np.arange(len(v)) % 5 for c, v in probs.items()}
    linhas = fit_edge.mosaic_frames(probs, pools, tiles=40, incidences=[0.02, 0.2], plants=6,
                                    healthy=12, rng=rng, labels=labels, path="cabeca:teste",
                                    n_pools=5)
    assert {x["group"] for x in linhas} == {f"dobra:{g}" for g in range(5)}
    frames = tmp_path / "mosaico.jsonl"
    frames.write_text("".join(json.dumps(x) + "\n" for x in linhas), encoding="utf-8")
    out = tmp_path / "c.edge.json"
    fit_edge.main(["fit", "--model", "coffee-foliar", "--frames", str(frames), "--out", str(out)])
    perfil = json.loads(out.read_text(encoding="utf-8"))
    assert perfil["head"] == {"encoder_model_id": "teste",
                              "encoder_sha256": hashlib.sha256(b"torre de visao").hexdigest(),
                              "input_size": 224}
    fora = perfil["measured"]["evidencia_fora_da_amostra"]
    pegas = {k: int(v.split("/")[0]) / int(v.split("/")[1])
             for k, v in fora["plantas_doentes_pegas_por_incidencia"].items()}
    assert pegas["0.2"] >= pegas["0.02"] and pegas["0.2"] > 0.8
