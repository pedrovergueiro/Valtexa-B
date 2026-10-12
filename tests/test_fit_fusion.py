"""Calibração da fusão por planta (training/fit_fusion.py), em plantas sintéticas.

O modelo sintético faz o que o real faz no rover: põe probabilidade de doença em pedaço de
folha sadia (40% em média), e só a planta doente tem recortes com a cauda perto de 1.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.plant_fusion import (
    CROP_CLASSES,
    HEALTHY,
    LLR_BIN_EDGES,
    PlantFusionStore,
    load_fusion_calibration,
    logit,
)
from tests.test_plant_fusion import frame, tile
from training.fit_fusion import (
    Settings,
    apply,
    calibrate,
    evaluate,
    fa_thresholds,
    fit_tables,
    load_plants,
    main,
    parse_args,
    platt,
    procedure_of,
    read_labels,
)

CLASSES = CROP_CLASSES["banana"]
DISEASES = tuple(c for c in CLASSES if c != HEALTHY)
N_BINS = len(LLR_BIN_EDGES) - 1
HEIGHTS = (450.0, 1000.0, 1540.0)


def _tile(rng: np.random.Generator, label: str) -> dict[str, Any]:
    if label != HEALTHY and rng.random() < 0.35:
        target, p = label, rng.uniform(0.9, 1.0)  # recorte com lesão
    else:
        target, p = DISEASES[rng.integers(len(DISEASES))], rng.uniform(0.2, 0.6)
    probs = dict.fromkeys(CLASSES, 0.0)
    probs[target] = p
    probs[HEALTHY] = 1.0 - p
    return {
        "box": [0, 0, 10, 10],
        "predicted_class": max(probs, key=probs.__getitem__),
        "probabilities": probs,
        "prediction_set": None,
        "needs_review": False,
        "in_scope": True,
    }


def write_frames(path: Path, seed: int = 0, n_sick: int = 12, n_healthy: int = 60) -> Path:
    rng = np.random.default_rng(seed)
    labels = [d for d in ("Sigatoka", "Cordana") for _ in range(n_sick)] + [HEALTHY] * n_healthy
    with path.open("w", encoding="utf-8") as fh:
        for k, label in enumerate(labels):
            for h in HEIGHTS:
                fh.write(
                    json.dumps(
                        {
                            "crop": "banana",
                            "plant": f"P{k:03d}",
                            "label": label,
                            "group": "campo",
                            "camera_height_mm": h,
                            "tiles": [_tile(rng, label) for _ in range(20)],
                        }
                    )
                    + "\n"
                )
    return path


@pytest.fixture
def frames(tmp_path: Path) -> Path:
    return write_frames(tmp_path / "plantas.jsonl")


def settings() -> Settings:
    return Settings(diseases=DISEASES, n_bins=N_BINS)


class TestPecas:
    def test_tabela_separa_a_cauda(self, frames: Path) -> None:
        plants = load_plants([frames], DISEASES)
        tables = fit_tables(plants, DISEASES, N_BINS)
        sig = DISEASES.index("Sigatoka")
        assert tables[sig][-1] > 1.0  # p perto de 1: só planta com Sigatoka tem
        assert tables[sig][-1] > tables[sig][LLR_BIN_EDGES.index(0.35)]
        # Pestalotiopsis não tem planta rotulada: herda a distribuição combinada, finita.
        assert np.isfinite(tables[DISEASES.index("Pestalotiopsis")]).all()

    def test_limiares_pelo_alarme_falso(self) -> None:
        t_s, t_d = fa_thresholds(np.arange(100.0), 0.05, 0.02)
        values = np.arange(100.0)
        assert (values >= t_s).sum() == 5
        assert (values >= t_d).sum() == 2

    def test_platt_recupera_a_escala(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 2.0, 20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-(1.5 * x - 1.0)))).astype(float)
        offset, scale = platt(x, y)
        assert scale == pytest.approx(1.5, abs=0.1)
        assert offset == pytest.approx(-1.0 - logit(y.mean()), abs=0.15)


class TestCalibra:
    def test_grava_calibracao_que_o_app_le(self, frames: Path, tmp_path: Path) -> None:
        out = tmp_path / "banana.fusion.json"
        main(["fit", "--model", "banana-foliar", "--frames", str(frames), "--out", str(out)])
        cal = load_fusion_calibration(out)
        assert cal.model_key == "banana-foliar" and cal.crop == "banana"
        assert set(cal.llr_tables) == set(DISEASES)
        assert cal.plant_sensitivity > 0.7
        assert cal.plant_false_alarm < 0.15
        assert cal.n_diseased == 24 and cal.n_healthy == 60
        report = json.loads(out.read_text(encoding="utf-8"))["report"]
        cv, base = report["validacao_cruzada"], report["sem_calibracao"]
        assert cv["detectadas"] > base["detectadas"]

    def test_a_loja_reproduz_a_evidencia_do_fit(self, frames: Path, tmp_path: Path) -> None:
        """Mesma conta nos dois lados: o que o fit mediu é o que o rover vai ver."""
        plants = load_plants([frames], DISEASES)
        cal, _ = calibrate(plants, "banana", "banana-foliar", settings(), outer_folds=3)
        evidence, status = apply(procedure_of(cal, DISEASES), plants, settings())
        store = PlantFusionStore(tmp_path / "loja", autosave=False)
        for k, line in enumerate(frames.read_text(encoding="utf-8").splitlines()):
            rec = json.loads(line)
            store.ingest(
                {
                    "field_id": "T",
                    "row_id": "1",
                    "side": "left",
                    "camera_height_mm": rec["camera_height_mm"],
                    "distance_mm": 270.0,
                    "x_m": 0.0,
                    "pass_id": "p1",
                    "timestamp": "2026-10-02T10:00:00Z",
                    "crop": "banana",
                    "tiles": rec["tiles"],
                    "plant_id": rec["plant"],
                    "frame_id": f"q{k}",
                    "camera_id": f"h{round(rec['camera_height_mm'])}",
                },
                calibration=cal,
            )
        prior = logit(0.05)
        for plant, ev, st in zip(plants, evidence, status, strict=True):
            summary = store.plant("T", plant.plant_id)
            theirs = np.array([summary.log_odds[d] - prior for d in DISEASES])
            np.testing.assert_allclose(
                theirs, cal.evidence_offset + cal.evidence_scale * ev, atol=1e-9
            )
            assert summary.status == st

    def test_sem_sadias_suficientes_recusa(self, tmp_path: Path) -> None:
        path = write_frames(tmp_path / "poucas.jsonl", n_healthy=1)
        plants = load_plants([path], DISEASES)
        with pytest.raises(SystemExit):
            calibrate(plants, "banana", "banana-foliar", settings(), outer_folds=3)

    def test_evaluate_mede_sem_reajustar(self, frames: Path, tmp_path: Path) -> None:
        """A validação externa: a calibração de um lote medida em plantas novas."""
        out = tmp_path / "banana.fusion.json"
        main(["fit", "--model", "banana-foliar", "--frames", str(frames), "--out", str(out)])
        novas = write_frames(tmp_path / "novas.jsonl", seed=7)

        result = evaluate(
            parse_args(
                [
                    "evaluate",
                    "--model",
                    "banana-foliar",
                    "--frames",
                    str(novas),
                    "--calibration",
                    str(out),
                ]
            )
        )

        assert result["calibracao"] == load_fusion_calibration(out).id
        assert result["detectadas"] >= 0.7 * result["doentes"]
        assert result["incidencia"]["real"] == pytest.approx(24 / 84)
        lo, hi = result["incidencia"]["ci95"]
        assert lo <= result["incidencia"]["real"] <= hi


class TestRegistroDeCampo:
    """Recalibrar com o rover de verdade: registro de recortes + rótulo do agrônomo."""

    def test_rotulo_por_planta_e_aliases(self, tmp_path: Path) -> None:
        fields_dir = tmp_path / "fields"
        store = PlantFusionStore(fields_dir / "plantas", tile_log=True)
        for x, pid in ((0.0, "B1"), (5.0, "B2"), (10.0, "B3")):
            for i, side in enumerate(("left", "right")):
                store.ingest(
                    frame(
                        x,
                        [tile(0.0, side, "Sigatoka")],
                        side=side,
                        plant_id=pid,
                        frame_id=f"{pid}-{i}",
                        field_id="T9",
                    )
                )
        # O fechamento da passada fundiu B2 em B1 (simulado): o registro guarda o id antigo.
        state = fields_dir / "plantas" / "T9.json"
        data = json.loads(state.read_text(encoding="utf-8"))
        data["aliases"] = {"B2": "B1"}
        state.write_text(json.dumps(data), encoding="utf-8")
        labels = tmp_path / "rotulos.csv"
        with labels.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["field_id", "plant_id", "label"])
            writer.writerow(["T9", "B1", "Sigatoka"])  # B3 ficou sem rótulo

        plants = load_plants(
            [store.tile_log_path("T9")],
            DISEASES,
            labels=read_labels(labels),
            fields_dir=fields_dir,
        )

        assert [p.label for p in plants] == ["Sigatoka"]
        assert plants[0].plant_id == "T9|p1|B1"
        assert len(plants[0].frames) == 4  # os dois quadros de B1 e os dois de B2
        assert math.isclose(plants[0].frames[0].probs[0, DISEASES.index("Sigatoka")], 0.95)
