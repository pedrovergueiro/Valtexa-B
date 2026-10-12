"""``training.import_campo``: planta sadia vira Healthy; planta doente vai para revisão."""

from __future__ import annotations

import csv
import json

from PIL import Image

from training import import_campo


def test_recortes_por_planta_rotulada(tmp_path) -> None:
    originais = tmp_path / "originais"
    originais.mkdir()
    Image.new("RGB", (600, 400), (30, 140, 30)).save(originais / "r-1.jpg")
    Image.new("RGB", (600, 400), (150, 90, 30)).save(originais / "r-2.jpg")
    registro = tmp_path / "registro.jsonl"

    def tiles(p_ferrugem: list[float]) -> list[dict]:
        return [{"box": [i * 150, 0, i * 150 + 150, 150],
                 "probabilities": {"Rust": p, "Healthy": 1 - p}} for i, p in enumerate(p_ferrugem)]

    linhas = [
        {"field_id": "T1", "pass_id": "m-1", "plant": "L03-P0001", "frame": "r-1",
         "tiles": tiles([0.1, 0.2])},
        {"field_id": "T1", "pass_id": "m-1", "plant": "L03-P0002", "frame": "r-2",
         "tiles": tiles([0.9, 0.3, 0.7])},
        {"field_id": "T1", "pass_id": "m-1", "plant": "L03-P0009", "frame": "r-2",
         "tiles": tiles([0.9])},  # sem rótulo
        {"field_id": "T1", "pass_id": "m-1", "plant": "L03-P0001", "frame": "r-9",
         "tiles": tiles([0.1])},  # original não exportado
    ]
    registro.write_text("".join(json.dumps(x) + "\n" for x in linhas), encoding="utf-8")
    labels = {("T1", "L03-P0001", "m-1"): "Healthy", ("T1", "L03-P0002", "m-1"): "Rust"}

    contagem = import_campo.importa(labels, [registro], originais, tmp_path / "ds", limiar=0.5)
    assert contagem == {"Healthy": 2, "revisar:Rust": 2, "doente_descartado": 1,
                        "sem_rotulo": 1, "sem_original": 1}
    assert len(list((tmp_path / "ds" / "Healthy").glob("*.jpg"))) == 2
    assert not (tmp_path / "ds" / "Rust").exists()  # doente só entra depois da revisão
    with (tmp_path / "ds" / "_revisar" / "manifesto.csv").open(encoding="utf-8") as fh:
        manifesto = list(csv.DictReader(fh))
    assert [m["prob_modelo"] for m in manifesto] == ["0.9000", "0.7000"]
    with Image.open(tmp_path / "ds" / "_revisar" / manifesto[0]["arquivo"]) as im:
        assert im.size == (150, 150)
