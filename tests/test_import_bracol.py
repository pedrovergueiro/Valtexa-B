"""Testes do importador do BRACOL e da leitura de zip truncado."""

from __future__ import annotations

import io
import random
import zipfile

from training.import_bracol import STRESS_TO_CLASS, leaf_files_from_zip, read_zip, salvage_zip


def _zip(entries: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class TestSalvageZip:
    def test_reads_every_complete_entry_without_the_central_directory(self):
        data = _zip({"a.txt": b"um" * 100, "b.txt": b"dois" * 100})
        cut = data[: data.find(b"PK\x01\x02")]  # sem diretório central, como o do Mendeley

        assert salvage_zip(cut) == {"a.txt": b"um" * 100, "b.txt": b"dois" * 100}

    def test_stops_at_the_entry_cut_in_half(self):
        data = _zip({"a.txt": b"um" * 100, "b.bin": random.Random(0).randbytes(5000)})
        second = data.find(b"PK\x03\x04", 4)

        assert set(salvage_zip(data[: second + 200])) == {"a.txt"}

    def test_read_zip_prefers_the_central_directory(self):
        data = _zip({"x/y.jpg": b"img"}, method=zipfile.ZIP_STORED)

        assert read_zip(data) == {"x/y.jpg": b"img"}


class TestLeafFiles:
    def test_opens_the_inner_zip_and_keeps_only_the_leaf_folder(self, tmp_path):
        inner = _zip(
            {
                "coffee-datasets/coffee-datasets/leaf/dataset.csv": b"id,predominant_stress\n1,2\n",
                "coffee-datasets/coffee-datasets/leaf/images/1.jpg": b"jpeg",
                "coffee-datasets/coffee-datasets/symptom/x.jpg": b"outro",
            }
        )
        inner = inner[: inner.find(b"PK\x01\x02")]  # o interno é o truncado
        outer = tmp_path / "bracol.zip"
        outer.write_bytes(_zip({"BRACOL.zip": inner}, method=zipfile.ZIP_STORED))

        files = leaf_files_from_zip(outer)

        assert files == {"dataset.csv": b"id,predominant_stress\n1,2\n", "images/1.jpg": b"jpeg"}

    def test_codes_follow_the_columns_of_the_csv(self):
        """miner, rust, phoma, cercospora — a ordem das colunas do dataset.csv."""
        assert [STRESS_TO_CLASS[k] for k in range(1, 5)] == [
            "Leaf_Miner_Damage",
            "Rust",
            "Phoma_Leaf_Spot",
            "Cercospora_Leaf_Spot",
        ]
        assert 5 not in STRESS_TO_CLASS
