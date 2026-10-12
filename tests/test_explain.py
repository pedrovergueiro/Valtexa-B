"""Testes da explicação por oclusão."""

from __future__ import annotations

import base64
import io
import sys

import numpy as np
import pytest
from PIL import Image

from app.inference.explain import (
    OcclusionMap,
    cell_edges,
    occlusion_map,
    region_box,
    render_overlay,
)
from app.registry import get_spec, registry
from tests.conftest import DEFAULT_MODEL, FakeKerasModel, make_image_bytes

CLASSES = get_spec(DEFAULT_MODEL).class_names
SIZE = 224
REGION = 56
"""O "campo de visão" do dublê: só o canto superior esquerdo decide."""


def corner_predictor(batch: np.ndarray) -> np.ndarray:
    """Probabilidade da última classe = brilho médio do canto superior esquerdo."""
    corner = (np.asarray(batch)[:, :REGION, :REGION] + 1.0) / 2.0
    last = np.clip(corner.mean(axis=(1, 2, 3)), 0.0, 1.0)
    rest = (1.0 - last) / 3.0
    return np.stack([rest, rest, rest, last], axis=1)


class CornerModel(FakeKerasModel):
    """O mesmo dublê, com a superfície que o ``CornDiseaseModel`` consome."""

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        self.batch_shapes.append(batch.shape)
        return corner_predictor(batch)


def _lesion_in_the_corner() -> np.ndarray:
    """Imagem preta com um quadrado branco de 32 px no canto — a célula (0, 0)."""
    image = np.full((SIZE, SIZE, 3), -1.0, dtype=np.float32)
    image[:32, :32] = 1.0
    return image


def _png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _corner_upload(width: int = SIZE, height: int = SIZE) -> bytes:
    pixels = np.zeros((height, width, 3))
    pixels[: height // 7, : width // 7] = 255
    return _png(pixels)


class TestOcclusionMap:
    def test_points_to_the_region_that_decides(self):
        occlusion = occlusion_map(corner_predictor, _lesion_in_the_corner(), class_index=3)

        assert occlusion.peak == (0, 0)
        assert occlusion.max_drop > 0.1

    def test_regions_the_model_does_not_see_weigh_nothing(self):
        occlusion = occlusion_map(corner_predictor, _lesion_in_the_corner(), class_index=3)

        assert occlusion.drops[6, 6] == pytest.approx(0.0)
        assert occlusion.drops[3, 3] == pytest.approx(0.0)

    def test_covering_the_background_can_argue_for_the_class(self):
        """Cinza sobre preto clareia o canto: evidência contra, queda negativa."""
        occlusion = occlusion_map(corner_predictor, _lesion_in_the_corner(), class_index=3)

        assert occlusion.drops[0, 1] < 0

    def test_a_model_reading_the_whole_frame_has_no_decisive_region(self):
        """O atalho do rover-understory: a resposta vem da cor média da foto."""

        def mean_color(batch):
            last = np.clip((np.asarray(batch).mean(axis=(1, 2, 3)) + 1.0) / 2.0, 0.0, 1.0)
            rest = (1.0 - last) / 3.0
            return np.stack([rest, rest, rest, last], axis=1)

        bright = np.full((SIZE, SIZE, 3), 0.8, dtype=np.float32)
        occlusion = occlusion_map(mean_color, bright, class_index=3)

        assert occlusion.base_probability == pytest.approx(0.9)
        assert occlusion.max_drop < 0.02

    def test_reuses_a_baseline_the_caller_already_has(self):
        calls = []

        def counting(batch):
            calls.append(len(batch))
            return corner_predictor(batch)

        occlusion_map(
            counting, _lesion_in_the_corner(), class_index=3, grid=4, base_probability=0.5
        )

        assert sum(calls) == 16, "só as células; a linha de base veio de fora"

    def test_bounds_memory_with_chunks(self):
        sizes = []

        def recording(batch):
            sizes.append(len(batch))
            return corner_predictor(batch)

        occlusion_map(recording, _lesion_in_the_corner(), class_index=3, grid=8)

        assert max(sizes) <= 32
        assert sum(sizes) == 64 + 1

    @pytest.mark.parametrize("grid", [0, 300])
    def test_refuses_an_impossible_grid(self, grid):
        with pytest.raises(ValueError):
            occlusion_map(corner_predictor, _lesion_in_the_corner(), class_index=3, grid=grid)

    def test_refuses_a_class_the_model_does_not_output(self):
        with pytest.raises(ValueError):
            occlusion_map(corner_predictor, _lesion_in_the_corner(), class_index=9)


class TestGeometry:
    @pytest.mark.parametrize(("size", "grid"), [(224, 7), (225, 7), (100, 3)])
    def test_cells_cover_the_axis_without_gaps(self, size, grid):
        edges = cell_edges(size, grid)

        assert edges[0] == 0 and edges[-1] == size
        assert np.all(np.diff(edges) > 0)

    def test_maps_a_cell_back_to_the_original_photo(self):
        """O pré-processamento estica para o quadrado; a célula volta retângulo."""
        assert region_box(1, 2, grid=4, width=640, height=480) == [320, 120, 480, 240]


class TestOverlay:
    def _map(self, drops: np.ndarray) -> OcclusionMap:
        return OcclusionMap(class_index=3, base_probability=0.8, drops=drops)

    def test_is_a_png_no_larger_than_asked(self):
        photo = Image.new("RGB", (1200, 800), (40, 120, 40))
        png = render_overlay(photo, self._map(np.eye(4) * 0.5), max_side=300)

        rendered = Image.open(io.BytesIO(png))
        assert rendered.format == "PNG"
        assert max(rendered.size) == 300

    def test_no_evidence_paints_nothing(self):
        """Intensidade absoluta: um mapa sem região decisiva não pode sair quente."""
        photo = Image.new("RGB", (70, 70), (40, 120, 40))
        png = render_overlay(photo, self._map(np.full((7, 7), 0.001)))

        assert (
            np.abs(
                np.asarray(Image.open(io.BytesIO(png)), dtype=int) - np.asarray(photo, dtype=int)
            ).max()
            <= 1
        )

    def test_paints_the_decisive_cell_red(self):
        photo = Image.new("RGB", (70, 70), (40, 120, 40))
        drops = np.zeros((7, 7))
        drops[0, 0] = 0.8

        pixels = np.asarray(Image.open(io.BytesIO(render_overlay(photo, self._map(drops)))))
        assert pixels[5, 5, 0] > 150, "a célula decisiva fica vermelha"
        assert tuple(pixels[65, 65]) == (40, 120, 40), "o resto fica como estava"


class TestExplainApi:
    @pytest.fixture
    def corner_model(self, client) -> CornerModel:
        double = CornerModel()
        registry.get(DEFAULT_MODEL).attach(double, CLASSES)
        return double

    def _post(self, client, data: bytes | None = None, **params):
        return client.post(
            "/predict/explain",
            params=params,
            files={"file": ("folha.png", data or _corner_upload(), "image/png")},
        )

    def test_explains_the_predicted_class(self, client, corner_model):
        body = self._post(client, grid=7).json()

        assert body["target_class"] == body["prediction"]["predicted_class"]
        assert body["method"] == "occlusion"
        assert len(body["drops"]) == 7 and all(len(row) == 7 for row in body["drops"])
        assert body["passes"] == 50

    def test_locates_the_evidence_in_the_uploaded_photo(self, client, corner_model):
        body = self._post(client, _corner_upload(width=700, height=350), grid=7).json()

        assert body["peak_region"] == [0, 0, 100, 50]

    def test_costs_exactly_the_passes_it_reports(self, client, corner_model):
        body = self._post(client, grid=5).json()

        assert sum(shape[0] for shape in corner_model.batch_shapes) == body["passes"]

    def test_returns_a_displayable_overlay(self, client, corner_model):
        body = self._post(client).json()

        image = Image.open(io.BytesIO(base64.b64decode(body["overlay_png"])))
        assert image.format == "PNG"

    def test_overlay_can_be_skipped(self, client, corner_model):
        assert self._post(client, overlay=False).json()["overlay_png"] is None

    def test_answers_why_not_another_class(self, client, corner_model):
        body = self._post(client, target_class=CLASSES[0]).json()

        assert body["target_class"] == CLASSES[0]

    def test_rejects_a_class_the_model_does_not_have(self, client, corner_model):
        response = self._post(client, target_class="Cigarrinha")

        assert response.status_code == 422
        assert "Classes" in response.json()["detail"]

    @pytest.mark.parametrize("grid", [2, 15])
    def test_rejects_a_grid_out_of_range(self, client, corner_model, grid):
        assert self._post(client, grid=grid).status_code == 422

    def test_needs_a_loaded_model(self, client):
        assert self._post(client).status_code == 503

    def test_never_uses_tta(self, client, corner_model, monkeypatch):
        """O mapa explica uma passada; a média de quatro seria outra resposta."""
        from app.config import settings

        monkeypatch.setattr(settings, "inference_tta", True)
        body = self._post(client).json()

        assert body["prediction"]["view_agreement"] is None


class TestAuditCli:
    def test_summarises_per_class_and_writes_overlays(self, client, tmp_path):
        from training.audit_attention import audit, sample_per_class, summarize

        registry.get(DEFAULT_MODEL).attach(CornerModel(), CLASSES)
        data_dir = tmp_path / "data"
        for class_name in (CLASSES[0], CLASSES[-1]):
            (data_dir / class_name).mkdir(parents=True)
            for index in range(3):
                (data_dir / class_name / f"{index}.png").write_bytes(_corner_upload())

        items = sample_per_class(data_dir, per_class=2, seed=0)
        # Grade 7: a célula (0, 0) tem 32 px, o tamanho exato do quadrado claro.
        rows = audit(registry.get(DEFAULT_MODEL), items, grid=7, out_dir=tmp_path / "out")
        summary = {line["class_name"]: line for line in summarize(rows)}

        assert len(rows) == 4, "duas por classe"
        assert summary[CLASSES[-1]]["accuracy"] == 1.0
        assert summary[CLASSES[0]]["accuracy"] == 0.0
        assert summary[CLASSES[-1]]["median_max_drop"] > 0.05
        assert len(list((tmp_path / "out").rglob("*.png"))) == 4

    def test_help_does_not_load_a_model(self, monkeypatch):
        from training import audit_attention

        monkeypatch.setattr(sys, "argv", ["audit_attention", "--help"])
        with pytest.raises(SystemExit) as exit_info:
            audit_attention.parse_args()
        assert exit_info.value.code == 0


def test_the_default_upload_is_not_mistaken_for_a_corner(client):
    """Sanidade do dublê: sem canto claro, a última classe perde."""
    registry.get(DEFAULT_MODEL).attach(CornerModel(), CLASSES)
    body = client.post(
        "/predict", files={"file": ("x.jpg", make_image_bytes(color=(0, 0, 0)), "image/jpeg")}
    ).json()

    assert body["predicted_class"] != CLASSES[-1]


def two_lesions(batch: np.ndarray) -> np.ndarray:
    """Doença com lesões redundantes: basta um dos dois cantos claro."""
    bright = (np.asarray(batch) + 1.0) / 2.0
    corner_a = bright[:, :32, :32].mean(axis=(1, 2, 3))
    corner_b = bright[:, -32:, -32:].mean(axis=(1, 2, 3))
    last = np.maximum(corner_a, corner_b)
    rest = (1.0 - last) / 3.0
    return np.stack([rest, rest, rest, last], axis=1)


def _two_lesion_image() -> np.ndarray:
    image = np.full((SIZE, SIZE, 3), -1.0, dtype=np.float32)
    image[:32, :32] = 1.0
    image[-32:, -32:] = 1.0
    return image


class TestRise:
    def test_occlusion_is_blind_to_redundant_lesions(self):
        """O defeito medido na ferrugem: cobrir uma lesão não muda nada."""
        occlusion = occlusion_map(two_lesions, _two_lesion_image(), class_index=3)

        assert occlusion.max_drop == pytest.approx(0.0, abs=1e-6)

    def test_rise_finds_both_redundant_lesions(self):
        from app.inference.explain import rise_map

        rise = rise_map(two_lesions, _two_lesion_image(), class_index=3)

        assert rise.drops[0, 0] > 0.15 and rise.drops[6, 6] > 0.15
        assert abs(rise.drops[3, 3]) < 0.06, "onde não há lesão, não há evidência"

    def test_is_reproducible(self):
        from app.inference.explain import rise_map

        first = rise_map(two_lesions, _two_lesion_image(), class_index=3, samples=64)
        second = rise_map(two_lesions, _two_lesion_image(), class_index=3, samples=64)

        np.testing.assert_array_equal(first.drops, second.drops)

    def test_refuses_too_few_masks(self):
        from app.inference.explain import rise_map

        with pytest.raises(ValueError):
            rise_map(two_lesions, _two_lesion_image(), class_index=3, samples=1)

    def test_is_available_through_the_api(self, client):
        double = CornerModel()
        registry.get(DEFAULT_MODEL).attach(double, CLASSES)

        body = client.post(
            "/predict/explain",
            params={"method": "rise", "samples": 64, "overlay": False},
            files={"file": ("folha.png", _corner_upload(), "image/png")},
        ).json()

        assert body["method"] == "rise"
        assert body["passes"] == 65
        assert sum(shape[0] for shape in double.batch_shapes) == body["passes"]
        assert body["peak_region"][:2] == [0, 0]
