"""Testes da rota do quadro do rover."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.framing import framing_model
from app.registry import get_spec, registry
from tests.conftest import FakeKerasModel, make_image_bytes
from tests.test_framing import _axis, _reference

BANANA = "banana-foliar"
CLASSES = get_spec(BANANA).class_names


class HealthyModel(FakeKerasModel):
    """Dublê que vê folha sadia em todo recorte."""

    def __call__(self, batch, training=False):
        rows = super().__call__(batch, training)
        rows[:] = 0.0
        rows[:, CLASSES.index("Healthy")] = 1.0
        return rows


def _attach(healthy: bool = False) -> FakeKerasModel:
    """Dublê que responde Sigatoka (a última classe) — ou Healthy, se pedido."""
    double = (HealthyModel if healthy else FakeKerasModel)(num_classes=len(CLASSES))
    registry.get(BANANA).attach(double, CLASSES)
    return double


def _post(client, **params):
    query = {"model": BANANA, "camera_height_mm": 450, "distance_mm": 200, **params}
    return client.post(
        "/predict/rover-frame",
        params=query,
        files={"file": ("quadro.jpg", make_image_bytes(size=(1000, 750)), "image/jpeg")},
    )


class TestRoverFrame:
    def test_cuts_the_frame_into_physical_scale_tiles(self, client):
        _attach()
        body = _post(client).json()

        assert body["crop"] == "banana"
        assert body["frame_size"] == [1000, 750]
        assert body["tile_px"] > 100
        assert body["tiles_analyzed"] > 1

    def test_reports_findings_by_zone_of_the_plant(self, client):
        _attach()
        body = _post(client, only_findings=True).json()

        assert body["class_counts"] == {"Sigatoka": body["tiles_analyzed"]}
        assert set(body["zone_summary"]) <= {"pseudocaule", "folhas_baixas"}
        assert all(t["height_mm"] >= 0 and t["zones"] for t in body["tiles"])
        assert len(body["tiles"]) == body["tiles_analyzed"], "todo recorte tem Sigatoka"

    def test_healthy_tiles_are_left_out_of_findings(self, client):
        _attach(healthy=True)
        body = _post(client).json()

        assert body["tiles"] == []
        assert all(z["com_achado"] == 0 for z in body["zone_summary"].values())

    def test_all_tiles_on_request(self, client):
        _attach(healthy=True)
        body = _post(client, only_findings=False).json()

        assert len(body["tiles"]) == body["tiles_analyzed"]

    def test_passes_are_bounded(self, client, monkeypatch):
        """Centenas de recortes, mas nunca mais que MAX_FORWARD_BATCH por passada."""
        monkeypatch.setattr(settings, "max_forward_batch", 4)
        double = _attach()
        _post(client)

        assert max(shape[0] for shape in double.batch_shapes) <= 4

    def test_refuses_a_distance_that_makes_tiles_too_small(self, client):
        _attach()
        response = _post(client, distance_mm=4000)

        assert response.status_code == 422
        assert "menos que" in response.json()["detail"]

    def test_needs_the_model_loaded(self, client):
        assert _post(client).status_code == 503

    @pytest.mark.parametrize("param", ["camera_height_mm", "distance_mm"])
    def test_geometry_is_required(self, client, param):
        _attach()
        query = {"model": BANANA, "camera_height_mm": 450, "distance_mm": 200}
        del query[param]
        response = client.post(
            "/predict/rover-frame",
            params=query,
            files={"file": ("q.jpg", make_image_bytes(), "image/jpeg")},
        )
        assert response.status_code == 422


@pytest.fixture
def gate_sees():
    """Portão dublê que vê a mesma cena em todo recorte."""

    def attach(scene: str):
        framing_model.attach(lambda images: np.array([_axis(scene)] * len(images)), _reference())

    yield attach
    framing_model.unload()


class TestRoverFrameWithTheGate:
    def test_a_pseudostem_is_not_diagnosed_as_a_leaf_disease(self, client, gate_sees):
        """O modelo foliar diria Sigatoka; o portão diz que não é folha."""
        _attach()
        gate_sees("pseudocaule")

        body = _post(client, only_findings=True).json()

        assert body["tiles"] == [], "recorte recusado não é achado"
        assert body["class_counts"] == {}
        for zone, summary in body["zone_summary"].items():
            assert summary["com_achado"] == 0
            assert summary["fora_de_escopo"] == summary["recortes"]
            assert body["out_of_scope"][zone] == {"pseudocaule": summary["recortes"]}

    def test_a_leaf_in_scope_is_diagnosed_as_before(self, client, gate_sees):
        _attach()
        gate_sees("folha_banana")

        body = _post(client, only_findings=True).json()

        assert len(body["tiles"]) == body["tiles_analyzed"]
        assert all(t["in_scope"] and t["scene"] == "folha_banana" for t in body["tiles"])
        assert body["out_of_scope"] == {}

    def test_without_the_gate_nothing_is_refused(self, client):
        _attach()
        body = _post(client).json()

        assert all(t["in_scope"] is None for t in body["tiles"])
        assert all(z["fora_de_escopo"] == 0 for z in body["zone_summary"].values())


class TestRoverTta:
    def test_the_rover_predicts_in_a_single_pass_by_default(self, client):
        """O TTA melhora o recorte mas piora a planta: a fusão foi ajustada com saídas de
        uma passada só (29 para 24 de 33 plantas de citros detectadas, medido)."""
        double = _attach()
        tiles = _post(client).json()["tiles_analyzed"]

        assert sum(shape[0] for shape in double.batch_shapes) == tiles

    def test_tta_can_be_turned_on_per_request(self, client):
        double = _attach()
        tiles = _post(client, tta=True).json()["tiles_analyzed"]

        assert sum(shape[0] for shape in double.batch_shapes) == 4 * tiles

    def test_tta_can_be_turned_on_by_setting(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rover_tta", True)
        double = _attach()
        tiles = _post(client).json()["tiles_analyzed"]

        assert sum(shape[0] for shape in double.batch_shapes) == 4 * tiles


class TestRoverChunking:
    def test_no_call_to_the_model_gets_more_tiles_than_the_cap(self, client, monkeypatch):
        """Memória do computador: o TTA empilha quatro vistas do lote que receber."""
        import app.rover_api as rover_api

        monkeypatch.setattr(rover_api, "TILES_PER_CALL", 5)
        double = _attach()
        tiles = _post(client, tta=False).json()["tiles_analyzed"]

        assert tiles > 5, "o quadro de teste precisa ter mais recortes que o teto"
        assert max(shape[0] for shape in double.batch_shapes) <= 5
        assert sum(shape[0] for shape in double.batch_shapes) == tiles

    def test_chunking_does_not_change_any_answer(self, client, monkeypatch):
        import app.rover_api as rover_api

        _attach()
        whole = _post(client, only_findings=False).json()
        monkeypatch.setattr(rover_api, "TILES_PER_CALL", 3)
        chunked = _post(client, only_findings=False).json()

        assert chunked["tiles"] == whole["tiles"]
        assert chunked["class_counts"] == whole["class_counts"]


class TestRoverFusionGetsTheOriginalSets:
    """Alargar o conjunto com o campo não pode mudar o que a fusão por planta recebe."""

    def test_the_response_is_honest_and_the_fusion_gets_the_original(
        self, client, tmp_path, monkeypatch
    ):
        import app.rover_api as rover_api
        from app.inference.conformal import ConformalReference

        monkeypatch.setattr(settings, "models_dir", tmp_path)
        ConformalReference(
            class_names=tuple(CLASSES),
            alpha=0.1,
            thresholds={"plain": [1.0] * len(CLASSES), "tta": [1.0] * len(CLASSES)},
            calibrated_on=10,
            class_counts=[1] * len(CLASSES),
            field_calibrated_on=5,
            base_thresholds={"plain": [0.0] * len(CLASSES), "tta": [0.0] * len(CLASSES)},
        ).save(get_spec(BANANA).conformal_path)
        _attach()  # o dublê dá 1,0 à última classe: com limiar 0 só ela entra; com 1, todas

        seen: list[dict] = []

        def spy(spec, frame, in_scope):
            seen.append(frame)
            return [], None

        monkeypatch.setattr(rover_api.fields, "record_frame", spy)
        monkeypatch.setattr(rover_api.fields, "directory", tmp_path / "talhoes")
        body = _post(
            client,
            only_findings=False,
            field_id="T",
            row_id="L1",
            side="left",
            x_m=1.0,
            pass_id="P1",
        ).json()

        assert all(len(tile["prediction_set"]) == len(CLASSES) for tile in body["tiles"])
        [frame] = seen
        assert all(obs["prediction_set"] == [CLASSES[-1]] for obs in frame["tiles"])


class TestRoverFeedsTheReviewQueue:
    """Sem esta ligação, o dataset próprio do rover nunca começava a existir."""

    @pytest.fixture
    def queue(self, tmp_path, monkeypatch):
        import app.main
        from app.review import ReviewQueue

        queue = ReviewQueue(tmp_path / "revisao")
        monkeypatch.setattr(app.main, "review_queue", queue)
        return queue

    def test_a_refused_tile_reaches_the_queue_with_where_it_came_from(
        self, client, gate_sees, queue
    ):
        _attach()
        gate_sees("pseudocaule")

        body = _post(client, frame_id="Q1", camera_id="esq-450").json()

        assert body["queued_for_review"] == 1, "um só recorte recusado por quadro"
        [item] = queue.pending()
        assert item["original_filename"].startswith("Q1#recorte_")
        assert item["context"]["source"] == "rover"
        assert item["context"]["camera_id"] == "esq-450"
        assert item["context"]["scene"] == "pseudocaule"
        assert "pseudocaule" in item["context"]["zones"]

        # E o agrônomo vê isso pela API, não só em disco.
        [listed] = client.get("/review").json()
        assert listed["context"]["source"] == "rover"
        assert listed["context"]["camera_id"] == "esq-450"

    def test_without_client_consent_nothing_is_queued(self, client, gate_sees, queue):
        """Brief, seção 32: com X-Training-Consent: 0 o quadro é diagnosticado e
        não vai para a fila que alimenta o treino."""
        _attach()
        gate_sees("pseudocaule")
        query = {"model": BANANA, "camera_height_mm": 450, "distance_mm": 200}
        resposta = client.post(
            "/predict/rover-frame",
            params=query,
            files={"file": ("quadro.jpg", make_image_bytes(size=(1000, 750)), "image/jpeg")},
            headers={"X-Training-Consent": "0"},
        )
        assert resposta.status_code == 200
        assert resposta.json()["queued_for_review"] == 0
        assert queue.pending() == []

    def test_with_client_consent_the_frame_is_queued(self, client, gate_sees, queue):
        _attach()
        gate_sees("pseudocaule")
        query = {"model": BANANA, "camera_height_mm": 450, "distance_mm": 200}
        resposta = client.post(
            "/predict/rover-frame",
            params=query,
            files={"file": ("quadro.jpg", make_image_bytes(size=(1000, 750)), "image/jpeg")},
            headers={"X-Training-Consent": "1"},
        )
        assert resposta.json()["queued_for_review"] == 1

    def test_a_plain_healthy_frame_queues_nothing(self, client, gate_sees, queue):
        _attach(healthy=True)
        gate_sees("folha_banana")

        assert _post(client).json()["queued_for_review"] == 0
        assert queue.pending() == []

    def test_the_cap_per_frame_holds(self):
        from app.rover_api import _review_candidates
        from app.schemas import RoverTile

        def tile(confidence, *, novel=False, ambiguous=False, refused=False):
            return RoverTile(
                box=[0, 0, 1, 1],
                height_mm=500.0,
                zones=["pseudocaule"],
                predicted_class="Sigatoka",
                confidence=confidence,
                prediction_set=["Sigatoka", "Cordana"] if ambiguous else ["Sigatoka"],
                needs_review=novel,
                in_scope=not refused,
            )

        tiles = [
            tile(0.9, refused=True),
            tile(0.8, ambiguous=True),
            tile(0.99),
            tile(0.7, novel=True),
            tile(0.6, refused=True),
            tile(0.5, ambiguous=True),
        ]

        assert _review_candidates(tiles, 3) == [3, 5, 1]
        assert _review_candidates(tiles, 10) == [3, 5, 1, 0]
        assert _review_candidates(tiles, 0) == []
