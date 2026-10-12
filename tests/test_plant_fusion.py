"""Testes da fusão multivista por planta (app/plant_fusion.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from app.plant_fusion import (
    CHANGE_FIRST,
    CHANGE_IMPROVEMENT,
    CHANGE_NEW_INFECTION,
    CROP_CLASSES,
    FRAME_WIDTH_PX,
    HFOV_DEG,
    LLR_BIN_EDGES,
    STATUS_DISEASED,
    STATUS_HEALTHY,
    STATUS_REVIEW,
    STATUS_SUSPECT,
    CropConfig,
    FrameObservation,
    FusionCalibration,
    FusionConfig,
    PlantFusionStore,
    audpc,
    effective_sample_size,
    estimate_incidence,
    ground_sample_distance_mm,
    load_fusion_calibration,
    tile_log_likelihood_ratio,
)

DISTANCE_MM = 270.0
GSD = ground_sample_distance_mm(DISTANCE_MM, HFOV_DEG, FRAME_WIDTH_PX)
TILE_PX = 150.0 / GSD


def banana_probs(disease: str = "Healthy", p: float = 0.95) -> dict[str, float]:
    """Vetor de probabilidades com ``p`` na classe dada e o resto dividido igualmente."""
    classes = CROP_CLASSES["banana"]
    rest = (1.0 - p) / (len(classes) - 1)
    return {c: (p if c == disease else rest) for c in classes}


def tile(
    offset_m: float,
    side: str,
    disease: str = "Healthy",
    p: float = 0.95,
    *,
    in_scope: bool | None = True,
    scene: str | None = "folha",
    needs_review: bool = False,
    zones: list[str] | None = None,
) -> dict[str, Any]:
    """Recorte cujo centro fica ``offset_m`` à frente do eixo óptico (convenção do módulo)."""
    sign = 1.0 if side == "left" else -1.0
    cx = FRAME_WIDTH_PX / 2.0 + sign * offset_m * 1000.0 / GSD
    half = TILE_PX / 2.0
    return {
        "box": [cx - half, 1000.0, cx + half, 1000.0 + TILE_PX],
        "height_mm": 1500.0,
        "zones": zones if zones is not None else ["folhas_baixas"],
        "predicted_class": disease,
        "confidence": p,
        "probabilities": banana_probs(disease, p),
        "prediction_set": None,
        "needs_review": needs_review,
        "in_scope": in_scope,
        "scene": scene,
    }


def frame(
    x_m: float,
    tiles: list[dict[str, Any]],
    *,
    side: str = "left",
    height_mm: float = 1000.0,
    pass_id: str = "p1",
    timestamp: str = "2026-09-01T10:00:00Z",
    row_id: str = "1",
    field_id: str = "talhao-a",
    plant_id: str | None = None,
    frame_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    heading_deg: float | None = None,
) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "row_id": row_id,
        "side": side,
        "camera_height_mm": height_mm,
        "distance_mm": DISTANCE_MM,
        "x_m": x_m,
        "pass_id": pass_id,
        "timestamp": timestamp,
        "crop": "banana",
        "tiles": tiles,
        "plant_id": plant_id,
        "frame_id": frame_id,
        "lat": lat,
        "lon": lon,
        "heading_deg": heading_deg,
    }


CAMERAS = [
    ("left", 450.0),
    ("left", 1000.0),
    ("left", 1540.0),
    ("right", 450.0),
    ("right", 1000.0),
    ("right", 1540.0),
]


def sick_views(
    n_cameras: int, disease: str = "Sigatoka", p: float = 0.95, **kw: Any
) -> list[dict[str, Any]]:
    """Um quadro por câmera, cada um com um recorte doente, todos da mesma planta."""
    out = []
    for i, (side, h) in enumerate(CAMERAS[:n_cameras]):
        out.append(
            frame(
                0.0,
                [tile(0.0, side, disease, p)],
                side=side,
                height_mm=h,
                plant_id="B1",
                frame_id=f"f{i}",
                **kw,
            )
        )
    return out


@pytest.fixture
def store(tmp_path: Path) -> PlantFusionStore:
    return PlantFusionStore(tmp_path / "estado")


def posterior_of(store: PlantFusionStore, frames: list[dict[str, Any]], disease: str) -> float:
    for f in frames:
        store.ingest(f)
    return store.plant("talhao-a", "B1").posteriors[disease]


class TestGeometria:
    """Conversão de coluna de pixel em posição ao longo da linha."""

    def test_gsd_a_270mm(self) -> None:
        """A 270 mm e 127 graus, o quadro de 4056 px cobre ~1,08 m (GSD ~0,267 mm/px)."""
        assert abs(GSD * FRAME_WIDTH_PX - 1083.0) < 2.0

    def test_sinal_por_lado(self) -> None:
        """À direita no quadro = à frente (+x) na câmera esquerda e atrás (-x) na direita."""
        cx = FRAME_WIDTH_PX / 2.0 + 1000.0
        box = [cx - 10, 0, cx + 10, 20]
        base = frame(10.0, [{"box": box, "predicted_class": "Healthy", "probabilities": {}}])
        left = FrameObservation.from_dict(base)
        right = FrameObservation.from_dict({**base, "side": "right"})
        expected = 1000.0 * GSD / 1000.0
        assert left.tile_along_row_m(left.tiles[0]) == pytest.approx(10.0 + expected)
        assert right.tile_along_row_m(right.tiles[0]) == pytest.approx(10.0 - expected)

    def test_lado_invalido(self) -> None:
        """Lado fora de left/right é rejeitado na entrada."""
        with pytest.raises(ValueError):
            FrameObservation.from_dict({**frame(0.0, []), "side": "frente"})


class TestAssociacao:
    """Recorte -> planta: agrupamento 1D estável entre passadas."""

    PLANTS = (0.0, 2.5, 5.0)

    def _pass(self, store: PlantFusionStore, pass_id: str, jitter: float, ts: str) -> set[str]:
        ids: set[str] = set()
        for k, px in enumerate(self.PLANTS):
            for j, dx in enumerate((-0.35, 0.0, 0.35)):
                # O rover para em posições diferentes; os recortes caem espalhados na planta.
                x_m = px + jitter + dx / 2
                obs = frame(
                    x_m,
                    [tile(px + jitter + dx - x_m, "left")],
                    pass_id=pass_id,
                    timestamp=ts,
                    frame_id=f"{pass_id}-{k}-{j}",
                )
                ids.update(store.ingest(obs))
        store.close_pass("talhao-a", pass_id)
        return {store.resolve_plant_id("talhao-a", i) for i in ids}

    def test_ids_estaveis_entre_passadas(self, store: PlantFusionStore) -> None:
        """A segunda passada, com posições deslocadas 20 cm, reencontra as mesmas 3 plantas."""
        first = self._pass(store, "p1", 0.0, "2026-09-01T10:00:00Z")
        second = self._pass(store, "p2", 0.2, "2026-09-08T10:00:00Z")
        assert len(first) == 3
        assert second == first
        assert store.plant_ids("talhao-a") == sorted(first)

    def test_espalhamento_da_folhagem_nao_cria_plantas_extras(
        self, store: PlantFusionStore
    ) -> None:
        """Recortes a +-1 m do centro de uma bananeira (espaçamento 2,5 m) são a mesma planta."""
        for i, off in enumerate((0.9, 0.5, 0.0, -0.5, -0.9)):
            store.ingest(frame(off, [tile(0.0, "left")], frame_id=f"f{i}"))
        store.close_pass("talhao-a", "p1")
        assert len(store.plant_ids("talhao-a")) == 1

    def test_plant_id_explicito_vence(self, store: PlantFusionStore) -> None:
        """Com plant_id na observação, a posição não é usada para associar."""
        store.ingest(frame(0.0, [tile(0.0, "left")], plant_id="PL-7"))
        store.ingest(frame(30.0, [tile(0.0, "left")], plant_id="PL-7", frame_id="x"))
        assert store.plant_ids("talhao-a") == ["PL-7"]

    def test_fora_do_escopo_longe_nao_cria_planta(self, store: PlantFusionStore) -> None:
        """Recorte fora do escopo sem planta por perto é contado no talhão, sem criar planta."""
        store.ingest(frame(0.0, [tile(0.0, "left", in_scope=False, scene="solo")]))
        assert store.plant_ids("talhao-a") == []
        assert store.unassigned_out_of_scope("talhao-a") == {"solo": 1}


class TestFusao:
    """Fusão bayesiana por planta com desconto de correlação."""

    def test_llr_neutra_em_meio_a_meio(self) -> None:
        """p = 0,5 não é evidência; p alta é positiva; p baixa é negativa e mais fraca."""
        prm = CropConfig(CROP_CLASSES["banana"], 2.5, 3.0).params("Cordana")
        assert tile_log_likelihood_ratio(0.5, prm, 3.0) == pytest.approx(0.0)
        pos = tile_log_likelihood_ratio(1.0, prm, 3.0)
        neg = tile_log_likelihood_ratio(0.0, prm, 3.0)
        assert pos > 0 > neg
        assert abs(neg) < pos

    def test_fora_do_escopo_nao_conta(self, tmp_path: Path) -> None:
        """Recortes in_scope=False (pseudocaule) não mudam a posterior, só as contagens."""
        base = [frame(0.0, [tile(0.0, "left")], plant_id="B1", frame_id="f0")]
        extra = frame(
            0.0,
            [
                tile(0.1, "left", "Fusarium_Wilt", 0.99, in_scope=False, scene="pseudocaule"),
                tile(0.2, "left", "Fusarium_Wilt", 0.99, in_scope=False, scene="pseudocaule"),
                tile(0.3, "left", "Sigatoka", 0.99, in_scope=False, scene="fruto"),
            ],
            plant_id="B1",
            frame_id="f1",
        )
        a = PlantFusionStore(tmp_path / "a")
        b = PlantFusionStore(tmp_path / "b")
        pa = posterior_of(a, base, "Fusarium_Wilt")
        pb = posterior_of(b, [*base, extra], "Fusarium_Wilt")
        assert pb == pytest.approx(pa)
        summary = b.plant("talhao-a", "B1")
        assert summary.n_tiles == 1
        assert summary.n_tiles_out_of_scope == 3
        assert summary.out_of_scope_by_scene == {"fruto": 1, "pseudocaule": 2}
        assert summary.incidence == 0.0

    def test_vistas_concordantes_aumentam_posterior(self, tmp_path: Path) -> None:
        """Duas e três câmeras concordando dão posterior maior que uma câmera só."""
        posts = [
            posterior_of(PlantFusionStore(tmp_path / str(n)), sick_views(n), "Sigatoka")
            for n in (1, 2, 3)
        ]
        assert posts[0] < posts[1] < posts[2]

    def test_recortes_sobrepostos_nao_sao_independentes(self, tmp_path: Path) -> None:
        """4 recortes sobrepostos de UM quadro valem mais que 1, mas menos que 4 câmeras."""
        one = posterior_of(PlantFusionStore(tmp_path / "um"), sick_views(1), "Sigatoka")
        overlapping = frame(
            0.0,
            # Passo de 75% do lado do recorte = sobreposição de 25%, como no fatiamento real.
            [tile(k * 0.1125, "left", "Sigatoka", 0.95) for k in range(4)],
            plant_id="B1",
            frame_id="f0",
        )
        same_frame = posterior_of(PlantFusionStore(tmp_path / "mesmo"), [overlapping], "Sigatoka")
        four_cams = posterior_of(PlantFusionStore(tmp_path / "quatro"), sick_views(4), "Sigatoka")
        assert one < same_frame < four_cams

    def test_quadros_consecutivos_da_mesma_camera_sao_descontados(self, tmp_path: Path) -> None:
        """3 quadros seguidos da mesma câmera valem menos que 3 câmeras diferentes."""
        consecutive = [
            frame(0.1 * k, [tile(-0.1 * k, "left", "Sigatoka")], plant_id="B1", frame_id=f"q{k}")
            for k in range(3)
        ]
        same_cam = posterior_of(PlantFusionStore(tmp_path / "a"), consecutive, "Sigatoka")
        three = posterior_of(PlantFusionStore(tmp_path / "b"), sick_views(3), "Sigatoka")
        assert same_cam < three

    def test_desconto_de_correlacao(self) -> None:
        """ESS: independentes somam, cópias perfeitas valem uma."""
        assert effective_sample_size(4, 0.0) == 4
        assert effective_sample_size(4, 1.0) == 1
        assert effective_sample_size(4, 0.5) == pytest.approx(1.6)
        assert effective_sample_size(0, 0.5) == 0

    def test_prior_baixo_exige_mais_evidencia(self, tmp_path: Path) -> None:
        """Com prevalência a priori menor, são precisas mais câmeras para chegar a 'suspeita'."""

        def views_needed(prior: float) -> int:
            for n in range(1, 7):
                crops = {"banana": CropConfig(CROP_CLASSES["banana"], 2.5, 3.0, prior)}
                st = PlantFusionStore(tmp_path / f"{prior}-{n}", FusionConfig(crops=crops))
                if posterior_of(st, sick_views(n), "Sigatoka") >= 0.5:
                    return n
            return 99

        high, low = views_needed(0.20), views_needed(0.01)
        assert high < low < 99

    def test_status_doente_e_zonas(self, store: PlantFusionStore) -> None:
        """Seis câmeras vendo Sigatoka -> 'doente', com zona e incidência registradas."""
        for f in sick_views(6):
            store.ingest(f)
        s = store.plant("talhao-a", "B1")
        assert s.status == STATUS_DISEASED
        assert s.primary_disease == "Sigatoka"
        assert s.incidence == 1.0
        assert s.n_cameras == 6 and s.n_frames == 6
        assert s.zones_by_disease == {"Sigatoka": ["folhas_baixas"]}

    def test_novidade_sem_achado_vira_revisar(self, store: PlantFusionStore) -> None:
        """Planta sem doença, mas com maioria de recortes needs_review, vai para revisão."""
        tiles = [tile(0.1 * k, "left", needs_review=k < 3) for k in range(4)]
        store.ingest(frame(0.0, tiles, plant_id="B1"))
        s = store.plant("talhao-a", "B1")
        assert s.status == STATUS_REVIEW
        assert s.n_needs_review == 3
        assert len(s.needs_review_refs) == 3

    def test_sadia(self, store: PlantFusionStore) -> None:
        """Recortes sadios em várias câmeras -> 'sadia' com posteriores abaixo do prior."""
        for i, (side, h) in enumerate(CAMERAS):
            store.ingest(
                frame(0.0, [tile(0.0, side)], side=side, height_mm=h, plant_id="B1", frame_id=i)
            )
        s = store.plant("talhao-a", "B1")
        assert s.status == STATUS_HEALTHY
        assert all(p < 0.05 for p in s.posteriors.values())


class TestEvolucao:
    """Histórico por passada, mudanças de status e AUDPC."""

    def _ingest_pass(
        self, store: PlantFusionStore, pass_id: str, ts: str, diseases: list[str]
    ) -> None:
        for i, disease in enumerate(diseases):
            side, h = CAMERAS[i % len(CAMERAS)]
            store.ingest(
                frame(
                    0.0,
                    [tile(0.0, side, disease)],
                    side=side,
                    height_mm=h,
                    pass_id=pass_id,
                    timestamp=ts,
                    plant_id="B1",
                    frame_id=f"{pass_id}-{i}",
                )
            )
        store.close_pass("talhao-a", pass_id)

    def test_nova_infeccao_e_melhora(self, store: PlantFusionStore) -> None:
        """sadia -> doente é nova infecção; doente -> sadia é melhora."""
        self._ingest_pass(store, "p1", "2026-09-01T10:00:00Z", ["Healthy"] * 6)
        self._ingest_pass(store, "p2", "2026-09-08T10:00:00Z", ["Sigatoka"] * 6)
        self._ingest_pass(store, "p3", "2026-09-15T10:00:00Z", ["Healthy"] * 6)
        evo = store.evolution("talhao-a", "B1")
        assert [h["status"] for h in evo["history"]] == [
            STATUS_HEALTHY,
            STATUS_DISEASED,
            STATUS_HEALTHY,
        ]
        assert [c["change"] for c in evo["changes"]] == [
            CHANGE_FIRST,
            CHANGE_NEW_INFECTION,
            CHANGE_IMPROVEMENT,
        ]
        alerts = store.changes_in_pass("talhao-a", "p2")
        assert alerts == [
            {
                "plant_id": "B1",
                "from_pass": "p1",
                "to_pass": "p2",
                "from_status": STATUS_HEALTHY,
                "to_status": STATUS_DISEASED,
                "change": CHANGE_NEW_INFECTION,
            }
        ]

    def test_audpc_calculado_a_mao(self, store: PlantFusionStore) -> None:
        """Incidência 0; 0,5; 1,0 nos dias 0, 7, 14: (0+0,5)/2*7 + (0,5+1)/2*7 = 7,0."""
        self._ingest_pass(store, "p1", "2026-09-01T10:00:00Z", ["Healthy", "Healthy"])
        self._ingest_pass(store, "p2", "2026-09-08T10:00:00Z", ["Healthy", "Sigatoka"])
        self._ingest_pass(store, "p3", "2026-09-15T10:00:00Z", ["Sigatoka", "Sigatoka"])
        evo = store.evolution("talhao-a", "B1")
        assert [h["incidence"] for h in evo["history"]] == [0.0, 0.5, 1.0]
        assert evo["audpc"] == pytest.approx(7.0)
        assert audpc([0, 7, 14], [0.0, 0.5, 1.0]) == pytest.approx(7.0)
        assert audpc([0], [1.0]) == 0.0

    def test_passada_fechada_nao_aceita_mais_quadros(self, store: PlantFusionStore) -> None:
        """Depois de close_pass, a passada fica imutável."""
        self._ingest_pass(store, "p1", "2026-09-01T10:00:00Z", ["Healthy"])
        with pytest.raises(ValueError):
            store.ingest(frame(0.0, [tile(0.0, "left")], plant_id="B1", frame_id="novo"))


class TestGeoJSON:
    """Mapa do talhão em GeoJSON."""

    @staticmethod
    def _check_valid(fc: dict[str, Any]) -> None:
        json.dumps(fc)  # serializável
        assert fc["type"] == "FeatureCollection"
        for feat in fc["features"]:
            assert feat["type"] == "Feature"
            assert feat["geometry"]["type"] == "Point"
            coords = feat["geometry"]["coordinates"]
            assert len(coords) == 2 and all(math.isfinite(c) for c in coords)
            for key in (
                "plant_id",
                "status",
                "primary_disease",
                "posterior",
                "incidence",
                "n_views",
                "last_pass",
            ):
                assert key in feat["properties"]

    def test_sem_lat_lon_usa_coordenadas_locais(self, store: PlantFusionStore) -> None:
        """Sem GNSS, as coordenadas são métricas locais e marcadas como tal."""
        store.ingest(frame(0.0, [tile(0.0, "left")], row_id="1"))
        store.ingest(frame(10.0, [tile(0.0, "left")], row_id="2", frame_id="b"))
        fc = store.export_geojson("talhao-a")
        self._check_valid(fc)
        assert fc["valtexa"]["coordinate_system"] == "local_row_metric"
        coords = sorted(f["geometry"]["coordinates"] for f in fc["features"])
        assert coords == [[0.0, 0.0], [10.0, 3.0]]
        assert all(
            f["properties"]["coordinate_system"] == "local_row_metric" for f in fc["features"]
        )

    def test_com_lat_lon_projeta_pelo_rumo(self, store: PlantFusionStore) -> None:
        """Rumo norte, câmera esquerda: planta fica a oeste do rover, ~0,27 m de distância."""
        lat0, lon0 = -22.0, -47.0
        store.ingest(
            frame(0.0, [tile(0.0, "left")], lat=lat0, lon=lon0, heading_deg=0.0, plant_id="B1")
        )
        fc = store.export_geojson("talhao-a")
        self._check_valid(fc)
        assert fc["valtexa"]["coordinate_system"] == "WGS84"
        lon, lat = fc["features"][0]["geometry"]["coordinates"]
        assert lat == pytest.approx(lat0, abs=1e-8)
        assert lon < lon0
        east_m = math.radians(lon - lon0) * 6_371_008.8 * math.cos(math.radians(lat0))
        assert east_m == pytest.approx(-0.27, abs=0.01)

    def test_mistura_cai_para_local(self, store: PlantFusionStore) -> None:
        """Se alguma planta não tem posição geográfica, nenhuma sai como lon/lat."""
        store.ingest(frame(0.0, [tile(0.0, "left")], lat=-22.0, lon=-47.0, heading_deg=90.0))
        store.ingest(frame(10.0, [tile(0.0, "left")], frame_id="b"))
        assert store.export_geojson("talhao-a")["valtexa"]["coordinate_system"] == (
            "local_row_metric"
        )


class TestPersistencia:
    """Estado por talhão em JSON com escrita atômica."""

    def test_ida_e_volta(self, tmp_path: Path) -> None:
        """Um store novo no mesmo diretório reproduz plantas, histórico e associação."""
        root = tmp_path / "estado"
        first = PlantFusionStore(root)
        for f in sick_views(3, pass_id="p1"):
            first.ingest(f)
        first.ingest(frame(5.0, [tile(0.0, "left")], frame_id="outra"))
        first.close_pass("talhao-a", "p1")
        before = [s.to_dict() for s in first.plants("talhao-a")]
        evo_before = first.evolution("talhao-a", "B1")

        files = sorted(p.name for p in root.iterdir())
        assert files == ["talhao-a.json"]  # nenhum temporário largado

        second = PlantFusionStore(root)
        assert second.fields() == ["talhao-a"]
        assert [s.to_dict() for s in second.plants("talhao-a")] == before
        assert second.evolution("talhao-a", "B1") == evo_before

        # A associação continua de onde parou: a planta em x=5 é reencontrada.
        other = next(i for i in second.plant_ids("talhao-a") if i != "B1")
        touched = second.ingest(
            frame(5.3, [tile(0.0, "left")], pass_id="p2", timestamp="2026-09-08T10:00:00Z")
        )
        assert touched == [other]

    def test_nome_de_arquivo_seguro(self, tmp_path: Path) -> None:
        """field_id com barra não escapa do diretório."""
        st = PlantFusionStore(tmp_path)
        st.ingest(frame(0.0, [tile(0.0, "left")], field_id="../fora"))
        assert all(p.parent == tmp_path for p in tmp_path.iterdir())
        assert PlantFusionStore(tmp_path).fields() == ["../fora"]


BANANA_DISEASES = tuple(c for c in CROP_CLASSES["banana"] if c != "Healthy")


def calibration(**changes: Any) -> FusionCalibration:
    """Calibração de mentira: LLR sobe com a probabilidade, -1 embaixo e +2 em cima."""
    n = len(LLR_BIN_EDGES) - 1
    table = tuple(-1.0 + 3.0 * i / (n - 1) for i in range(n))
    fields: dict[str, Any] = {
        "id": "banana-foliar@teste-a",
        "model_key": "banana-foliar",
        "crop": "banana",
        "bin_edges": LLR_BIN_EDGES,
        "llr_tables": {d: table for d in BANANA_DISEASES},
        "llr_clip": 1.5,
        "tile_correlation": 1.0,
        "evidence_offset": -0.5,
        "evidence_scale": 2.0,
        "suspect_evidence": 1.0,
        "diseased_evidence": 3.0,
        "plant_sensitivity": 0.8,
        "plant_false_alarm": 0.05,
        "n_diseased": 100,
        "n_healthy": 200,
    }
    fields.update(changes)
    return FusionCalibration(**fields)


def healthy_views(plant_id: str, tiles_per_frame: int = 1) -> list[dict[str, Any]]:
    """Uma vista sadia por câmera da planta ``plant_id``."""
    return [
        frame(
            0.0,
            [tile(0.0, side)] * tiles_per_frame,
            side=side,
            height_mm=h,
            plant_id=plant_id,
            frame_id=f"{plant_id}-{i}",
        )
        for i, (side, h) in enumerate(CAMERAS)
    ]


class TestCalibracao:
    """Fusão com evidência medida (training.fit_fusion), por passada."""

    def test_llr_da_faixa_limitada(self) -> None:
        cal = calibration()
        table = cal.llr_tables["Sigatoka"]
        assert cal.tile_llr("Sigatoka", 0.0) == pytest.approx(table[0])
        assert cal.tile_llr("Sigatoka", 0.5) == pytest.approx(table[LLR_BIN_EDGES.index(0.5)])
        assert cal.tile_llr("Sigatoka", 1.0) == pytest.approx(1.5)  # +2 cortado no clip

    def test_ida_e_volta_e_arquivo(self, tmp_path: Path) -> None:
        cal = calibration()
        assert FusionCalibration.from_dict(cal.to_dict()) == cal
        path = tmp_path / "m.fusion.json"
        path.write_text(json.dumps({"version": 1, "calibration": cal.to_dict()}))
        assert load_fusion_calibration(path) == cal
        path.write_text(json.dumps({"version": 99, "calibration": cal.to_dict()}))
        with pytest.raises(ValueError, match="versão"):
            load_fusion_calibration(path)

    @pytest.mark.parametrize(
        "changes",
        [
            {"evidence_scale": 0.0},
            {"suspect_evidence": 4.0},
            {"suspect_evidence": -0.5},
            {"plant_false_alarm": 0.9},
            {"llr_tables": {"Sigatoka": (0.0, 1.0)}},
            {"bin_edges": (0.0, 0.5, 0.4, 1.0)},
            {"n_healthy": 0},
        ],
    )
    def test_recusa_calibracao_incoerente(self, changes: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            calibration(**changes)

    def test_platt_e_limiares_na_evidencia(self, store: PlantFusionStore) -> None:
        """Seis câmeras com Sigatoka a 0,95: L = 6 x 1,5; evidência = -0,5 + 2 L."""
        cal = calibration()
        for f in sick_views(6):
            store.ingest(f, calibration=cal)
        s = store.plant("talhao-a", "B1")
        evidence = -0.5 + 2.0 * 6 * 1.5
        expected = 1 / (1 + math.exp(-(math.log(0.05 / 0.95) + evidence)))
        assert s.posteriors["Sigatoka"] == pytest.approx(expected)
        assert s.status == STATUS_DISEASED
        assert s.calibration_id == cal.id

    def test_planta_sadia_fica_sadia(self, store: PlantFusionStore) -> None:
        cal = calibration()
        for f in healthy_views("B1"):
            store.ingest(f, calibration=cal)
        assert store.plant("talhao-a", "B1").status == STATUS_HEALTHY

    def test_limiar_de_suspeita(self, store: PlantFusionStore) -> None:
        """Evidência entre os dois limiares -> suspeita, mesmo com posterior baixa."""
        cal = calibration(suspect_evidence=1.0, diseased_evidence=10.0)
        store.ingest(sick_views(1)[0], calibration=cal)  # evidência -0,5 + 2 x 1,5 = 2,5
        s = store.plant("talhao-a", "B1")
        assert s.status == STATUS_SUSPECT
        assert s.primary_posterior < 0.5

    def test_passada_segue_a_calibracao_com_que_comecou(self, tmp_path: Path) -> None:
        """Trocar a calibração no meio da passada não mistura LLR de modelos diferentes."""
        a, b = calibration(), calibration(id="banana-foliar@teste-b", llr_clip=0.5)
        st = PlantFusionStore(tmp_path / "e")
        views = sick_views(2)
        st.ingest(views[0], calibration=a)
        st.ingest(views[1], calibration=b)  # mesma passada: continua com a
        assert st.plant("talhao-a", "B1").calibration_id == a.id
        nova = sick_views(1, pass_id="p2", timestamp="2026-09-08T10:00:00Z")[0]
        st.ingest(nova, calibration=b)
        assert st.plant("talhao-a", "B1", "p2").calibration_id == b.id
        # E sobrevive a recarregar o estado do disco.
        reloaded = PlantFusionStore(tmp_path / "e")
        assert reloaded.plant("talhao-a", "B1", "p1").calibration_id == a.id

    def test_estado_antigo_sem_calibracao_carrega(self, tmp_path: Path) -> None:
        st = PlantFusionStore(tmp_path)
        st.ingest(sick_views(1)[0])
        path = tmp_path / "talhao-a.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for info in data["passes"].values():
            info.pop("evidence")
        path.write_text(json.dumps(data), encoding="utf-8")
        assert PlantFusionStore(tmp_path).plant("talhao-a", "B1").calibration_id is None

    def test_recusa_calibracao_de_outra_cultura(self, store: PlantFusionStore) -> None:
        with pytest.raises(ValueError, match="citrus"):
            store.ingest(sick_views(1)[0], calibration=calibration(crop="citrus"))
        tables = {d: calibration().llr_tables[d] for d in BANANA_DISEASES[:-1]}
        with pytest.raises(ValueError, match="sem tabela"):
            store.ingest(sick_views(1)[0], calibration=calibration(llr_tables=tables))


class TestIncidencia:
    """Fração de PLANTAS doentes no talhão, corrigida pelos erros medidos."""

    def test_rogan_gladen(self) -> None:
        r = estimate_incidence(25, 100, calibration())
        assert r["method"] == "rogan_gladen"
        assert r["apparent"] == pytest.approx(0.25)
        assert r["incidence"] == pytest.approx((0.25 - 0.05) / (0.8 - 0.05))
        assert r["ci95"][0] < r["incidence"] < r["ci95"][1]

    def test_corta_em_zero_e_sem_calibracao_fica_a_aparente(self) -> None:
        assert estimate_incidence(2, 100, calibration())["incidence"] == 0.0
        raw = estimate_incidence(2, 100, None)
        assert raw["method"] == "aparente" and raw["incidence"] is None
        assert raw["apparent"] == pytest.approx(0.02)
        assert estimate_incidence(0, 0, calibration())["apparent"] is None

    def test_intervalo_encolhe_com_mais_plantas_rotuladas(self) -> None:
        small = estimate_incidence(25, 100, calibration(n_diseased=20, n_healthy=40))
        large = estimate_incidence(25, 100, calibration(n_diseased=2000, n_healthy=4000))
        assert large["ci95"][1] - large["ci95"][0] < small["ci95"][1] - small["ci95"][0]

    def test_conta_plantas_e_nao_recortes(self, store: PlantFusionStore) -> None:
        cal = calibration()
        for f in sick_views(6):
            store.ingest(f, calibration=cal)
        for pid in ("B2", "B3"):
            for f in healthy_views(pid, tiles_per_frame=5):
                store.ingest(f, calibration=cal)
        r = store.incidence("talhao-a")
        assert r["n_plants"] == 3 and r["n_judged"] == 3
        assert r["status_counts"][STATUS_DISEASED] == 1
        assert r["apparent"] == pytest.approx(1 / 3)
        assert r["incidence"] == pytest.approx((1 / 3 - 0.05) / 0.75)
        assert r["flagged_by_disease"] == {"Sigatoka": 1}
        assert r["calibration_id"] == cal.id

    def test_passada_desconhecida(self, store: PlantFusionStore) -> None:
        store.ingest(sick_views(1)[0])
        with pytest.raises(KeyError):
            store.incidence("talhao-a", "nao-existe")
        assert store.incidence("talhao-a")["method"] == "aparente"


class TestRegistroDeRecortes:
    """O dado da validação de campo: cada recorte com a planta a que foi associado."""

    def test_grava_recortes_no_escopo_com_a_planta(self, tmp_path: Path) -> None:
        st = PlantFusionStore(tmp_path, tile_log=True)
        st.ingest(
            frame(
                0.0,
                [tile(0.0, "left", "Sigatoka"), tile(0.1, "left", in_scope=False, scene="solo")],
                plant_id="B1",
                frame_id="q1",
            ),
            calibration=calibration(),
        )
        lines = st.tile_log_path("talhao-a").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["plant"] == "B1" and rec["frame"] == "q1" and rec["pass_id"] == "p1"
        assert rec["calibration_id"] == "banana-foliar@teste-a"
        assert len(rec["tiles"]) == 1
        assert rec["tiles"][0]["probabilities"]["Sigatoka"] == pytest.approx(0.95)

    def test_desligado_nao_grava(self, store: PlantFusionStore) -> None:
        store.ingest(sick_views(1)[0])
        assert not store.tile_log_path("talhao-a").exists()
