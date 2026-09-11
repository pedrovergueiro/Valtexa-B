"""Treino da severidade: o pareamento dos dados e as métricas que importam.

Nada aqui carrega o Keras: é a parte do treino que decide o que entra e como se
mede, e é onde um erro silencioso contaminaria o número final.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from training.train_severity import _augment, evaluate, load_arrays, plantseg_pairs


def _save(path, size=(40, 30), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, (80, 140, 60) if mode == "RGB" else 0).save(path)
    return path


class TestPlantSegPairs:
    def test_keeps_leaf_lesions_and_drops_smut(self, tmp_path):
        """Carvão é doença de espiga: segmentá-lo ensinaria a marcar espiga como lesão."""
        for name in (
            "corn_rust_1",
            "corn_gray_leaf_spot_2",
            "corn_northern_leaf_blight_3",
            "corn_smut_4",
            "apple_scab_5",
        ):
            _save(tmp_path / "images" / "train" / f"{name}.jpg")
            _save(tmp_path / "annotations" / "train" / f"{name}.png", mode="L")

        names = sorted(image.stem for image, _ in plantseg_pairs(tmp_path, "train"))
        assert names == ["corn_gray_leaf_spot_2", "corn_northern_leaf_blight_3", "corn_rust_1"]

    def test_mask_without_image_is_skipped(self, tmp_path):
        _save(tmp_path / "annotations" / "test" / "corn_rust_9.png", mode="L")
        (tmp_path / "images" / "test").mkdir(parents=True)
        assert plantseg_pairs(tmp_path, "test") == []


class TestLoadArrays:
    def test_mask_is_binary_and_resized_with_the_image(self, tmp_path):
        image = _save(tmp_path / "a.jpg", size=(60, 40))
        mask = Image.new("L", (60, 40), 0)
        mask.paste(45, (0, 0, 30, 40))  # 45: o valor de classe que o PlantSeg grava
        mask.save(tmp_path / "a.png")

        images, masks, names = load_arrays([(image, tmp_path / "a.png")], 20)

        assert images.shape == (1, 20, 20, 3)
        assert set(np.unique(masks)) == {0, 1}
        assert masks.mean() == pytest.approx(0.5, abs=0.05)
        assert names == ["a.jpg"]

    def test_healthy_leaf_gets_an_empty_mask(self, tmp_path):
        _, masks, _ = load_arrays([(_save(tmp_path / "sadia.jpg"), None)], 16)
        assert masks.sum() == 0

    def test_misaligned_pair_is_skipped(self, tmp_path):
        """Máscara de tamanho diferente não casa pixel a pixel: melhor perder o par."""
        good = _save(tmp_path / "ok.jpg", size=(40, 30))
        _save(tmp_path / "ok.png", size=(40, 30), mode="L")
        bad = _save(tmp_path / "ruim.jpg", size=(40, 30))
        _save(tmp_path / "ruim.png", size=(30, 40), mode="L")

        _, _, names = load_arrays([(good, tmp_path / "ok.png"), (bad, tmp_path / "ruim.png")], 16)
        assert names == ["ok.jpg"]


class TestAugment:
    def test_image_and_mask_move_together(self):
        images = np.zeros((8, 6, 6, 3), dtype=np.uint8)
        masks = np.zeros((8, 6, 6, 1), dtype=np.uint8)
        images[:, 0, 0] = 255
        masks[:, 0, 0] = 1

        out_images, out_masks = _augment(images, masks, np.random.default_rng(0))

        assert np.array_equal(out_images[..., 0] == 255, out_masks[..., 0] == 1)


class TestEvaluate:
    def test_perfect_prediction(self):
        masks = np.zeros((2, 4, 4, 1), dtype=np.uint8)
        masks[0, :2] = 1
        result = evaluate(masks.astype(np.float32), masks, ["corn_rust_1.jpg", "sadia.jpg"])

        assert result["com_lesao"]["dice_medio"] == 1.0
        assert result["com_lesao"]["erro_absoluto_medio"] == 0.0
        assert result["saudaveis"]["lesao_falsa_media"] == 0.0

    def test_false_lesion_on_a_healthy_leaf_is_reported(self):
        masks = np.zeros((1, 4, 4, 1), dtype=np.uint8)
        predicted = np.zeros((1, 4, 4, 1), dtype=np.float32)
        predicted[0, 0] = 0.9  # uma linha: 4 de 16 pixels

        result = evaluate(predicted, masks, ["sadia.jpg"])

        assert result["saudaveis"]["lesao_falsa_media"] == pytest.approx(0.25)
        assert result["saudaveis"]["com_mais_de_5pct"] == 1.0
        assert result["com_lesao"] == {}
