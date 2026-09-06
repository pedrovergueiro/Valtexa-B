"""Testes do catálogo de modelos e da consolidação de datasets."""

from __future__ import annotations

import pytest

from app.config import settings
from app.registry import (
    MODEL_SPECS,
    SOURCE_CLASS_ALIASES,
    ModelRegistry,
    Platform,
    get_spec,
    normalize_class_name,
    spec_for_platform,
)
from tests.conftest import make_image_bytes
from training.prepare_dataset import execute_plan, find_class_dirs, plan_dataset


class TestCatalogue:
    def test_every_spec_key_matches_its_dict_entry(self):
        assert all(key == spec.key for key, spec in MODEL_SPECS.items())

    def test_specs_have_distinct_weight_files(self):
        """Dois modelos gravando no mesmo arquivo sobrescreveriam um ao outro."""
        names = [spec.weights_name for spec in MODEL_SPECS.values()]
        assert len(names) == len(set(names))

    def test_specs_declare_at_least_two_classes(self):
        assert all(spec.num_classes >= 2 for spec in MODEL_SPECS.values())

    def test_labels_path_sits_next_to_the_weights(self):
        spec = get_spec("rover-understory")
        assert spec.labels_path.parent == spec.weights_path.parent
        assert spec.labels_path.name == "rover_understory.labels.json"

    def test_default_model_path_honours_the_env_override(self, tmp_path, monkeypatch):
        """Deploys antigos apontam MODEL_PATH e precisam continuar funcionando."""
        override = tmp_path / "pesos.keras"
        monkeypatch.setattr(settings, "model_path", override)
        assert get_spec(settings.default_model).weights_path == override

    def test_other_models_ignore_the_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "model_path", tmp_path / "pesos.keras")
        assert get_spec("aerial-canopy").weights_path.name == "aerial_canopy.keras"

    def test_rover_covers_foliar_and_understory_pathologies(self):
        """O rover vê as foliares antes do drone, e o que o drone nunca vê."""
        classes = set(get_spec("rover-understory").class_names)
        assert {"Gray_Leaf_Spot", "Blight"} <= classes
        assert {"Stalk_Rot", "Corn_Stunt"} <= classes

    def test_coffee_is_trainable_today_like_leaf_foliar(self):
        """Café tem dataset público maduro — não deveria depender de coleta."""
        coffee = get_spec("coffee-foliar")
        assert coffee.needs_field_data is False
        assert coffee.platform == Platform.ANY
        assert set(coffee.class_names) == {
            "Healthy",
            "Rust",
            "Cercospora_Leaf_Spot",
            "Phoma_Leaf_Spot",
            "Leaf_Miner_Damage",
        }

    def test_corn_and_coffee_do_not_share_a_weights_file(self):
        assert get_spec("leaf-foliar").weights_path != get_spec("coffee-foliar").weights_path

    def test_unknown_key_lists_the_alternatives(self):
        with pytest.raises(KeyError, match="Disponíveis"):
            get_spec("modelo-que-nao-existe")


class TestPlatformRouting:
    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            (Platform.GROUND, "rover-understory"),
            (Platform.AERIAL, "aerial-canopy"),
            (Platform.ANY, "leaf-foliar"),
        ],
    )
    def test_maps_platform_to_its_model(self, platform, expected):
        assert spec_for_platform(platform).key == expected

    def test_registry_defaults_to_the_configured_model(self):
        assert ModelRegistry().resolve().key == settings.default_model

    def test_registry_prefers_the_explicit_key(self):
        resolved = ModelRegistry().resolve("aerial-canopy", Platform.GROUND)
        assert resolved.key == "aerial-canopy"


class TestClassNameNormalisation:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("Corn_(maize)___Common_rust_", "Common_Rust"),
            ("Corn_(maize)___Northern_Leaf_Blight", "Blight"),
            ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Gray_Leaf_Spot"),
            ("Corn_(maize)___healthy", "Healthy"),
            ("Corn rust leaf", "Common_Rust"),
            ("Gray_Leaf_Spot", "Gray_Leaf_Spot"),
        ],
    )
    def test_maps_public_dataset_folders_to_canonical_classes(self, source, expected):
        assert SOURCE_CLASS_ALIASES[normalize_class_name(source)] == expected

    def test_normalisation_is_stable_across_punctuation(self):
        assert normalize_class_name("Corn_(maize)___Common_rust_") == "corn maize common rust"

    def test_all_aliases_point_to_a_real_class(self):
        every_class = {name for spec in MODEL_SPECS.values() for name in spec.class_names}
        assert set(SOURCE_CLASS_ALIASES.values()) <= every_class

    def test_bare_rust_folder_resolves_to_coffee_not_corn(self, tmp_path):
        """ "Rust" isolado é a pasta do JMuBEN; corn usa sempre "Common_Rust"."""
        assert SOURCE_CLASS_ALIASES["rust"] == "Rust"
        assert "Rust" not in get_spec("leaf-foliar").class_names

    def test_bare_rust_folder_is_unmapped_for_corn(self, tmp_path):
        """Mesmo com o alias global, o filtro por modelo evita a classe errada."""
        self._touch_images(tmp_path / "Rust", count=2)
        plan, unmapped = plan_dataset([tmp_path], get_spec("leaf-foliar"))
        assert plan == {}
        assert [d.name for d in unmapped] == ["Rust"]

    @staticmethod
    def _touch_images(directory, count):
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.jpg").write_bytes(make_image_bytes(size=(8, 8)))


class TestDatasetPreparation:
    @staticmethod
    def _make_dataset(root, folders):
        for folder, count in folders.items():
            directory = root / folder
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / f"{index}.jpg").write_bytes(make_image_bytes(size=(8, 8)))

    def test_finds_directories_that_hold_images(self, tmp_path):
        self._make_dataset(tmp_path, {"train/Blight": 2, "train/Healthy": 1})
        assert {d.name for d in find_class_dirs(tmp_path)} == {"Blight", "Healthy"}

    def test_consolidates_two_datasets_into_canonical_classes(self, tmp_path):
        plantvillage = tmp_path / "PlantVillage"
        kaggle = tmp_path / "kaggle"
        self._make_dataset(plantvillage, {"Corn_(maize)___Common_rust_": 2})
        self._make_dataset(kaggle, {"Common_Rust": 3, "Healthy": 1})

        plan, unmapped = plan_dataset([plantvillage, kaggle], get_spec("leaf-foliar"))

        assert len(plan["Common_Rust"]) == 2, "as duas origens caem na mesma classe"
        assert len(plan["Healthy"]) == 1
        assert unmapped == []

    def test_reports_unmapped_folders_instead_of_guessing(self, tmp_path):
        """Um palpite errado aqui contamina o dataset em silêncio."""
        self._make_dataset(tmp_path, {"Tomato___Early_blight": 2, "Healthy": 1})
        plan, unmapped = plan_dataset([tmp_path], get_spec("leaf-foliar"))

        assert list(plan) == ["Healthy"]
        assert [d.name for d in unmapped] == ["Tomato___Early_blight"]

    def test_ignores_classes_outside_the_target_model(self, tmp_path):
        """O modelo aéreo é binário: as classes foliares não pertencem a ele."""
        self._make_dataset(tmp_path, {"Common_Rust": 2})
        plan, unmapped = plan_dataset([tmp_path], get_spec("aerial-canopy"))

        assert plan == {}
        assert len(unmapped) == 1

    def test_copies_images_without_name_collisions(self, tmp_path):
        origem_a = tmp_path / "a"
        origem_b = tmp_path / "b"
        self._make_dataset(origem_a, {"Common_Rust": 2})
        self._make_dataset(origem_b, {"Corn rust leaf": 2})

        plan, _ = plan_dataset([origem_a, origem_b], get_spec("leaf-foliar"))
        counts = execute_plan(plan, tmp_path / "saida")

        # 0.jpg e 1.jpg existem nas duas origens; o prefixo evita a sobrescrita.
        assert counts["Common_Rust"] == 4
        assert len(list((tmp_path / "saida" / "Common_Rust").iterdir())) == 4

    def test_leaves_the_source_intact_when_copying(self, tmp_path):
        self._make_dataset(tmp_path / "origem", {"Healthy": 2})
        plan, _ = plan_dataset([tmp_path / "origem"], get_spec("leaf-foliar"))
        execute_plan(plan, tmp_path / "saida")

        assert len(list((tmp_path / "origem" / "Healthy").iterdir())) == 2
