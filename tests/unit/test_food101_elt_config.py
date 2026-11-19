"""Regression tests for side-effect-free Food-101 ELT config validation."""

from pathlib import Path

import pytest
import yaml

from food_assistant.data_pipeline.config import load_pipeline_config

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "food101_elt.yaml"


def test_default_config_loads_without_side_effects(tmp_path):
    """Load the checked-in config and confirm it creates no files in the workspace."""
    config_path = tmp_path / "food101_elt.yaml"
    config_path.write_bytes(CONFIG_PATH.read_bytes())
    config = load_pipeline_config(config_path, tmp_path)
    assert config.download_policy == "disabled"
    assert config.dataset_scope == "poc"
    assert config.manifest_root == (tmp_path / "data/splits/v2").resolve()
    assert list(tmp_path.iterdir()) == [config_path]


def test_full_v2_config_loads_with_safe_clean_split_policy(tmp_path):
    """Load approved full-scope config without creating staging or lock files."""
    source = CONFIG_PATH.parents[0] / "food101_elt_full_v2.yaml"
    copied = tmp_path / "configs/food101_elt_full_v2.yaml"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source.read_bytes())
    document = yaml.safe_load(copied.read_text(encoding="utf-8"))
    for name, value in document["paths"].items():
        document["paths"][name] = str((tmp_path / value).resolve())
    document["approval_file"] = str((tmp_path / "approval.json").resolve())
    document["class_selection_file"] = str((tmp_path / "classes_full.yaml").resolve())
    document["fixtures"]["root"] = str((tmp_path / "fixtures").resolve())
    copied.write_text(yaml.safe_dump(document), encoding="utf-8")

    config = load_pipeline_config(copied, tmp_path)

    assert config.dataset_scope == "full"
    assert config.split_version == "v2"
    assert config.preserve_official_test is True
    assert config.exclude_train_hashes_in_test is True
    assert config.create_validation is False
    assert config.download_policy == "disabled"
    assert list(tmp_path.rglob("*.lock")) == []


def test_full_v2_config_rejects_unsafe_split_policy(tmp_path):
    """Refuse full-scope config that would alter official test or omit leakage filtering."""
    original = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    original["dataset_scope"] = "full"
    original["split_policy"] = {
        "preserve_official_test": True,
        "exclude_train_hashes_in_test": False,
        "create_validation": False,
        "reject_cross_partition_duplicates": True,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(ValueError, match="Full v2 policy must preserve"):
        load_pipeline_config(config_path, tmp_path)


def test_full_v3_config_loads_with_source_and_quarantine_policies(tmp_path):
    """Parse full v3 source, duplicate, validation and test-preservation policies."""
    source = CONFIG_PATH.parents[0] / "food101_elt_full_v3.yaml"
    config_path = tmp_path / "configs/food101_elt_full_v3.yaml"
    config_path.parent.mkdir(parents=True)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    for name, value in document["paths"].items():
        document["paths"][name] = str((tmp_path / value).resolve())
    document["source_manifest_root"] = str((tmp_path / document["source_manifest_root"]).resolve())
    document["approval_file"] = str((tmp_path / "approval.json").resolve())
    document["class_selection_file"] = str((tmp_path / "classes_full.yaml").resolve())
    document["fixtures"]["root"] = str((tmp_path / "fixtures").resolve())
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    config = load_pipeline_config(config_path, tmp_path)

    assert config.split_version == "v3"
    assert config.dataset_scope == "full"
    assert config.source_manifest_root == (tmp_path / "data/splits/v2").resolve()
    assert config.val_fraction == 0.1
    assert config.seed == 42
    assert config.preserve_official_test is True
    assert config.create_validation is True
    assert config.deduplicate_train is True
    assert config.same_label_duplicate_policy == "keep_lexicographic_min"
    assert config.cross_label_duplicate_policy == "quarantine_all"
    assert config.download_policy == "disabled"
    assert list(tmp_path.rglob("*.lock")) == []


@pytest.mark.parametrize("source_path", ["data/raw/food-101", "data/splits/v1"])
def test_full_v3_rejects_protected_or_overlapping_source_root(tmp_path, source_path):
    """Reject raw/v1 inputs before any v3 filesystem side effects."""
    source = CONFIG_PATH.parents[0] / "food101_elt_full_v3.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    for name, value in document["paths"].items():
        document["paths"][name] = str((tmp_path / value).resolve())
    document["source_manifest_root"] = str((tmp_path / source_path).resolve())
    document["approval_file"] = str((tmp_path / "approval.json").resolve())
    document["class_selection_file"] = str((tmp_path / "classes_full.yaml").resolve())
    document["fixtures"]["root"] = str((tmp_path / "fixtures").resolve())
    config_path = tmp_path / "food101_elt_full_v3.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="source_manifest_root overlaps"):
        load_pipeline_config(config_path, tmp_path)

    assert not list(tmp_path.rglob("*.lock"))


def test_full_v3_rejects_output_report_path_collision(tmp_path):
    """Reject report paths nested under the immutable v3 manifest output."""
    source = CONFIG_PATH.parents[0] / "food101_elt_full_v3.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    for name, value in document["paths"].items():
        document["paths"][name] = str((tmp_path / value).resolve())
    document["paths"]["report_root"] = str((tmp_path / "data/splits/v3/reports").resolve())
    document["source_manifest_root"] = str((tmp_path / document["source_manifest_root"]).resolve())
    document["approval_file"] = str((tmp_path / "approval.json").resolve())
    document["class_selection_file"] = str((tmp_path / "classes_full.yaml").resolve())
    document["fixtures"]["root"] = str((tmp_path / "fixtures").resolve())
    config_path = tmp_path / "food101_elt_full_v3.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="resources .* must not overlap"):
        load_pipeline_config(config_path, tmp_path)

    assert not list(tmp_path.rglob("*.lock"))


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("seed",), -1),
        (("split_version",), "v1"),
        (("paths", "manifest_root"), "data/raw/food-101/output"),
        (("paths", "report_root"), "data/splits/v1/reports"),
        (("paths", "staging_root"), "../../outside"),
    ],
)
def test_invalid_config_values_are_rejected(tmp_path, field_path, value):
    """Reject invalid scalar and protected path values before any filesystem writes."""
    original = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    section = original
    for key in field_path[:-1]:
        section = section[key]
    section[field_path[-1]] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_config(config_path, tmp_path)
    assert list(tmp_path.iterdir()) == [config_path]
