import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from food_assistant.config import PROJECT_ROOT
from food_assistant.domain.errors import AppError
from food_assistant.nutrition.catalog import (
    NUTRIENTS,
    PORTIONS,
    identity,
    manifest_labels,
    merge_references,
    read_csv,
    validate_reference,
)
from food_assistant.nutrition.prepare import prepare
from food_assistant.services.nutrition_service import NutritionService


def profiles():
    rows = read_csv(PROJECT_ROOT / "data/nutrition/source_catalog.csv")
    return {
        r["source_record_id"]: SimpleNamespace(
            **{
                **r,
                "reviewed": True,
                **{k: Decimal(r[k]) if r.get(k) else None for k in (*NUTRIENTS, *PORTIONS)},
            }
        )
        for r in rows
    }


def test_exact_manifest_mapping_and_validated_provenance():
    root = PROJECT_ROOT / "data/nutrition"
    labels = manifest_labels(PROJECT_ROOT / "data/splits/v1/dataset_manifest.csv")
    mappings = read_csv(root / "curated_mapping.csv")
    rows = read_csv(root / "source_catalog.csv")
    assert {r["dataset_key"] for r in mappings} == set(labels)
    assert len(labels) == len(mappings) == 10
    references = {identity(r): r for r in rows}
    for row in mappings:
        validate_reference(references[identity(row)])
    assert json.loads((root / "coverage.json").read_text(encoding="utf-8"))["unavailable"] == []


@pytest.mark.parametrize(
    "record,medium", [("170694", "394.50"), ("167575", "207.00"), ("170317", "536.00")]
)
def test_real_usda_portions_custom_and_manual(record, medium):
    profile = profiles()[record]
    engine = NutritionService(None)
    default = engine.estimate(profile, "medium")
    assert default.estimated_kcal == Decimal(medium)
    assert default.mass_source == "design_assumption"
    assert engine.estimate(profile, "small").estimated_kcal < default.estimated_kcal
    assert engine.estimate(profile, "large").estimated_kcal > default.estimated_kcal
    custom = engine.estimate(profile, "custom", Decimal("123.45"))
    assert custom.estimated_kcal == (profile.kcal_per_100g * Decimal("1.2345")).quantize(
        Decimal("0.01")
    )
    assert custom.mass_source == "user_input"
    manual = engine.estimate(profile, "medium", manual_kcal=Decimal("42"))
    assert manual.source == manual.source_type == "user_input"
    assert manual.protein_g is None and manual.kcal_per_100g is None
    assert profile.kcal_per_100g == default.kcal_per_100g


def test_missing_data_and_unreviewed_reference():
    engine = NutritionService(None)
    profile = profiles()["170694"]
    assert engine.estimate(profile, "custom").unavailable_reason == "missing_grams"
    profile.kcal_per_100g = None
    result = engine.estimate(profile, "medium")
    assert result.estimated_kcal is None and result.status == "unavailable"
    profile.reviewed = False
    assert engine.estimate(profile, "medium").source_type == "unavailable"
    with pytest.raises(AppError):
        engine.estimate(profile, "bad_portion")
    for invalid in (0, -1, "NaN", "Infinity", "10001", "1.001"):
        with pytest.raises(AppError):
            engine.estimate(None, "custom", invalid)


def test_immutable_catalog_merge():
    row = read_csv(PROJECT_ROOT / "data/nutrition/source_catalog.csv")[0]
    assert merge_references([row], [dict(row)])[0] == row
    with pytest.raises(ValueError, match="Historical profile differs"):
        merge_references([row], [{**row, "kcal_per_100g": "999"}])


def test_rebuild_from_official_archives(tmp_path):
    raw = PROJECT_ROOT / "data/raw"
    if not all(
        (raw / name).exists() for name in ("usda_sr_legacy_2018.zip", "usda_fndds_2021_2023.zip")
    ):
        pytest.skip("Local official USDA archives not available; no download")
    report = prepare(
        PROJECT_ROOT / "data/splits/v1/dataset_manifest.csv",
        PROJECT_ROOT / "configs/nutrition_poc.json",
        raw,
        tmp_path,
    )
    assert len(report["available"]) == 10 and not report["unavailable"]
    actual = {identity(row): row for row in read_csv(tmp_path / "source_catalog.csv")}
    for row in read_csv(PROJECT_ROOT / "data/nutrition/source_catalog.csv"):
        for field in NUTRIENTS:
            assert Decimal(row[field]) == Decimal(actual[identity(row)][field])
