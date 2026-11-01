"""Extract reviewed USDA records; nutrient values are read from archives, never authored here."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from food_assistant.config import PROJECT_ROOT
from food_assistant.nutrition.catalog import (
    identity,
    manifest_labels,
    merge_references,
    numeric,
    read_csv,
    write_csv,
)

SOURCES = {
    "sr_legacy": {
        "file": "usda_sr_legacy_2018.zip",
        "name": "USDA FoodData Central SR Legacy",
        "version": "SR Legacy 2018-04 (FDC 2019 publication)",
        "url": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip",
    },
    "fndds": {
        "file": "usda_fndds_2021_2023.zip",
        "name": "USDA FoodData Central FNDDS",
        "version": "FNDDS 2021-2023 (2024-10-31 release)",
        "url": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_json_2024-10-31.zip",
    },
}
IDS = {
    "1008": ("kcal_per_100g", "kcal"),
    "1003": ("protein_per_100g", "g"),
    "1005": ("carbohydrate_per_100g", "g"),
    "1004": ("fat_per_100g", "g"),
}


def sr_records(archive, selected):
    with zipfile.ZipFile(archive) as z:

        def rows(filename):
            member = next(n for n in z.namelist() if n.endswith("/" + filename))
            return csv.DictReader(io.TextIOWrapper(z.open(member), encoding="utf-8-sig"))

        units = {r["id"]: r["unit_name"].lower() for r in rows("nutrient.csv") if r["id"] in IDS}
        foods = {
            r["fdc_id"]: {"description": r["description"], "nutrients": {}}
            for r in rows("food.csv")
            if r["fdc_id"] in selected
        }
        for row in rows("food_nutrient.csv"):
            if row["fdc_id"] in foods and row["nutrient_id"] in IDS:
                field, expected_unit = IDS[row["nutrient_id"]]
                if units[row["nutrient_id"]] != expected_unit:
                    raise ValueError("USDA SR nutrient unit mismatch")
                foods[row["fdc_id"]]["nutrients"][field] = row["amount"]
        return foods


def fndds_records(archive, selected):
    with zipfile.ZipFile(archive) as z:
        data = json.loads(z.read(next(n for n in z.namelist() if n.endswith(".json"))))
    foods = {}
    for record in data["SurveyFoods"]:
        key = str(record["fdcId"])
        if key not in selected:
            continue
        amounts = {}
        for nutrient in record["foodNutrients"]:
            code = str(nutrient["nutrient"]["id"])
            if code in IDS and nutrient.get("amount") is not None:
                field, expected_unit = IDS[code]
                if nutrient["nutrient"]["unitName"].lower() != expected_unit:
                    raise ValueError("USDA FNDDS nutrient unit mismatch")
                amounts[field] = str(nutrient["amount"])
        foods[key] = {"description": record["description"], "nutrients": amounts}
    return foods


def prepare(manifest, curation_path, raw_dir, output):
    output = Path(output)
    config = json.loads(Path(curation_path).read_text(encoding="utf-8"))
    keys = manifest_labels(manifest)
    labels = json.loads(Path(manifest).with_name("labels.json").read_text(encoding="utf-8"))
    if [row["class_key"] for row in labels["labels"]] != keys:
        raise ValueError("Manifest and labels.json disagree")
    records, source_evidence = {}, {}
    for dataset, source in SOURCES.items():
        archive = Path(raw_dir) / source["file"]
        selected = {
            row["fdc_id"]
            for key, row in config["foods"].items()
            if key in keys and row["dataset"] == dataset
        }
        if not selected:
            continue
        if not archive.exists():
            raise FileNotFoundError(
                f"Required USDA archive missing: {archive}. No data fabricated."
            )
        records[dataset] = (
            sr_records(archive, selected)
            if dataset == "sr_legacy"
            else fndds_records(archive, selected)
        )
        source_evidence[dataset] = {
            "download_url": source["url"],
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "checksum_kind": "computed local integrity, not an official published checksum",
        }
    additions, mappings, evidence, unavailable = [], [], [], []
    retrieved = datetime.now(UTC).isoformat()
    for key in keys:
        choice = config["foods"].get(key)
        if not choice:
            unavailable.append({"class_key": key, "reason": "no_reviewed_mapping"})
            continue
        food = records[choice["dataset"]].get(choice["fdc_id"])
        if not food or not food["nutrients"].get("kcal_per_100g"):
            unavailable.append({"class_key": key, "reason": "source_record_or_energy_missing"})
            continue
        if food["description"] != choice["expected_description"]:
            raise ValueError(f"Source description changed for {key}; curator review required")
        source = SOURCES[choice["dataset"]]
        notes = {
            "curation_version": config["curation_version"],
            "match_note": choice["match_note"],
            "match_quality": choice.get("match_quality", "representative"),
            "portion_basis": config["portion_basis"],
            "portion_note": config["portion_note"],
        }
        row = {
            "source_name": source["name"],
            "source_record_id": choice["fdc_id"],
            "name": food["description"],
            **food["nutrients"],
            "source_version": source["version"],
            "provenance_url": f"https://fdc.nal.usda.gov/food-details/{choice['fdc_id']}/nutrients",
            "retrieved_at": retrieved,
            **dict(zip(["small_grams", "medium_grams", "large_grams"], map(str, choice["grams"]))),
            "mapping_note": json.dumps(notes, ensure_ascii=False),
            "reviewed": "true",
            "display_name_vi": config["display_names_vi"].get(key, key),
        }
        for field in food["nutrients"]:
            numeric(row[field])
        additions.append(row)
        mappings.append(
            {
                "dataset_key": key,
                "source_name": source["name"],
                "source_record_id": choice["fdc_id"],
                "source_version": source["version"],
                "is_default": "true",
                "confidence_note": choice["match_note"],
            }
        )
        evidence.append(
            {
                "class_key": key,
                "fdc_id": choice["fdc_id"],
                "description": food["description"],
                "dataset": choice["dataset"],
                "per_100g": food["nutrients"],
                "notes": notes,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    catalog = merge_references(read_csv(output / "source_catalog.csv"), additions)
    merged = {}
    for row in read_csv(output / "curated_mapping.csv"):
        if not row.get("source_version"):
            versions = {
                r["source_version"]
                for r in catalog
                if r["source_name"] == row["source_name"]
                and r["source_record_id"] == row["source_record_id"]
            }
            if len(versions) != 1:
                raise ValueError("Ambiguous historical mapping version")
            row["source_version"] = versions.pop()
        merged[(row["dataset_key"], *identity(row))] = row
    for row in mappings:
        merged.setdefault((row["dataset_key"], *identity(row)), row)
    write_csv(output / "source_catalog.csv", catalog)
    write_csv(output / "curated_mapping.csv", list(merged.values()))
    for label in labels["labels"]:
        label["display_name_vi"] = config["display_names_vi"].get(
            label["class_key"], label["display_name_vi"]
        )
    report = {
        "curation_version": config["curation_version"],
        "class_keys_from_manifest": keys,
        "available": [r["class_key"] for r in evidence],
        "unavailable": unavailable,
        "portion_basis": config["portion_basis"],
        "sources": source_evidence,
        "records": evidence,
    }
    (output / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "label_catalog.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data/splits/v1/dataset_manifest.csv"
    )
    parser.add_argument(
        "--curation", type=Path, default=PROJECT_ROOT / "configs/nutrition_poc.json"
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/nutrition")
    args = parser.parse_args()
    report = prepare(args.manifest, args.curation, args.raw_dir, args.output)
    print(
        json.dumps(
            {"available": report["available"], "unavailable": report["unavailable"]}, indent=2
        )
    )
