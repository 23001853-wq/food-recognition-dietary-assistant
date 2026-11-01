"""Validate immutable reference imports; preserve unrelated and historical catalog rows."""

import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

NUTRIENTS = ("kcal_per_100g", "protein_per_100g", "carbohydrate_per_100g", "fat_per_100g")
PORTIONS = ("small_grams", "medium_grams", "large_grams")


def manifest_labels(path):
    mapping = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index, key = int(row["class_id"]), row["class_key"]
            if mapping.setdefault(index, key) != key:
                raise ValueError("Manifest has an inconsistent class mapping")
    if sorted(mapping) != list(range(len(mapping))) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Manifest class indices/keys must be unique and contiguous")
    return [mapping[i] for i in sorted(mapping)]


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value, *, positive=False):
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid reference number") from exc
    if not number.is_finite() or number < 0 or (positive and number <= 0):
        raise ValueError("Reference number must be finite and non-negative; portions positive")
    if number.as_tuple().exponent < -2:
        raise ValueError("Reference precision exceeds database NUMERIC(10,2)")
    return number


def identity(row):
    return row["source_name"], row["source_record_id"], row["source_version"]


def validate_reference(row):
    if row.get("reviewed") != "true" or not row.get("provenance_url", "").startswith("https://"):
        raise ValueError("Reference must be reviewed and have a source URL")
    for key in ("name", "source_name", "source_record_id", "source_version", "retrieved_at"):
        if not row.get(key):
            raise ValueError(f"Missing reference field: {key}")
    for key in NUTRIENTS:
        numeric(row.get(key))
    for key in PORTIONS:
        numeric(row.get(key), positive=True)


def merge_references(existing, additions):
    result = {identity(row): dict(row) for row in existing}
    for row in additions:
        validate_reference(row)
        key = identity(row)
        if key in result:
            # Never silently alter a historical profile; use a new source version for corrections.
            for field in (*NUTRIENTS, *PORTIONS):
                if numeric(result[key].get(field), positive=field in PORTIONS) != numeric(
                    row.get(field), positive=field in PORTIONS
                ):
                    raise ValueError(
                        f"Historical profile differs for {key}/{field}; create a new version"
                    )
            row = {**row, **result[key]}
        result[key] = row
    return list(result.values())


def write_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def profile_notes(profile):
    raw = getattr(profile, "mapping_note", None) or ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    # Legacy pizza profile is retained; its note already declares the demo convention.
    basis = (
        "design_assumption"
        if "demo convention" in raw or "quy ước thiết kế" in raw
        else "unspecified"
    )
    return {
        "match_note": (
            "Tham chiếu pizza phô mai đế thường đông lạnh đã nướng; "
            "các loại đế, nhân và công thức khác có thể thay đổi dinh dưỡng."
            if "demo convention" in raw
            and str(getattr(profile, "source_record_id", "")) == "170317"
            else raw
        ),
        "portion_basis": basis,
        "portion_note": (
            "Khối lượng S/M/L là quy ước thiết kế, không phải khẩu phần USDA "
            "hoặc đo từ ảnh; có thể nhập gram thực tế."
        )
        if basis == "design_assumption"
        else "Khối lượng tham khảo không được đo từ ảnh; có thể nhập gram thực tế.",
    }
