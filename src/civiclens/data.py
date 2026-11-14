import csv
import unicodedata
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

from civiclens.taxonomy import (
    CATEGORIES,
    CSV_FIELDS,
    LABEL_SOURCES,
    LANGUAGES,
    PROVENANCE_FIELDS,
    SOURCE_TYPES,
    SPLITS,
    URGENCY_LEVELS,
)


class DatasetError(ValueError):
    """Raised when a complaint CSV does not match the expected format."""


def _parse_bool(value: str, row_number: int) -> bool:
    value = value.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise DatasetError(
        f"Row {row_number}: invalid is_emergency value {value!r}; use true or false."
    )


def _validate_choice(value: str, field: str, allowed: tuple[str, ...], row_number: int) -> None:
    if value not in allowed:
        choices = ", ".join(allowed)
        raise DatasetError(f"Row {row_number}: invalid {field} {value!r}. Expected: {choices}.")


def load_dataset(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise DatasetError("The dataset has no header row.")

        missing = set(CSV_FIELDS) - set(reader.fieldnames)
        if missing:
            names = ", ".join(sorted(missing))
            raise DatasetError(f"Missing required columns: {names}.")

        rows = []
        seen_ids = set()

        for row_number, raw in enumerate(reader, start=2):
            row = {field: (raw.get(field) or "").strip() for field in CSV_FIELDS}

            complaint_id = row["complaint_id"]
            if not complaint_id or not row["text"]:
                raise DatasetError(
                    f"Row {row_number}: complaint_id and text cannot be empty."
                )
            if complaint_id in seen_ids:
                raise DatasetError(f"Row {row_number}: duplicate complaint_id {complaint_id!r}.")
            seen_ids.add(complaint_id)

            for field, allowed in (
                ("language", LANGUAGES),
                ("category", CATEGORIES),
                ("urgency", URGENCY_LEVELS),
                ("label_source", LABEL_SOURCES),
                ("split", SPLITS),
            ):
                _validate_choice(row[field], field, allowed, row_number)

            row["is_emergency"] = _parse_bool(row["is_emergency"], row_number)

            if row["split"] == "test" and row["label_source"] != "human":
                raise DatasetError(
                    f"Row {row_number}: test records must have human labels."
                )

            rows.append(row)

    if not rows:
        raise DatasetError("The dataset contains no examples.")
    return rows


def split_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["split"] for row in rows))


def load_provenance(path: str | Path) -> dict[str, dict]:
    """Load one source-provenance record per complaint ID."""
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise DatasetError("The provenance file has no header row.")

        missing = set(PROVENANCE_FIELDS) - set(reader.fieldnames)
        if missing:
            names = ", ".join(sorted(missing))
            raise DatasetError(f"Missing provenance columns: {names}.")

        records = {}
        for row_number, raw in enumerate(reader, start=2):
            row = {field: (raw.get(field) or "").strip() for field in PROVENANCE_FIELDS}
            complaint_id = row["complaint_id"]
            if not complaint_id:
                raise DatasetError(f"Provenance row {row_number}: complaint_id cannot be empty.")
            if complaint_id in records:
                raise DatasetError(
                    f"Provenance row {row_number}: duplicate complaint_id {complaint_id!r}."
                )
            _validate_choice(row["source_type"], "source_type", SOURCE_TYPES, row_number)
            if not row["source_name"] or not row["source_reference"]:
                raise DatasetError(
                    f"Provenance row {row_number}: source_name and source_reference cannot be empty."
                )
            if row["source_type"] == "public" and not all(
                row[field] for field in ("license", "accessed_on")
            ):
                raise DatasetError(
                    f"Provenance row {row_number}: public sources require license and accessed_on."
                )
            if row["source_type"] == "public":
                parsed_reference = urlparse(row["source_reference"])
                if parsed_reference.scheme not in ("http", "https") or not parsed_reference.netloc:
                    raise DatasetError(
                        f"Provenance row {row_number}: public source_reference must be an HTTP(S) URL."
                    )
                try:
                    accessed_on = date.fromisoformat(row["accessed_on"])
                except ValueError as error:
                    raise DatasetError(
                        f"Provenance row {row_number}: accessed_on must use YYYY-MM-DD."
                    ) from error
                if accessed_on.isoformat() != row["accessed_on"]:
                    raise DatasetError(
                        f"Provenance row {row_number}: accessed_on must use YYYY-MM-DD."
                    )
            records[complaint_id] = row

    if not records:
        raise DatasetError("The provenance file contains no records.")
    return records


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in normalized).split())


def validate_dataset_readiness(
    rows: list[dict],
    provenance: dict[str, dict],
    *,
    min_examples: int = 300,
    min_test_examples: int = 100,
    min_emergency_test_examples: int = 20,
) -> dict:
    """Check provenance and split integrity, then report progress toward dataset targets."""
    nonhuman_test_ids = sorted(
        row["complaint_id"]
        for row in rows
        if row["split"] == "test" and row["label_source"] != "human"
    )
    if nonhuman_test_ids:
        raise DatasetError(
            "Test records must have human labels: " + ", ".join(nonhuman_test_ids) + "."
        )

    dataset_ids = {row["complaint_id"] for row in rows}
    provenance_ids = set(provenance)
    missing = sorted(dataset_ids - provenance_ids)
    orphaned = sorted(provenance_ids - dataset_ids)
    if missing or orphaned:
        details = []
        if missing:
            details.append(f"missing provenance for: {', '.join(missing)}")
        if orphaned:
            details.append(f"provenance without dataset rows: {', '.join(orphaned)}")
        raise DatasetError("; ".join(details) + ".")

    normalized_rows: dict[str, list[dict]] = {}
    for row in rows:
        normalized_rows.setdefault(_normalized_text(row["text"]), []).append(row)

    variant_candidates = []
    seen_pairs = set()
    for normalized, matches in normalized_rows.items():
        splits = {row["split"] for row in matches}
        if len(splits) > 1:
            ids = ", ".join(sorted(row["complaint_id"] for row in matches))
            raise DatasetError(f"Exact or normalized duplicate text crosses splits: {ids}.")

    unique_rows = list(rows)
    for index, left in enumerate(unique_rows):
        left_text = _normalized_text(left["text"])
        for right in unique_rows[index + 1:]:
            if left["split"] == right["split"]:
                continue
            right_text = _normalized_text(right["text"])
            pair = tuple(sorted((left["complaint_id"], right["complaint_id"])))
            if pair in seen_pairs or left_text == right_text:
                continue
            similarity = SequenceMatcher(None, left_text, right_text).ratio()
            if similarity >= 0.9:
                seen_pairs.add(pair)
                variant_candidates.append({
                    "complaint_ids": pair,
                    "similarity": round(similarity, 3),
                })

    counts = split_counts(rows)
    test_count = counts.get("test", 0)
    emergency_test_count = sum(
        row["split"] == "test" and row["is_emergency"] for row in rows
    )
    category_counts = dict(Counter(row["category"] for row in rows))
    language_counts = dict(Counter(row["language"] for row in rows))
    urgency_counts = dict(Counter(row["urgency"] for row in rows))
    label_source_counts = dict(Counter(row["label_source"] for row in rows))
    emergency_split_counts = {
        split: sum(
            row["split"] == split and row["is_emergency"] for row in rows
        )
        for split in SPLITS
    }
    target_gaps = []
    for label, actual, target in (
        ("total examples", len(rows), min_examples),
        ("test examples", test_count, min_test_examples),
        ("emergency test examples", emergency_test_count, min_emergency_test_examples),
    ):
        if actual < target:
            target_gaps.append({"target": label, "actual": actual, "needed": target - actual})

    return {
        "total_examples": len(rows),
        "split_counts": counts,
        "test_emergency_count": emergency_test_count,
        "emergency_split_counts": emergency_split_counts,
        "category_counts": category_counts,
        "language_counts": language_counts,
        "urgency_counts": urgency_counts,
        "label_source_counts": label_source_counts,
        "target_gaps": target_gaps,
        "cross_split_variant_candidates": variant_candidates,
    }
