import csv
from collections import Counter
from pathlib import Path

from civiclens.taxonomy import (
    CATEGORIES,
    CSV_FIELDS,
    LABEL_SOURCES,
    LANGUAGES,
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
