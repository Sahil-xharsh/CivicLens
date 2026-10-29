import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civiclens.data import DatasetError, load_dataset, split_counts  # noqa: E402


FIELDS = (
    "complaint_id",
    "text",
    "language",
    "category",
    "urgency",
    "is_emergency",
    "label_source",
    "split",
)


def sample_row(**changes):
    row = {
        "complaint_id": "case-1",
        "text": "A pothole is near the bus stop.",
        "language": "en",
        "category": "roads",
        "urgency": "low",
        "is_emergency": "false",
        "label_source": "synthetic",
        "split": "train",
    }
    row.update(changes)
    return row


class DatasetTests(unittest.TestCase):
    def write_csv(self, rows, fields=FIELDS):
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
        ) as file:
            path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_pilot_dataset_matches_the_contract(self):
        rows = load_dataset(ROOT / "data" / "interim" / "pilot.csv")

        self.assertEqual(len(rows), 30)
        self.assertEqual(split_counts(rows), {"train": 30})
        self.assertTrue(all(row["label_source"] == "synthetic" for row in rows))
        self.assertTrue(all(isinstance(row["is_emergency"], bool) for row in rows))

    def test_missing_required_columns_are_rejected(self):
        path = self.write_csv(
            [sample_row()],
            fields=("complaint_id", "text"),
        )

        with self.assertRaisesRegex(DatasetError, "Missing required columns"):
            load_dataset(path)

    def test_empty_complaint_text_is_rejected(self):
        path = self.write_csv([sample_row(text="   ")])

        with self.assertRaisesRegex(DatasetError, "complaint_id and text"):
            load_dataset(path)

    def test_duplicate_complaint_ids_are_rejected(self):
        path = self.write_csv([
            sample_row(),
            sample_row(text="Another complaint"),
        ])

        with self.assertRaisesRegex(DatasetError, "duplicate complaint_id"):
            load_dataset(path)

    def test_unknown_category_is_rejected(self):
        path = self.write_csv([sample_row(category="roads-and-water")])

        with self.assertRaisesRegex(DatasetError, "category"):
            load_dataset(path)

    def test_unknown_urgency_is_rejected(self):
        path = self.write_csv([sample_row(urgency="urgent")])

        with self.assertRaisesRegex(DatasetError, "urgency"):
            load_dataset(path)

    def test_invalid_emergency_value_is_rejected(self):
        path = self.write_csv([sample_row(is_emergency="maybe")])

        with self.assertRaisesRegex(DatasetError, "is_emergency"):
            load_dataset(path)

    def test_test_split_requires_human_labels(self):
        path = self.write_csv([
            sample_row(split="test", label_source="synthetic"),
        ])

        with self.assertRaisesRegex(DatasetError, "human labels"):
            load_dataset(path)


if __name__ == "__main__":
    unittest.main()