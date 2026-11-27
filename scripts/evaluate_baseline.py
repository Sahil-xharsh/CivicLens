"""Evaluate the baseline once on the checksum-verified frozen test set."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civiclens.baseline import TfidfComplaintBaseline  # noqa: E402
from civiclens.data import DatasetError, load_dataset  # noqa: E402
from civiclens.evaluation import (  # noqa: E402
    evaluate_development,
    evaluate_split,
    select_emergency_threshold,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dev",
        type=Path,
        default=ROOT / "data" / "processed" / "train_dev.csv",
        help="train/development CSV",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "data" / "processed" / "test.csv",
        help="frozen test CSV",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "test-manifest.json",
        help="frozen test checksum manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "baseline-test-report.json",
        help="JSON report output path",
    )
    args = parser.parse_args()

    try:
        test_hash = hashlib.sha256(args.test.read_bytes()).hexdigest()
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("sha256") != test_hash:
            raise DatasetError("Frozen test checksum does not match the manifest.")

        train_dev = load_dataset(args.train_dev)
        test_rows = load_dataset(args.test)
        if any(row["split"] == "test" for row in train_dev):
            raise DatasetError("Train/dev input must not contain test rows.")
        if any(row["split"] != "test" for row in test_rows):
            raise DatasetError("Test input must contain only test rows.")

        train_rows = [row for row in train_dev if row["split"] == "train"]
        dev_rows = [row for row in train_dev if row["split"] == "dev"]
        if not train_rows or not dev_rows:
            raise DatasetError("Both train and dev rows are required.")
        if len(test_rows) != manifest.get("rows"):
            raise DatasetError("Test row count does not match the freeze manifest.")
        model = TfidfComplaintBaseline().fit(train_rows)
        threshold_selection = select_emergency_threshold(model, dev_rows)
        threshold = threshold_selection["threshold"]
        dev_report = evaluate_development(
            model,
            dev_rows,
            emergency_threshold=threshold,
        )
        test_report = evaluate_split(
            model,
            test_rows,
            split="test",
            emergency_threshold=threshold,
        )
        predictions = model.predict_many(
            [row["text"] for row in test_rows],
            emergency_threshold=threshold,
        )
    except (DatasetError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Baseline evaluation failed: {error}", file=sys.stderr)
        return 1

    errors = []
    for row, prediction in zip(test_rows, predictions):
        if (
            row["category"] != prediction.category
            or row["urgency"] != prediction.urgency
            or row["is_emergency"] != prediction.is_emergency
        ):
            errors.append(
                {
                    "complaint_id": row["complaint_id"],
                    "language": row["language"],
                    "text": row["text"],
                    "actual": {
                        "category": row["category"],
                        "urgency": row["urgency"],
                        "is_emergency": row["is_emergency"],
                    },
                    "predicted": {
                        "category": prediction.category,
                        "urgency": prediction.urgency,
                        "is_emergency": prediction.is_emergency,
                        "emergency_probability": prediction.emergency_probability,
                    },
                }
            )

    report = {
        "model": "TF-IDF with Logistic Regression",
        "training_count": len(train_rows),
        "development_count": len(dev_rows),
        "test_count": len(test_rows),
        "test_manifest_sha256": test_hash,
        "threshold_selection": threshold_selection,
        "development_metrics": dev_report,
        "test_metrics": test_report,
        "test_errors_for_review": errors,
        "limitations": [
            "The training split includes the 30-row synthetic pilot.",
            "The report measures this authored dataset and does not establish real-world service performance.",
            "TF-IDF class probabilities are not calibrated confidence estimates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote frozen-test report to {args.output}")
    print(json.dumps({"test_metrics": test_report, "error_count": len(errors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
