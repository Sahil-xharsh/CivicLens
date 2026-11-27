import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civiclens.baseline import TfidfComplaintBaseline  # noqa: E402
from civiclens.data import DatasetError, load_dataset  # noqa: E402
from civiclens.evaluation import (  # noqa: E402
    evaluate_development,
    select_emergency_threshold,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complaints",
        type=Path,
        default=ROOT / "data" / "processed" / "train_dev.csv",
        help="train/development CSV (default: data/processed/train_dev.csv)",
    )
    parser.add_argument(
        "--emergency-threshold",
        type=float,
        default=None,
        help="override the emergency threshold selected from dev data",
    )
    args = parser.parse_args()

    try:
        rows = load_dataset(args.complaints)
        train_rows = [row for row in rows if row["split"] == "train"]
        dev_rows = [row for row in rows if row["split"] == "dev"]
        if not dev_rows:
            raise DatasetError("No dev rows are available for threshold development.")
        model = TfidfComplaintBaseline().fit(train_rows)
        threshold_selection = select_emergency_threshold(model, dev_rows)
        threshold = (
            args.emergency_threshold
            if args.emergency_threshold is not None
            else threshold_selection["threshold"]
        )
        report = evaluate_development(
            model,
            dev_rows,
            emergency_threshold=threshold,
        )
    except (DatasetError, OSError, ValueError, RuntimeError) as error:
        print(f"Baseline run failed: {error}", file=sys.stderr)
        return 1

    report["training_count"] = len(train_rows)
    report["threshold_selection"] = threshold_selection
    report["training_label_sources"] = {
        source: sum(row["label_source"] == source for row in train_rows)
        for source in sorted({row["label_source"] for row in train_rows})
    }
    report["limitations"] = [
        "This report uses the development split only; it is not a held-out test result.",
        "The training data includes the 30-row synthetic pilot.",
        "TF-IDF class probabilities are not calibrated confidence estimates.",
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
