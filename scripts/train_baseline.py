import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civiclens.baseline import TfidfComplaintBaseline  # noqa: E402
from civiclens.data import DatasetError, load_dataset  # noqa: E402
from civiclens.evaluation import evaluate_development  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complaints",
        type=Path,
        default=ROOT / "data" / "interim" / "complaints.csv",
        help="complaint CSV (default: data/interim/complaints.csv)",
    )
    parser.add_argument(
        "--emergency-threshold",
        type=float,
        default=0.5,
        help="emergency-risk probability threshold (default: 0.5)",
    )
    args = parser.parse_args()

    try:
        rows = load_dataset(args.complaints)
        train_rows = [row for row in rows if row["split"] == "train"]
        dev_rows = [row for row in rows if row["split"] == "dev"]
        if not dev_rows:
            raise DatasetError("No dev rows are available for threshold development.")
        model = TfidfComplaintBaseline().fit(train_rows)
        report = evaluate_development(
            model,
            dev_rows,
            emergency_threshold=args.emergency_threshold,
        )
    except (DatasetError, OSError, ValueError, RuntimeError) as error:
        print(f"Baseline run failed: {error}", file=sys.stderr)
        return 1

    report["training_count"] = len(train_rows)
    report["training_label_sources"] = {
        source: sum(row["label_source"] == source for row in train_rows)
        for source in sorted({row["label_source"] for row in train_rows})
    }
    report["limitations"] = [
        "This report uses the development split only; it is not a held-out test result.",
        "The current dataset includes a synthetic pilot and has not reached the planned size.",
        "TF-IDF class probabilities are not calibrated confidence estimates.",
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
