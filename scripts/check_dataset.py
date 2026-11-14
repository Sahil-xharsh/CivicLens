import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civiclens.data import (  # noqa: E402
    DatasetError,
    load_dataset,
    load_provenance,
    validate_dataset_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complaints",
        type=Path,
        default=ROOT / "data" / "interim" / "complaints.csv",
        help="complaint CSV (default: data/interim/complaints.csv)",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT / "data" / "interim" / "provenance.csv",
        help="provenance CSV (default: data/interim/provenance.csv)",
    )
    args = parser.parse_args()

    try:
        rows = load_dataset(args.complaints)
        provenance = load_provenance(args.provenance)
        report = validate_dataset_readiness(rows, provenance)
    except (DatasetError, OSError) as error:
        print(f"Dataset check failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["cross_split_variant_candidates"]:
        print(
            "Review the possible cross-split text variants listed above before freezing splits.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
