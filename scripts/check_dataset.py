import argparse
import hashlib
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
        "--train-dev",
        type=Path,
        default=ROOT / "data" / "processed" / "train_dev.csv",
        help="train/development CSV (default: data/processed/train_dev.csv)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "data" / "processed" / "test.csv",
        help="frozen test CSV (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT / "data" / "processed" / "provenance.csv",
        help="provenance CSV (default: data/processed/provenance.csv)",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "test-manifest.json",
        help="frozen test checksum manifest",
    )
    args = parser.parse_args()

    try:
        train_dev_rows = load_dataset(args.train_dev)
        test_rows = load_dataset(args.test)
        if any(row["split"] == "test" for row in train_dev_rows):
            raise DatasetError("The train/dev file must not contain test rows.")
        if any(row["split"] != "test" for row in test_rows):
            raise DatasetError("The test file must contain only test rows.")
        test_hash = hashlib.sha256(args.test.read_bytes()).hexdigest()
        manifest = json.loads(args.test_manifest.read_text(encoding="utf-8"))
        if manifest.get("sha256") != test_hash:
            raise DatasetError("Frozen test checksum does not match the manifest.")
        rows = train_dev_rows + test_rows
        provenance = load_provenance(args.provenance)
        report = validate_dataset_readiness(rows, provenance)
        report["test_frozen"] = True
        report["test_sha256"] = test_hash
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
