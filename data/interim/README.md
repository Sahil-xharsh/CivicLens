# Data handling

`interim/pilot.csv` is a small authored synthetic pilot for checking the CSV contract and labels. It is not a held-out evaluation set and must not support performance claims.

`interim/complaints.csv` combines the pilot with the first two human-reviewed authored training/development batches. The reviewed labels use `human` as the label source, and the table contains no test rows. It is preparation data, not an evaluation set.

`interim/provenance.csv` records the source of each complaint row. Keep it aligned with complaint IDs as examples are added; the expected columns and validation rules are described in `docs/dataset.md`.

`interim/annotation-draft-002.csv` records the second authored batch and its human-confirmed labels. Draft examples are added to `complaints.csv` only after label review.

Run `python scripts/check_dataset.py` from the project root to print split counts, label distributions, target gaps, and possible cross-split text variants.
