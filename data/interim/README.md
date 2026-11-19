# Data handling

`interim/pilot.csv` is a small authored synthetic pilot for checking the CSV contract and labels. It is not a held-out evaluation set and must not support performance claims.

`interim/complaints.csv` combines the pilot with the first two human-reviewed authored training/development batches. The reviewed labels use `human` as the label source, and the table contains no test rows. It is preparation data, not an evaluation set.

`interim/provenance.csv` records the source of each complaint row. Keep it aligned with complaint IDs as examples are added; the expected columns and validation rules are described in `docs/dataset.md`.

`interim/annotation-draft-002.csv` records the second authored batch and its human-confirmed labels. Draft examples are added to `complaints.csv` only after label review.

`interim/annotation-draft-003.csv` contains 21 new pending training/development candidates. Review the proposed labels before promoting them into `complaints.csv` and adding matching provenance rows.

`interim/annotation-draft-004.csv` adds 108 pending training/development candidates. `interim/annotation-test-draft.csv` holds 100 separate proposed test examples, including more than 20 proposed emergency cases. These files form a 300-row review package with the existing 71 complaint rows and batch 003; they are not a finalized dataset until reviewed candidates are promoted. Never use the pending test candidates for tuning or evaluation.

Run `python scripts/check_dataset.py` from the project root to print split counts, label distributions, target gaps, and possible cross-split text variants.
