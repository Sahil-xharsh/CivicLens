# Data handling

`interim/pilot.csv` is a small authored synthetic pilot for checking the CSV contract and labels. It is not a held-out evaluation set and must not support performance claims.

`interim/complaints.csv` combines the pilot with the first human-reviewed authored training/development batch. The reviewed labels use `human` as the label source, and the table contains no test rows. It is preparation data, not an evaluation set.

`interim/provenance.csv` records the source of each complaint row. Keep it aligned with complaint IDs as examples are added the expected columns and validation rules are described in `docs/dataset.md`.
