# Data handling

`interim/pilot.csv` is a small authored synthetic pilot for checking the CSV contract and labels. It is not a held-out evaluation set and must not support performance claims.

`interim/complaints.csv` combines the pilot with the first assistant-reviewed authored training/development batch. Its generated labels use `synthetic` as the label source, and it contains no test rows. It is preparation data, not an evaluation set.

`interim/provenance.csv` records the source of each complaint row. Keep it aligned with complaint IDs as examples are added the expected columns and validation rules are described in `docs/dataset.md`.
