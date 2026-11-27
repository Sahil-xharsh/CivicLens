# Interim data

`pilot.csv` contains 30 synthetic train examples for checking the CSV contract. They are not held-out evaluation data.

`complaints.csv` contains the 200 train/development rows. `provenance.csv` contains their matching source records.

The annotation draft files preserve the reviewed authored records. Confirmed train/development rows were promoted to the working table, and confirmed test rows were copied to `data/processed/test.csv`.

Run `python scripts/check_dataset.py` from the project root to check the processed splits and frozen-test checksum.
