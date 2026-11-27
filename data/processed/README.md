# Processed dataset

`train_dev.csv` contains 200 rows across the train and development splits. `test.csv` contains 100 human-reviewed rows and is frozen. The companion `provenance.csv` contains source records for all 300 rows.

`test-manifest.json` records the test row count, emergency count, freeze date, and SHA-256 checksum. A change to `test.csv` requires a new review and a new freeze before evaluation. Do not tune model choices or thresholds against test results.

Use `python scripts/check_dataset.py` to verify provenance, splits, counts, cross-split text variants, and the frozen-test checksum.

The baseline evaluation is recorded in `reports/baseline-results.md` and `reports/baseline-test-report.json`. The frozen test set has been evaluated; future model and threshold choices must use train/development data.
