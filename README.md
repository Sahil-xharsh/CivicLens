# CivicLens

CivicLens is a portfolio research project for classifying civic complaints and identifying cases that may need human review.

## Current status

- The label guide, dataset validator, and provenance records are in place.
- The dataset has 300 rows: 170 human-labeled train/development examples, the 30-row synthetic pilot, and 100 human-labeled frozen test examples.
- The TF-IDF and Logistic Regression baseline is implemented. On the frozen test set, category macro-F1 was 0.763, urgency macro-F1 was 0.534, and emergency recall was 6/24 (0.250). The report and limits are documented in `reports/baseline-results.md`.
- The project is a research demo, not a live civic service.

## Setup

Create an isolated Python environment and install the baseline dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Dataset check

```powershell
.venv\Scripts\python scripts/check_dataset.py
```

The report checks provenance, split integrity, counts, and possible cross-split text variants. It also verifies the checksum for the frozen test file.

## Baseline reports

The development report fits on train rows and reports dev metrics:

```powershell
.venv\Scripts\python scripts/train_baseline.py
```

The frozen test report selects its emergency threshold from dev, verifies the test checksum, fits on train, and evaluates once on test:

```powershell
.venv\Scripts\python scripts/evaluate_baseline.py
```

The test report is written to `reports/baseline-test-report.json`. Results describe this authored dataset and do not establish real-world service performance.
