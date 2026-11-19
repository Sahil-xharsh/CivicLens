# CivicLens

CivicLens is a portfolio research project for classifying civic complaints and identifying.
## Current status

- The label guide and CSV validator are in place.
- The working dataset has 71 train/development rows: 30 synthetic pilot examples and 41 reviewed authored examples.
- A further 129 train/development candidates and 100 test candidates are prepared separately for label review. The test candidate file is pending and is not a frozen evaluation set.
- A TF-IDF and Logistic Regression baseline and development-only evaluation command are implemented. No test results or performance claims are available yet.
- The project is a research demo, not a live civic service.

## Setup

Create an isolated Python environment and install the baseline dependency:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Dataset checks

From the project root, run:

```powershell
.venv\Scripts\python scripts/check_dataset.py
```

The report shows dataset size, split and label counts, target gaps, and possible cross-split text variants. Pending annotation files are not included in the model-ready complaint table.

## Baseline development report

```powershell
.venv\Scripts\python scripts/train_baseline.py
```

The baseline fits only rows marked `train` and reports metrics on `dev`. The report is for development only; use a human-reviewed, frozen test set for final evaluation.
