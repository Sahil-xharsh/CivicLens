# Dataset preparation contract

## Provenance CSV

The companion file uses these columns:

| Column | Meaning |
| --- | --- |
| `complaint_id` | Stable ID matching exactly one complaint row. |
| `source_type` | `synthetic`, `authored`, or `public`. |
| `source_name` | Dataset, publisher, or authored batch name. |
| `source_reference` | Relative file reference for synthetic examples, authoring batch reference for authored examples, or URL for public examples. |
| `license` | License or reuse terms; use `not_applicable` for synthetic examples and `author-created` for authored examples. |
| `accessed_on` | Access date for public sources in ISO `YYYY-MM-DD` format; leave blank otherwise. |

Each complaint must have exactly one provenance row. Public records must include a license and access date. Keep restricted or personally identifying source material out of the repository and remove direct personal identifiers from complaint text before model use.

## Reviewed dataset

The processed dataset has 300 rows: 170 human-labeled train/development examples, the 30-row synthetic pilot, and 100 human-labeled test examples. The synthetic pilot is for workflow checks and must not support performance claims. The test split has 24 emergency examples and is stored separately from train/development data.

`data/processed/test-manifest.json` records the frozen test checksum. Verify the manifest before evaluation and do not tune model choices or thresholds using test results. The annotation draft files preserve reviewed label decisions and authored source text.

## Split and readiness checks

`validate_dataset_readiness` checks that provenance IDs match complaint IDs, that normalized exact duplicates do not cross splits, and reports possible cross-split text variants for manual review. Similarity candidates are review prompts, not automatic duplicate decisions. Human review is still needed for paraphrases, translations and transliterations that automated text matching cannot reliably identify.

Use the readiness check from the project root:

```powershell
python scripts/check_dataset.py
```

Use `--train-dev`, `--test`, and `--provenance` to check alternate files. The report includes split and label counts, emergency counts, target gaps, possible text variants, and frozen-test checksum verification.

## Baseline

The TF-IDF baseline is in `src/civiclens/baseline.py`. Run `python scripts/train_baseline.py` for development metrics and development-based emergency threshold selection. Run `python scripts/evaluate_baseline.py` to refit on train, verify the test checksum, and report metrics on the frozen test set. The test report is written to `reports/baseline-test-report.json`; test metrics are not used for further tuning.
