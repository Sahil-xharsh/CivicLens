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

## Annotation drafts

`data/interim/annotation-draft-*.csv` records authored examples with proposed labels and review status. Draft rows use `draft_id` and `proposed_*` fields, so they do not follow the complaint CSV contract. Unreviewed generated labels use `label_source=synthetic`; labels confirmed by a human use `label_source=human`. Only human-labeled rows may enter the test set.

The current working table is `data/interim/complaints.csv`; it contains 71 rows: the 30-row synthetic pilot and 41 human-reviewed authored training/development examples. It contains no test data and is not suitable for performance evaluation.

The remaining 229 proposed rows are stored separately: 21 in batch 003, 108 in batch 004, and 100 in `annotation-test-draft.csv`. All remain pending and are excluded from the working table. Confirm every proposed label before promotion. Test candidates require human label confirmation and must be frozen before model tuning the proposal file is not the frozen test set.

## Split and readiness checks

`validate_dataset_readiness` checks that provenance IDs match complaint IDs, that normalized exact duplicates do not cross splits, and reports possible cross-split text variants for manual review. Similarity candidates are review prompts, not automatic duplicate decisions. Human review is still needed for paraphrases, translations and transliterations that automated text matching cannot reliably identify.

Use the checks together after loading both files:

```python
from civiclens.data import load_dataset, load_provenance, validate_dataset_readiness

rows = load_dataset("data/interim/complaints.csv")
sources = load_provenance("data/interim/provenance.csv")
report = validate_dataset_readiness(rows, sources)
```

For a compact report from the project root, run `python scripts/check_dataset.py`. Use `--complaints` and `--provenance` to check alternate files. Target shortfalls are shown in the report; they do not make a preparation dataset invalid.

The TF-IDF baseline is in `src/civiclens/baseline.py`. Run `python scripts/train_baseline.py` to fit on train rows and report development metrics. The command excludes test rows from fitting and reporting. Confidence intervals on development metrics describe only the development sample; final test metrics must be generated after the test labels are reviewed and frozen.

The readiness report counts rows by split and measures progress toward 300 total examples, 100 human-labeled test examples and 20 emergency test examples. Target shortfalls are reported without rejecting the synthetic pilot. Keep the test set separate and frozen before model tuning - do not use the synthetic pilot for performance claims.
