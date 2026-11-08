# Dataset preparation contract

The complaint table keeps the fields documented in the label guide. Store text-source provenance in a companion CSV keyed by `complaint_id`, so public-source details do not get mixed with label provenance or model inputs.

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

Each complaint must have exactly one provenance row. Public records must include a license and access date. Keep restricted or personally identifying source material out of the repository, and remove direct personal identifiers from complaint text before model use.

## Annotation drafts

`data/interim/annotation-draft-*.csv` records authored examples with proposed labels and review status. Draft rows use `draft_id` and `proposed_*` fields; they do not follow the complaint CSV contract. Assistant-reviewed authored examples may be promoted to training/development with `label_source=synthetic` and a provenance record. Only labels confirmed by a human may use `label_source=human`, and only human-labeled rows may enter the frozen test set.

The current working table is `data/interim/complaints.csv`; it combines the 30-row pilot and the first 20 authored training/development examples. It contains no test data and is not suitable for performance evaluation.

## Split and readiness checks

`validate_dataset_readiness` checks that provenance IDs match complaint IDs, that normalized exact duplicates do not cross splits, and reports possible cross-split text variants for manual review. Similarity candidates are review prompts, not automatic duplicate decisions. Human review is still needed for paraphrases, translations, and transliterations that automated text matching cannot reliably identify.

Use the checks together after loading both files:

```python
from civiclens.data import load_dataset, load_provenance, validate_dataset_readiness

rows = load_dataset("data/interim/complaints.csv")
sources = load_provenance("data/interim/provenance.csv")
report = validate_dataset_readiness(rows, sources)
```

The readiness report counts rows by split and measures progress toward 300 total examples, 100 human-labeled test examples, and 20 emergency test examples. Target shortfalls are reported without rejecting the synthetic pilot. Keep the test set separate and frozen before model tuning; do not use the synthetic pilot for performance claims.
