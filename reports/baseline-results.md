# TF-IDF baseline results

## Evaluation setup

- Training: 163 rows, including 30 synthetic pilot examples.
- Development: 37 human-labeled rows; three are emergency cases.
- Frozen test: 100 human-labeled rows, including 24 emergency cases.
- Emergency threshold: 0.55, selected on development data by maximizing emergency F1; tied scores prefer the higher threshold.
- The frozen test checksum is recorded in `data/processed/test-manifest.json`.

## Frozen test metrics

| Metric | Result |
| --- | ---: |
| Category macro-F1 | 0.763 (95% bootstrap interval: 0.674–0.842) |
| Urgency macro-F1 | 0.534 (95% bootstrap interval: 0.402–0.630) |
| Emergency recall | 6/24 = 0.250 (95% Wilson interval: 0.120–0.449) |
| Records with one or more incorrect outputs | 71/100 |

Emergency recall is weak: the baseline missed 18 of 24 emergency cases. The emergency threshold came from only three development emergencies, so the selection is unstable. This result does not support dependable emergency handling.

Category F1 was highest for streetlights (0.870) and drainage (0.839). Urgency F1 was 0.250 for `low`, 0.607 for `medium`, 0.611 for `high`, and 0.667 for `critical`. The full per-class metrics and error records are in `baseline-test-report.json`.

## Limits and next evaluation

The corpus is authored and small, with a synthetic pilot in training. These metrics describe only this dataset and are not real-world service-performance estimates. TF-IDF probabilities are not calibrated confidence estimates.

The frozen test file has now been evaluated. Keep it unchanged and do not tune this baseline using its errors. Compare later models against the same frozen set with all model and threshold choices fixed from train/development data; treat this baseline result as the recorded reference.
