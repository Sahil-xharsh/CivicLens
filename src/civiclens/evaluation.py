from collections import defaultdict
import random

from sklearn.metrics import f1_score, precision_recall_fscore_support

from civiclens.taxonomy import CATEGORIES, URGENCY_LEVELS


def _bootstrap_macro_f1_ci(
    actual: list[str],
    predicted: list[str],
    labels: tuple[str, ...],
    *,
    iterations: int = 1_000,
    seed: int = 42,
) -> list[float] | None:
    if not actual:
        return None
    rng = random.Random(seed)
    scores = []
    for _ in range(iterations):
        indices = [rng.randrange(len(actual)) for _ in actual]
        scores.append(
            f1_score(
                [actual[index] for index in indices],
                [predicted[index] for index in indices],
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        )

    ordered_scores = sorted(scores)

    def quantile(value: float) -> float:
        position = (len(ordered_scores) - 1) * value
        lower = int(position)
        upper = min(lower + 1, len(ordered_scores) - 1)
        fraction = position - lower
        return ordered_scores[lower] + fraction * (
            ordered_scores[upper] - ordered_scores[lower]
        )

    lower, upper = quantile(0.025), quantile(0.975)
    return [float(lower), float(upper)]


def _per_class_metrics(
    actual: list[str],
    predicted: list[str],
    labels: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(labels),
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(labels)
    }


def _wilson_interval(caught: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = caught / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate_development(
    model,
    rows: list[dict],
    *,
    emergency_threshold: float = 0.5,
) -> dict:
    """Evaluate on dev rows only and return counts, metrics, and intervals."""
    if not rows:
        raise ValueError("Development rows cannot be empty.")
    invalid = [row.get("complaint_id", "?") for row in rows if row.get("split") != "dev"]
    if invalid:
        raise ValueError(
            "Development evaluation accepts dev rows only; rejected: "
            + ", ".join(invalid)
        )

    predictions = model.predict_many(
        [row["text"] for row in rows],
        emergency_threshold=emergency_threshold,
    )
    category_actual = [row["category"] for row in rows]
    category_predicted = [prediction.category for prediction in predictions]
    urgency_actual = [row["urgency"] for row in rows]
    urgency_predicted = [prediction.urgency for prediction in predictions]
    emergency_actual = [bool(row["is_emergency"]) for row in rows]
    emergency_predicted = [prediction.is_emergency for prediction in predictions]

    caught = sum(actual and predicted for actual, predicted in zip(emergency_actual, emergency_predicted))
    emergency_total = sum(emergency_actual)
    category_f1 = float(
        f1_score(
            category_actual,
            category_predicted,
            labels=list(CATEGORIES),
            average="macro",
            zero_division=0,
        )
    )
    urgency_f1 = float(
        f1_score(
            urgency_actual,
            urgency_predicted,
            labels=list(URGENCY_LEVELS),
            average="macro",
            zero_division=0,
        )
    )

    by_language = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_language[row["language"]].append(
            row["category"] == prediction.category
        )

    return {
        "split": "dev",
        "example_count": len(rows),
        "emergency_threshold": emergency_threshold,
        "category": {
            "macro_f1": category_f1,
            "macro_f1_ci_95": _bootstrap_macro_f1_ci(
                category_actual, category_predicted, CATEGORIES
            ),
            "per_category": _per_class_metrics(
                category_actual, category_predicted, CATEGORIES
            ),
        },
        "urgency": {
            "macro_f1": urgency_f1,
            "macro_f1_ci_95": _bootstrap_macro_f1_ci(
                urgency_actual, urgency_predicted, URGENCY_LEVELS
            ),
            "per_urgency": _per_class_metrics(
                urgency_actual, urgency_predicted, URGENCY_LEVELS
            ),
        },
        "emergency_recall": {
            "caught": caught,
            "total": emergency_total,
            "value": caught / emergency_total if emergency_total else None,
            "ci_95": _wilson_interval(caught, emergency_total),
        },
        "category_accuracy_by_language": {
            language: sum(scores) / len(scores)
            for language, scores in sorted(by_language.items())
        },
    }
