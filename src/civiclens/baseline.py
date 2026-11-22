from dataclasses import asdict, dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Prediction:
    category: str
    urgency: str
    is_emergency: bool
    emergency_probability: float
    confidence: float


def _classifier() -> Pipeline:
    return Pipeline(
        steps=(
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    max_features=50_000,
                ),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        )
    )


class TfidfComplaintBaseline:
    """Fit category, urgency, and emergency-risk classifiers on training rows."""

    def __init__(self) -> None:
        self._category_model = _classifier()
        self._urgency_model = _classifier()
        self._emergency_model = _classifier()
        self._fitted = False

    def fit(self, rows: list[dict]) -> "TfidfComplaintBaseline":
        if not rows:
            raise ValueError("Training rows cannot be empty.")
        nontraining = [row.get("complaint_id", "?") for row in rows if row.get("split") != "train"]
        if nontraining:
            raise ValueError(
                "Baseline fitting accepts training rows only; rejected: "
                + ", ".join(nontraining)
            )

        texts = [row["text"] for row in rows]
        self._category_model.fit(texts, [row["category"] for row in rows])
        self._urgency_model.fit(texts, [row["urgency"] for row in rows])
        self._emergency_model.fit(
            texts,
            ["true" if row["is_emergency"] else "false" for row in rows],
        )
        self._fitted = True
        return self

    def predict(self, text: str, *, emergency_threshold: float = 0.5) -> Prediction:
        if not self._fitted:
            raise RuntimeError("Call fit with training rows before prediction.")
        if not text.strip():
            raise ValueError("Complaint text cannot be empty.")
        if not 0 <= emergency_threshold <= 1:
            raise ValueError("emergency_threshold must be between 0 and 1.")

        category_probabilities = self._category_model.predict_proba([text])[0]
        category_index = int(category_probabilities.argmax())
        emergency_classes = list(self._emergency_model.classes_)
        emergency_index = emergency_classes.index("true")
        emergency_probability = float(
            self._emergency_model.predict_proba([text])[0][emergency_index]
        )
        return Prediction(
            category=str(self._category_model.classes_[category_index]),
            urgency=str(self._urgency_model.predict([text])[0]),
            is_emergency=emergency_probability >= emergency_threshold,
            emergency_probability=emergency_probability,
            confidence=float(category_probabilities[category_index]),
        )

    def predict_many(
        self,
        texts: list[str],
        *,
        emergency_threshold: float = 0.5,
    ) -> list[Prediction]:
        return [
            self.predict(text, emergency_threshold=emergency_threshold)
            for text in texts
        ]


def prediction_dict(prediction: Prediction) -> dict:
    """Return the shared prediction shape as a plain dictionary."""
    return asdict(prediction)
