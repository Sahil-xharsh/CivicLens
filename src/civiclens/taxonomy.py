"""Labels and CSV columns used by CivicLens."""

CATEGORIES = (
    "roads",
    "water",
    "sanitation",
    "electricity",
    "drainage",
    "streetlights",
    "other",
)

URGENCY_LEVELS = ("low", "medium", "high", "critical")
LANGUAGES = ("en", "hi", "hinglish", "mixed")
LABEL_SOURCES = ("human", "synthetic", "public")
SPLITS = ("train", "dev", "test")

CSV_FIELDS = (
    "complaint_id",
    "text",
    "language",
    "category",
    "urgency",
    "is_emergency",
    "label_source",
    "split",
)

