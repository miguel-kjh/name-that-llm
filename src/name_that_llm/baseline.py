"""Mission baseline: character n-gram TF-IDF + logistic regression (issue #3).

The configuration is fixed and untuned on purpose: it is the reference that
later methods must beat, so it must not change between experiments.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASELINE_CONFIG = {
    "tfidf": {
        "analyzer": "char_wb",
        "ngram_range": (1, 5),
        "sublinear_tf": True,
        "min_df": 2,
        "lowercase": False,  # capitalisation is a style signal
    },
    "logreg": {
        "C": 10.0,
        "max_iter": 2000,
    },
}


def make_baseline() -> Pipeline:
    """A fresh, unfitted baseline pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**BASELINE_CONFIG["tfidf"])),
            ("logreg", LogisticRegression(**BASELINE_CONFIG["logreg"])),
        ]
    )
