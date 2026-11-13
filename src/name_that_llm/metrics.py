"""Evaluation metrics for model attribution (issue #3).

All functions take true and predicted model names and are pure.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from name_that_llm.labels import lab_of

# Word-count buckets for accuracy by length: [0, 80), [80, 150), [150, inf).
LENGTH_BUCKET_EDGES = (0, 80, 150, np.inf)
LENGTH_BUCKET_LABELS = ("<80", "80-149", "150+")


def _labs(models: Iterable[str]) -> list[str]:
    return [lab_of(m) for m in models]


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def lab_balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Balanced accuracy after mapping both true and predicted models to labs."""
    return float(balanced_accuracy_score(_labs(y_true), _labs(y_pred)))


def per_class_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.Series:
    labels = sorted(set(y_true))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return pd.Series(recalls, index=labels, name="recall")


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Confusion counts: rows are true models, columns are predicted models."""
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def within_lab_error_share(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Share of errors whose predicted model is from the true model's lab.

    NaN when there are no errors.
    """
    errors = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    if not errors:
        return float("nan")
    same_lab = sum(lab_of(t) == lab_of(p) for t, p in errors)
    return same_lab / len(errors)


def within_lab_error_share_chance(models: Iterable[str]) -> float:
    """Within-lab error share if each error picked a wrong model uniformly at random.

    Assumes balanced classes: sum over labs of s(s-1) / (k(k-1)), s = lab size.
    """
    models = sorted(set(models))
    k = len(models)
    lab_sizes = Counter(_labs(models)).values()
    return sum(s * (s - 1) for s in lab_sizes) / (k * (k - 1))


def accuracy_by_length(
    y_true: Sequence[str], y_pred: Sequence[str], n_words: Sequence[int]
) -> pd.DataFrame:
    """Count, accuracy and balanced accuracy per word-count bucket."""
    df = pd.DataFrame({"true": list(y_true), "pred": list(y_pred)})
    df["bucket"] = pd.cut(
        list(n_words),
        bins=LENGTH_BUCKET_EDGES,
        labels=LENGTH_BUCKET_LABELS,
        right=False,
    )
    rows = []
    for bucket, group in df.groupby("bucket", observed=False):
        rows.append(
            {
                "bucket": bucket,
                "count": len(group),
                "accuracy": float((group.true == group.pred).mean()) if len(group) else np.nan,
                "balanced_accuracy": (
                    balanced_accuracy(group.true, group.pred) if len(group) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Headline metrics for one set of predictions."""
    n_errors = sum(t != p for t, p in zip(y_true, y_pred))
    return {
        "n": len(y_true),
        "n_errors": n_errors,
        "accuracy": 1 - n_errors / len(y_true),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "lab_balanced_accuracy": lab_balanced_accuracy(y_true, y_pred),
        "within_lab_error_share": within_lab_error_share(y_true, y_pred),
    }
