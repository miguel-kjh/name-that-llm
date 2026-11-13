import math

import pandas as pd
import pytest

from name_that_llm.baseline import make_baseline
from name_that_llm.metrics import (
    accuracy_by_length,
    balanced_accuracy,
    confusion,
    lab_balanced_accuracy,
    per_class_recall,
    summarize,
    within_lab_error_share,
    within_lab_error_share_chance,
)

TRAIN_MODELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]


def test_balanced_accuracy_by_hand():
    # Recall a = 2/2, b = 1/2 -> mean 0.75 (plain accuracy would also be 0.75 here).
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "a", "b", "a"]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_balanced_accuracy_differs_from_accuracy_when_unbalanced():
    # Recall a = 3/3, b = 0/1 -> 0.5, while plain accuracy is 0.75.
    assert balanced_accuracy(["a", "a", "a", "b"], ["a", "a", "a", "a"]) == pytest.approx(0.5)


def test_per_class_recall_by_hand():
    recall = per_class_recall(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert recall.to_dict() == {"a": 0.5, "b": 1.0}


def test_confusion_rows_are_truth():
    cm = confusion(["a", "a", "b"], ["a", "b", "b"])
    assert cm.loc["a", "b"] == 1 and cm.loc["b", "a"] == 0 and cm.values.sum() == 3


def test_within_lab_error_share_on_toy_case():
    y_true = ["gpt-5.6-luna", "gpt-5.6-luna", "claude-opus-5", "kimi-k3", "qwen3.8-max"]
    y_pred = ["gpt-5.6-sol", "claude-opus-5", "claude-sonnet-5", "kimi-k3", "kimi-k3"]
    # Errors: luna->sol (same lab), luna->claude (other), opus->sonnet (same), qwen->kimi (other)
    assert within_lab_error_share(y_true, y_pred) == pytest.approx(0.5)


def test_within_lab_error_share_is_nan_without_errors():
    assert math.isnan(within_lab_error_share(["kimi-k3"], ["kimi-k3"]))


def test_within_lab_chance_for_competition_models():
    # Labs of size 5, 5, 4, 2, 1, 1 -> (20 + 20 + 12 + 2) / (18 * 17) = 54 / 306
    assert within_lab_error_share_chance(TRAIN_MODELS) == pytest.approx(54 / 306)


def test_lab_balanced_accuracy_forgives_within_lab_errors():
    y_true = ["gpt-5.6-luna", "gpt-5.6-sol", "kimi-k3"]
    y_pred = ["gpt-5.6-sol", "gpt-5.6-luna", "kimi-k3"]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(1 / 3)
    assert lab_balanced_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_accuracy_by_length_buckets():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "b", "a", "b"]
    table = accuracy_by_length(y_true, y_pred, [40, 79, 80, 200]).set_index("bucket")
    assert table["count"].tolist() == [2, 1, 1]
    assert table.loc["<80", "accuracy"] == pytest.approx(0.5)
    assert table.loc["80-149", "accuracy"] == pytest.approx(1.0)


def test_summarize_counts_errors():
    s = summarize(["kimi-k3", "qwen3.8-max"], ["kimi-k3", "kimi-k3"])
    assert s["n"] == 2 and s["n_errors"] == 1 and s["accuracy"] == pytest.approx(0.5)


def test_baseline_learns_an_obvious_character_signal():
    # Class identity is a typographic habit (em dash vs semicolon), the kind of
    # signal character n-grams should pick up.
    texts = [f"word{i} — more words here" for i in range(30)] + [
        f"word{i}; more words here" for i in range(30)
    ]
    labels = ["dash"] * 30 + ["semicolon"] * 30
    model = make_baseline().fit(texts[::2], labels[::2])
    preds = model.predict(texts[1::2])
    assert (pd.Series(preds) == pd.Series(labels[1::2])).mean() == 1.0
