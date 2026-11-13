import numpy as np
import pandas as pd
import pytest

from name_that_llm.validation import make_folds, out_of_fold_predictions

N_PROMPTS, N_MODELS = 150, 18


@pytest.fixture
def crossed():
    """Fully crossed design like train.csv, minus one row."""
    rows = [
        {"id": f"{p}-{m}", "prompt_id": p, "model": f"m{m}"}
        for p in range(N_PROMPTS)
        for m in range(N_MODELS)
    ]
    return pd.DataFrame(rows).drop(index=7).reset_index(drop=True)


def test_no_prompt_is_split_across_folds(crossed):
    folds = make_folds(crossed)
    assert (crossed.assign(fold=folds).groupby("prompt_id").fold.nunique() == 1).all()


def test_every_row_gets_exactly_one_valid_fold(crossed):
    folds = make_folds(crossed)
    assert len(folds) == len(crossed)
    assert set(folds) == set(range(5))


def test_folds_have_about_thirty_prompts(crossed):
    folds = make_folds(crossed)
    prompts_per_fold = crossed.assign(fold=folds).groupby("fold").prompt_id.nunique()
    assert prompts_per_fold.between(29, 31).all()
    assert prompts_per_fold.sum() == N_PROMPTS


def test_same_seed_same_folds_and_other_seed_differs(crossed):
    assert make_folds(crossed, seed=0).equals(make_folds(crossed, seed=0))
    assert not make_folds(crossed, seed=0).equals(make_folds(crossed, seed=1))


class _Memorizer:
    """Predicts the label seen with each text in training, else 'unseen'."""

    def fit(self, X, y):
        self.table = dict(zip(X, y))
        return self

    def predict(self, X):
        return np.array([self.table.get(x, "unseen") for x in X])


def test_out_of_fold_predictions_never_see_their_own_fold():
    # Each text appears once, so a memorizer can only fail if training excludes it.
    texts = pd.Series([f"t{i}" for i in range(20)])
    labels = pd.Series(["a", "b"] * 10)
    folds = pd.Series(np.arange(20) % 4)
    preds = out_of_fold_predictions(texts, labels, folds, _Memorizer)
    assert (preds == "unseen").all()
    assert preds.index.equals(texts.index)
