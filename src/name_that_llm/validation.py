"""Prompt-grouped cross-validation (issue #3).

The hidden test set uses prompts that never appear in train, so every split
keeps all passages of a prompt in the same fold.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 0, group_col: str = "prompt_id"
) -> pd.Series:
    """Fold number (0..n_splits-1) for each row of `df`, grouped by `group_col`.

    The same `df` and `seed` always give the same folds.
    """
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=df.index, name="fold")
    for fold, (_, val_idx) in enumerate(splitter.split(df, groups=df[group_col])):
        folds.iloc[val_idx] = fold
    return folds


def out_of_fold_predictions(
    texts: pd.Series,
    labels: pd.Series,
    folds: pd.Series,
    make_model: Callable[[], object],
) -> pd.Series:
    """Predict each row with a fresh model trained on all other folds.

    `make_model` must return an unfitted sklearn-style estimator.
    """
    preds = pd.Series(np.empty(len(texts), dtype=object), index=texts.index, name="pred")
    for fold in sorted(folds.unique()):
        is_val = folds == fold
        model = make_model()
        model.fit(texts[~is_val], labels[~is_val])
        preds[is_val] = model.predict(texts[is_val])
    return preds
