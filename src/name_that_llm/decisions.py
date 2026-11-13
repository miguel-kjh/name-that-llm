"""Pre-registered decision rules of E003 (H003 and H004, issue #5).

Pure functions with no torch dependency, so the rules that decide each
hypothesis can be tested without a GPU. Thresholds were fixed in H003, H004
and E003 before any run.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

CONFIRMED, REJECTED, INCONCLUSIVE = "CONFIRMED", "REJECTED", "INCONCLUSIVE"

BASELINE_BALANCED_ACCURACY = 0.312  # F002, mean over fold seeds 0-2
H003_CONFIRM_AT = 0.412  # baseline + 0.10
H003_REJECT_BELOW = 0.362  # baseline + 0.05
H004_MIN_DIFF = 0.02
H004_MIN_WINS = 8  # out of 10 folds (fold seeds 1 and 2)
CONFIG_TIE_TOLERANCE = 0.005
MIN_LOSS_DROP = 0.5  # final loss must be at most half the initial loss
MIN_RUN_BALANCED_ACCURACY = 0.10

# Absorbs float error so that a mean of exactly 0.412 counts as 0.412.
_EPS = 1e-9


def run_is_stable(loss_history: Sequence[float], balanced_accuracy: float | None = None) -> bool:
    """A run trained if its loss at least halved and, when given, it beat 0.10.

    `loss_history[0]` is the loss before any update (first batch) and
    `loss_history[-1]` the mean loss of the last epoch.
    """
    if len(loss_history) < 2 or not np.all(np.isfinite(loss_history)):
        return False
    if loss_history[-1] > MIN_LOSS_DROP * loss_history[0] + _EPS:
        return False
    if balanced_accuracy is not None and balanced_accuracy < MIN_RUN_BALANCED_ACCURACY - _EPS:
        return False
    return True


def h003_status(best_mean: float, all_stable: bool, grid_complete: bool = True) -> str:
    """H003: the best arm's mean balanced accuracy over fold seeds 1 and 2."""
    if not all_stable:
        return INCONCLUSIVE
    if best_mean >= H003_CONFIRM_AT - _EPS:
        return CONFIRMED
    if best_mean < H003_REJECT_BELOW - _EPS and grid_complete:
        return REJECTED
    return INCONCLUSIVE


def h004_status(
    fold_a: Sequence[float],
    fold_b: Sequence[float],
    all_stable: bool = True,
    baseline: float = BASELINE_BALANCED_ACCURACY,
) -> dict:
    """H004: arm A (MLM) vs arm B (contrastive), paired by fold.

    Ties on a fold count as a win for neither arm.
    """
    a, b = np.asarray(fold_a, dtype=float), np.asarray(fold_b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("fold_a and fold_b must be non-empty and paired by fold")
    diff = float(a.mean() - b.mean())
    wins_a, wins_b = int((a > b).sum()), int((b > a).sum())
    result = {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff_a_minus_b": diff,
              "wins_a": wins_a, "wins_b": wins_b, "n_folds": int(a.size)}

    both_valid = all_stable and a.mean() > baseline and b.mean() > baseline
    if not both_valid:
        status = INCONCLUSIVE
    elif diff >= H004_MIN_DIFF - _EPS and wins_a >= H004_MIN_WINS:
        status = CONFIRMED
    elif -diff >= H004_MIN_DIFF - _EPS and wins_b >= H004_MIN_WINS:
        status = REJECTED
    else:
        status = INCONCLUSIVE
    return {**result, "status": status}


def pick_config(search: pd.DataFrame) -> dict:
    """Best (lr, epochs) by `balanced_accuracy`; near-ties go to fewer epochs, then lower lr.

    `search` has one row per configuration with columns lr, epochs and
    balanced_accuracy (the OOF score of fold seed 0).
    """
    best = search.balanced_accuracy.max()
    near = search[search.balanced_accuracy >= best - CONFIG_TIE_TOLERANCE - _EPS]
    row = near.sort_values(["epochs", "lr"]).iloc[0]
    return {"lr": float(row.lr), "epochs": int(row.epochs),
            "balanced_accuracy": float(row.balanced_accuracy)}
