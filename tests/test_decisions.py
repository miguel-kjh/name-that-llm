import pandas as pd
import pytest

from name_that_llm.decisions import (
    CONFIRMED,
    INCONCLUSIVE,
    REJECTED,
    h003_status,
    h004_status,
    pick_config,
    run_is_stable,
)


@pytest.mark.parametrize(
    "best_mean, expected",
    [
        (0.412, CONFIRMED),
        (0.5, CONFIRMED),
        (0.4119, INCONCLUSIVE),
        (0.362, INCONCLUSIVE),
        (0.3619, REJECTED),
        (0.2, REJECTED),
    ],
)
def test_h003_boundaries(best_mean, expected):
    assert h003_status(best_mean, all_stable=True) == expected


def test_h003_float_noise_at_the_threshold_still_confirms():
    assert h003_status((0.41 + 0.414) / 2, all_stable=True) == CONFIRMED


def test_h003_unstable_runs_are_always_inconclusive():
    assert h003_status(0.6, all_stable=False) == INCONCLUSIVE
    assert h003_status(0.2, all_stable=False) == INCONCLUSIVE


def test_h003_incomplete_grid_cannot_reject():
    assert h003_status(0.2, all_stable=True, grid_complete=False) == INCONCLUSIVE


def _folds(base, deltas):
    return [base + d for d in deltas]


def test_h004_confirms_with_enough_margin_and_wins():
    b = [0.40] * 10
    a = _folds(0.43, [0] * 8 + [-0.04, -0.04])  # wins 8, mean diff 0.022
    result = h004_status(a, b)
    assert result["wins_a"] == 8
    assert result["status"] == CONFIRMED


def test_h004_seven_wins_is_inconclusive():
    b = [0.40] * 10
    a = [0.45] * 7 + [0.39] * 3
    assert h004_status(a, b)["status"] == INCONCLUSIVE


def test_h004_margin_just_below_threshold_is_inconclusive():
    b = [0.40] * 10
    a = [0.419] * 10  # wins 10, diff 0.019
    assert h004_status(a, b)["status"] == INCONCLUSIVE


def test_h004_margin_exactly_at_threshold_confirms():
    b = [0.40] * 10
    a = [0.42] * 10
    assert h004_status(a, b)["status"] == CONFIRMED


def test_h004_rejects_when_b_wins():
    a = [0.40] * 10
    b = [0.43] * 10
    assert h004_status(a, b)["status"] == REJECTED


def test_h004_needs_both_arms_above_baseline():
    a = [0.40] * 10
    b = [0.30] * 10  # below 0.312
    assert h004_status(a, b)["status"] == INCONCLUSIVE


def test_h004_unstable_is_inconclusive():
    assert h004_status([0.5] * 10, [0.4] * 10, all_stable=False)["status"] == INCONCLUSIVE


def test_h004_fold_ties_count_for_nobody():
    result = h004_status([0.4] * 10, [0.4] * 10)
    assert result["wins_a"] == result["wins_b"] == 0


def test_h004_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        h004_status([0.4] * 10, [0.4] * 9)


def test_pick_config_takes_the_best_when_clear():
    search = pd.DataFrame({"lr": [1e-5, 3e-5, 5e-5], "epochs": [4, 4, 4],
                           "balanced_accuracy": [0.30, 0.40, 0.35]})
    assert pick_config(search)["lr"] == 3e-5


def test_pick_config_near_tie_prefers_fewer_epochs_then_lower_lr():
    search = pd.DataFrame({
        "lr": [5e-5, 3e-5, 1e-5, 5e-5],
        "epochs": [8, 4, 4, 4],
        "balanced_accuracy": [0.404, 0.400, 0.3995, 0.380],
    })
    choice = pick_config(search)
    assert (choice["epochs"], choice["lr"]) == (4, 1e-5)


def test_pick_config_outside_tolerance_is_not_a_tie():
    search = pd.DataFrame({"lr": [1e-5, 3e-5], "epochs": [8, 4],
                           "balanced_accuracy": [0.410, 0.404]})
    assert pick_config(search)["epochs"] == 8


@pytest.mark.parametrize(
    "history, bal_acc, expected",
    [
        ([2.9, 1.0, 0.5], None, True),
        ([2.9, 2.0, 1.45], None, True),  # exactly half
        ([2.9, 2.5, 1.5], None, False),
        ([2.9, 0.5], 0.05, False),
        ([2.9, 0.5], 0.10, True),
        ([2.9, float("nan")], None, False),
        ([2.9], None, False),
    ],
)
def test_run_is_stable(history, bal_acc, expected):
    assert run_is_stable(history, bal_acc) is expected
