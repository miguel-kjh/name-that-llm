import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from name_that_llm.experiments.run_encoder import (  # noqa: E402
    MAX_EPOCHS,
    SEARCH_LRS,
    append_row,
    chosen_configs,
    completed_keys,
    control_train_labels,
    load_folds,
    oof_path,
    read_oof,
)


@pytest.fixture
def train():
    return pd.DataFrame({"id": ["a", "b", "c", "d"], "text": list("wxyz")})


@pytest.fixture
def folds_csv():
    # Stored in a different row order than train, with two seeds.
    return pd.DataFrame({
        "id": ["d", "c", "b", "a", "a", "b", "c", "d"],
        "seed": [0, 0, 0, 0, 1, 1, 1, 1],
        "fold": [3, 2, 1, 0, 1, 1, 0, 0],
    })


def test_load_folds_aligns_by_id(train, folds_csv):
    assert load_folds(train, folds_csv, seed=0).tolist() == [0, 1, 2, 3]
    assert load_folds(train, folds_csv, seed=1).tolist() == [1, 1, 0, 0]


def test_load_folds_keeps_train_index(train, folds_csv):
    shifted = train.set_index(pd.Index([10, 11, 12, 13]))
    assert load_folds(shifted, folds_csv, seed=0).index.equals(shifted.index)


def test_load_folds_fails_on_missing_ids(train, folds_csv):
    with pytest.raises(ValueError):
        load_folds(train, folds_csv[folds_csv.id != "c"], seed=0)


def test_control_labels_are_permuted_inside_training_folds_only():
    y = np.arange(100)
    folds = pd.Series(np.arange(100) % 5)
    shuffled = control_train_labels(y, folds, fold=2, seed=0)
    is_val = (folds == 2).to_numpy()
    assert (shuffled[is_val] == y[is_val]).all()  # validation untouched
    assert sorted(shuffled[~is_val]) == sorted(y[~is_val])  # same labels, same train rows
    assert (shuffled[~is_val] != y[~is_val]).any()  # but in another order
    assert (y == np.arange(100)).all()  # input not modified


def _row(fold, **extra):
    return {"phase": "search", "arm": "A", "lr": 3e-5, "epochs": 4, "seed": 0,
            "fold": fold, "balanced_accuracy": 0.4, **extra}


def test_resume_sees_runs_already_written(tmp_path):
    csv = tmp_path / "search.csv"
    assert completed_keys(csv) == set()
    append_row(csv, _row(0))
    append_row(csv, _row(1))
    done = completed_keys(csv)
    assert ("search", "A", 3e-5, 4, 0, 1) in done
    assert ("search", "A", 3e-5, 4, 0, 2) not in done
    assert len(pd.read_csv(csv)) == 2  # header written once


def test_oof_paths_are_unique_per_run(tmp_path):
    keys = [("search", arm, lr, MAX_EPOCHS, 0, f) for arm in "AB" for lr in SEARCH_LRS
            for f in range(5)]
    keys += [("eval", arm, lr, ep, s, f) for arm in "AB" for lr in SEARCH_LRS
             for ep in range(1, MAX_EPOCHS + 1) for s in (1, 2) for f in range(5)]
    assert len({oof_path(tmp_path, k) for k in keys}) == len(keys)


def test_read_oof_tags_seed_and_fold(tmp_path):
    keys = [("eval", "A", 3e-5, 4, 1, 0), ("eval", "A", 3e-5, 4, 2, 3)]
    for k in keys:
        path = oof_path(tmp_path, k)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ["x"], "model": ["m"], "pred": ["m"]}).to_parquet(path)
    oof = read_oof(tmp_path, keys)
    assert oof[["seed", "fold"]].values.tolist() == [[1, 0], [2, 3]]


def _write_search(tmp_path, arm, lr, fold, correct_epochs):
    """A fake search run: 2 validation rows per epoch, right only at `correct_epochs`."""
    key = ("search", arm, lr, MAX_EPOCHS, 0, fold)
    rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        for i, true in enumerate(["m1", "m2"]):
            right = epoch in correct_epochs
            rows.append({"id": f"{fold}-{i}", "model": true, "epoch": epoch,
                         "pred": true if right else ("m2" if true == "m1" else "m1")})
    path = oof_path(tmp_path, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    append_row(tmp_path / "search.csv", dict(zip(
        ["phase", "arm", "lr", "epochs", "seed", "fold"], key)))


def test_chosen_config_picks_best_lr_and_epoch(tmp_path):
    folds = (0, 1)
    for lr in SEARCH_LRS:
        for fold in folds:
            # Only lr 3e-5 is right, and only at epochs 6 and 7: fewer epochs win ties.
            _write_search(tmp_path, "A", lr, fold, {6, 7} if lr == 3e-5 else set())
    choice = chosen_configs(tmp_path, ["A"], folds)["A"]
    assert (choice["lr"], choice["epochs"]) == (3e-5, 6)
    assert choice["balanced_accuracy"] == 1.0


def test_chosen_config_refuses_an_incomplete_search(tmp_path):
    for lr in SEARCH_LRS:
        _write_search(tmp_path, "A", lr, 0, set())
    with pytest.raises(SystemExit):
        chosen_configs(tmp_path, ["A"], (0, 1))


def test_stop_rule_needs_one_arm_at_the_floor():
    from name_that_llm.experiments.run_encoder import STOP_IF_SEARCH_BELOW, passes_stop_rule

    below = {"A": {"balanced_accuracy": STOP_IF_SEARCH_BELOW - 0.01},
             "B": {"balanced_accuracy": 0.1}}
    assert not passes_stop_rule(below)
    assert passes_stop_rule({**below, "B": {"balanced_accuracy": STOP_IF_SEARCH_BELOW}})
