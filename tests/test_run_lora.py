"""E004 runner logic on CPU: reduced protocol, budget projection, comparisons, arms."""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from name_that_llm.experiments import run_lora  # noqa: E402
from name_that_llm.experiments.run_lora import (  # noqa: E402
    ARMS,
    MAX_EPOCHS,
    MICRO_BATCH,
    SEARCH_LRS,
    chosen_configs,
    ordered,
    paired,
    projected_hours,
    search_lrs,
)
from name_that_llm.experiments.runs import append_row, oof_path  # noqa: E402

FOLDS = (0, 1)


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


def _q0_search(tmp_path, best_lr=3e-4, best_epochs=(4, 5)):
    for lr in SEARCH_LRS:
        for fold in FOLDS:
            _write_search(tmp_path, "Q0", lr, fold, set(best_epochs) if lr == best_lr else set())


# ---------------------------------------------------------------- reduced protocol


def test_g_and_q0_search_the_lr_grid(tmp_path):
    assert search_lrs("G", tmp_path, FOLDS) == SEARCH_LRS
    assert search_lrs("Q0", tmp_path, FOLDS) == SEARCH_LRS


@pytest.mark.parametrize("arm", ["Q4", "Q8"])
def test_large_arms_inherit_q0s_chosen_lr(tmp_path, arm):
    _q0_search(tmp_path, best_lr=3e-4)
    assert search_lrs(arm, tmp_path, FOLDS) == (3e-4,)


@pytest.mark.parametrize("arm", ["Q4", "Q8"])
def test_large_arms_refuse_without_a_complete_q0_search(tmp_path, arm):
    _write_search(tmp_path, "Q0", 1e-4, 0, set())  # one lr, one fold only
    with pytest.raises(SystemExit):
        search_lrs(arm, tmp_path, FOLDS)


def test_q4_picks_only_its_stopping_epoch_at_q0s_lr(tmp_path):
    _q0_search(tmp_path, best_lr=1e-4, best_epochs=(5,))
    for fold in FOLDS:  # Q4 only ran Q0's lr, best at epoch 2
        _write_search(tmp_path, "Q4", 1e-4, fold, {2, 3})
    chosen = chosen_configs(tmp_path, ["Q4", "Q0"], FOLDS)
    assert list(chosen) == ["Q0", "Q4"]  # experiment order, whatever the CLI order
    assert (chosen["Q0"]["lr"], chosen["Q0"]["epochs"]) == (1e-4, 5)
    assert (chosen["Q4"]["lr"], chosen["Q4"]["epochs"]) == (1e-4, 2)


def test_arms_run_in_experiment_order():
    assert ordered(["Q8", "G", "Q4", "Q0"]) == ["G", "Q0", "Q4", "Q8"]
    assert ordered(["Q8", "Q0"]) == ["Q0", "Q8"]


def test_stop_rule_is_on_q0():
    assert not run_lora.passes_stop_rule({"balanced_accuracy": 0.19})
    assert run_lora.passes_stop_rule({"balanced_accuracy": 0.20})


# ---------------------------------------------------------------- budget


def test_projected_hours_counts_every_planned_epoch():
    # 1 s per epoch, no validation or loading cost: count the epochs.
    ones = {a: 1.0 for a in ARMS}
    zeros = {a: 0.0 for a in ARMS}
    hours = projected_hours(ones, zeros, zeros)["per_arm_hours"]
    # G: search 2 lr x 5 folds x 6 ep = 60; eval 2 seeds x 5 folds x 6 ep (worst case) = 60.
    assert hours["G"] * 3600 == pytest.approx(120)
    # Q0: as G plus the shuffled-label control, 5 folds x 6 ep = 30.
    assert hours["Q0"] * 3600 == pytest.approx(150)
    # Q4, Q8: search 1 lr x 5 x 6 = 30; eval 60.
    assert hours["Q4"] * 3600 == pytest.approx(90)
    assert hours["Q8"] * 3600 == pytest.approx(90)


def test_projection_adds_validation_and_loading_and_flags_overruns():
    one = {"Q8": 1.0}
    p = projected_hours({"Q8": 100.0}, one, one)
    # search: 5 runs x (6 x (100 + 1) + 1); eval: 10 runs x (6 x 100 + 1 + 1)
    assert p["total_hours"] * 3600 == pytest.approx(5 * 607 + 10 * 602)
    assert p["within_budget"]
    assert not projected_hours({"Q8": 3600.0}, one, one)["within_budget"]


# ---------------------------------------------------------------- comparisons


def _scores(values, seeds=(1, 1, 2, 2), folds=(0, 1, 0, 1)):
    return pd.DataFrame({"seed": seeds, "fold": folds, "balanced_accuracy": values})


def test_paired_compares_the_same_folds():
    a = _scores([0.40, 0.30, 0.50, 0.35])
    b = _scores([0.35, 0.30, 0.40, 0.40]).iloc[::-1]  # other row order
    result = paired(a, b)
    assert result["mean_diff"] == pytest.approx((0.05 + 0 + 0.10 - 0.05) / 4)
    assert (result["wins_a"], result["wins_b"], result["n_folds"]) == (2, 1, 4)


# ---------------------------------------------------------------- arm design


def test_arms_follow_the_e004_design():
    assert set(ARMS) == {"G", "Q0", "Q4", "Q8"}
    for key, arm in ARMS.items():
        assert arm.lora is not None and arm.lora.r == 16 and arm.lora.alpha == 32
        assert arm.dtype == torch.bfloat16 and arm.normalize
        assert 16 % MICRO_BATCH[key] == 0  # effective batch 16 in every arm
    assert ARMS["G"].pooling == "mean" and ARMS["G"].text_format.end_token is None
    assert ARMS["G"].text_format.prompt == "task: classification | query: "
    for key in ("Q0", "Q4", "Q8"):
        assert ARMS[key].pooling == "last"
        assert ARMS[key].text_format.end_token == "<|endoftext|>"  # what Qwen3-Embedding pools
        assert ARMS[key].text_format.prompt.startswith("Instruct: ")
    assert ARMS["Q8"].gradient_checkpointing


def test_every_arm_uses_the_6_epoch_schedule_and_batch_16():
    for key in ARMS:
        cfg = run_lora.base_config(key)
        assert (cfg.schedule_epochs, cfg.batch_size, cfg.max_length) == (MAX_EPOCHS, 16, 512)
