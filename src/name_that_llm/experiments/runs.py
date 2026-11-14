"""Fold-level training runs shared by the encoder experiments (E003 #5, E004 #7).

One run trains on four folds of one fold seed and predicts the fifth. Every run
writes its out-of-fold predictions to `oof/` and then appends a row to its CSV,
so an interrupted phase resumes where it stopped. Moved out of `run_encoder.py`
without changing the logic; the arm and the base recipe are now arguments.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from name_that_llm.decisions import pick_config, run_is_stable
from name_that_llm.encoder import (
    EncoderArm,
    EncoderConfig,
    backbone_factory,
    load_tokenizer,
    predict_proba,
    train_classifier,
)
from name_that_llm.metrics import balanced_accuracy

# One training run = one fold. Search runs store predictions for every epoch;
# eval and control runs store their last epoch.
RUN_KEY = ["phase", "arm", "lr", "epochs", "seed", "fold"]


# ---------------------------------------------------------------- pure helpers


def load_folds(train: pd.DataFrame, folds_csv: pd.DataFrame, seed: int) -> pd.Series:
    """Fold of each row of `train` for fold seed `seed`, aligned by `id`."""
    table = folds_csv[folds_csv.seed == seed].set_index("id").fold
    missing = set(train.id) - set(table.index)
    if missing:
        raise ValueError(f"{len(missing)} ids have no fold for seed {seed}")
    return pd.Series(table.loc[train.id].to_numpy(), index=train.index, name="fold")


def control_train_labels(y: np.ndarray, folds: pd.Series, fold: int, seed: int) -> np.ndarray:
    """Labels for the leakage control: permuted inside the training folds only.

    Validation labels are left untouched, so the control is scored against the truth.
    """
    y_train = np.array(y, copy=True)
    is_train = (folds != fold).to_numpy()
    y_train[is_train] = np.random.default_rng(seed).permutation(y_train[is_train])
    return y_train


def completed_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    done = pd.read_csv(csv_path)
    return {run_key(row) for row in done[RUN_KEY].to_dict("records")}


def run_key(row: dict) -> tuple:
    return (row["phase"], row["arm"], float(row["lr"]), int(row["epochs"]),
            int(row["seed"]), int(row["fold"]))


def append_row(csv_path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def oof_path(out_dir: Path, key: tuple) -> Path:
    phase, arm, lr, epochs, seed, fold = key
    return out_dir / "oof" / f"{phase}_{arm}_lr{lr:g}_ep{epochs}_s{seed}_f{fold}.parquet"


def oof_frame(probs: np.ndarray, val: pd.DataFrame, classes: list[str], epoch: int) -> pd.DataFrame:
    oof = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
    oof.insert(0, "id", val.id.to_numpy())
    oof.insert(1, "model", val.model.to_numpy())
    oof.insert(2, "pred", [classes[i] for i in probs.argmax(axis=1)])
    oof.insert(3, "epoch", epoch)
    return oof


def read_oof(out_dir: Path, keys: list[tuple]) -> pd.DataFrame:
    """OOF predictions of the given runs, tagged with their fold seed and fold."""
    return pd.concat(
        [pd.read_parquet(oof_path(out_dir, k)).assign(seed=k[4], fold=k[5]) for k in keys],
        ignore_index=True,
    )


def search_curves(out_dir: Path, arm: str, fold_ids: tuple[int, ...], lrs: tuple[float, ...],
                  max_epochs: int) -> pd.DataFrame:
    """Pooled OOF balanced accuracy of the search seed per (lr, epoch) for one arm."""
    search = pd.read_csv(out_dir / "search.csv")
    rows = []
    for lr in lrs:
        runs = search[(search.arm == arm) & np.isclose(search.lr, lr)
                      & search.fold.isin(fold_ids)]
        if sorted(runs.fold) != sorted(fold_ids):
            raise SystemExit(f"Search for arm {arm} lr={lr:g} is incomplete.")
        oof = read_oof(out_dir, [run_key(r) for r in runs[RUN_KEY].to_dict("records")])
        for epoch in range(1, max_epochs + 1):
            at = oof[oof.epoch == epoch]
            rows.append({"arm": arm, "lr": lr, "epochs": epoch,
                         "balanced_accuracy": balanced_accuracy(at.model, at.pred)})
    return pd.DataFrame(rows)


def best_config(out_dir: Path, arm: str, fold_ids: tuple[int, ...], lrs: tuple[float, ...],
                max_epochs: int) -> dict:
    """Search winner (lr, epoch) of one arm."""
    return pick_config(search_curves(out_dir, arm, fold_ids, lrs, max_epochs))


# ---------------------------------------------------------------- training


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; the encoder experiments run on the GPU.")
    return torch.device("cuda")


def run_fold(
    arm: EncoderArm, cfg: EncoderConfig, train: pd.DataFrame, y_train: np.ndarray,
    folds: pd.Series, fold: int, tokenizer, classes: list[str], every_epoch: bool,
    predict_batch_size: int = 64,
) -> tuple[pd.DataFrame, list[float], float]:
    """Train on every fold but `fold` (with labels `y_train`) and predict `fold`.

    With `every_epoch`, the validation fold is scored after each epoch (search
    phase); otherwise only after the last one.
    """
    t0 = time.perf_counter()  # monotonic: time.time() can jump under WSL
    is_val = (folds == fold).to_numpy()
    val = train[is_val]
    frames = []

    def score(epoch, model):
        probs = predict_proba(model, tokenizer, val.text.tolist(), cfg.max_length,
                              batch_size=predict_batch_size, device=device())
        frames.append(oof_frame(probs, val, classes, epoch))

    model, history = train_classifier(
        train.text[~is_val].tolist(), y_train[~is_val], len(classes), cfg,
        tokenizer, backbone_factory(arm), device(),
        on_epoch_end=score if every_epoch else None,
        pooling=arm.pooling, text_format=arm.text_format,
    )
    if not every_epoch:
        score(cfg.epochs, model)
    del model
    torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True), history, time.perf_counter() - t0


def train_and_record(
    phase: str, arm_key: str, arm: EncoderArm, base_cfg: EncoderConfig,
    lr: float, epochs: int, seed: int, train: pd.DataFrame, folds: pd.Series,
    classes: list[str], csv_path: Path, out_dir: Path, fold_ids: tuple[int, ...],
    shuffle: bool = False, every_epoch: bool = False, predict_batch_size: int = 64,
) -> None:
    """Folds `fold_ids` of one (phase, arm, lr, epochs, seed); skips folds already recorded.

    Every run uses `base_cfg` (schedule length, batch, max length, ...) with this
    run's lr, epochs and seed.
    """
    done = completed_keys(csv_path)
    y = train.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    tokenizer = load_tokenizer(arm)
    cfg = replace(base_cfg, lr=lr, epochs=epochs, seed=seed)
    for fold in fold_ids:
        key = (phase, arm_key, float(lr), int(epochs), int(seed), int(fold))
        if key in done:
            continue
        y_train = control_train_labels(y, folds, fold, 1000 * seed + fold) if shuffle else y
        oof, history, seconds = run_fold(arm, cfg, train, y_train, folds, fold,
                                         tokenizer, classes, every_epoch, predict_batch_size)
        path = oof_path(out_dir, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(path, index=False)
        by_epoch = [balanced_accuracy(g.model, g.pred) for _, g in oof.groupby("epoch")]
        append_row(csv_path, {
            **dict(zip(RUN_KEY, key)), "balanced_accuracy": by_epoch[-1],
            "epoch_balanced_accuracy": json.dumps([round(v, 4) for v in by_epoch]),
            "n_val": int((oof.epoch == oof.epoch.max()).sum()),
            "initial_loss": history[0], "final_loss": history[-1],
            "loss_history": json.dumps([round(v, 5) for v in history]),
            "stable": run_is_stable(history), "seconds": round(seconds, 1),
        })
        curve = (f" (best epoch {int(np.argmax(by_epoch)) + 1}: {max(by_epoch):.3f})"
                 if every_epoch else "")
        print(f"{phase} {arm_key} lr={lr:g} ep={epochs} seed={seed} fold={fold}: "
              f"bal_acc={by_epoch[-1]:.3f}{curve} loss {history[0]:.2f}->{history[-1]:.3f} "
              f"({seconds:.0f}s)", flush=True)
