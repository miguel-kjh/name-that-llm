"""Baseline with prompt-grouped CV (issue #3, experiment E002).

Usage:
    uv run python scripts/run_baseline.py [--data-dir data] [--out-dir kb/research/data/E002]
                                          [--seeds 0 1 2] [--n-splits 5]

For each fold seed, trains the fixed baseline on 4 folds and predicts the 5th.
Also runs a label-shuffled control to detect leakage. Writes:
    manifest.json  config, data hash, code revision, versions, command
    summary.json   headline metrics per seed, their mean, and the control
    runs.csv       one row per (seed, fold)
    folds.csv      fold of each id per seed
    oof.csv        out-of-fold prediction of each id per seed
    per_class_recall.csv, confusion.csv, length_buckets.csv  (pooled over seeds)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from name_that_llm.baseline import BASELINE_CONFIG, make_baseline
from name_that_llm.metrics import (
    accuracy_by_length,
    confusion,
    per_class_recall,
    summarize,
    within_lab_error_share_chance,
)
from name_that_llm.validation import make_folds, out_of_fold_predictions


def run_seed(train: pd.DataFrame, seed: int, n_splits: int, labels: pd.Series):
    """OOF predictions for one fold seed, plus per-fold metric rows."""
    folds = make_folds(train, n_splits=n_splits, seed=seed)
    preds = out_of_fold_predictions(train.text, labels, folds, make_baseline)
    fold_rows = [
        {"seed": seed, "fold": fold, **summarize(labels[folds == fold], preds[folds == fold])}
        for fold in range(n_splits)
    ]
    return folds, preds, fold_rows


def git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("kb/research/data/E002"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.csv"
    train = pd.read_csv(train_path)
    n_words = train.text.str.split().str.len()
    started = time.time()

    all_folds, all_oof, run_rows, seed_summaries = [], [], [], {}
    for seed in args.seeds:
        t0 = time.time()
        folds, preds, fold_rows = run_seed(train, seed, args.n_splits, train.model)
        run_rows += fold_rows
        seed_summaries[seed] = summarize(train.model, preds)
        all_folds.append(pd.DataFrame({"id": train.id, "seed": seed, "fold": folds}))
        all_oof.append(pd.DataFrame({"id": train.id, "seed": seed, "model": train.model,
                                     "pred": preds, "n_words": n_words}))
        print(f"seed {seed}: {seed_summaries[seed]} ({time.time() - t0:.0f}s)", flush=True)

    # Leakage control: same pipeline and folds, labels permuted across rows.
    shuffled = pd.Series(
        np.random.default_rng(0).permutation(train.model.to_numpy()), index=train.index
    )
    _, control_preds, _ = run_seed(train, args.seeds[0], args.n_splits, shuffled)
    control = summarize(shuffled, control_preds)
    print(f"shuffled-label control: {control}", flush=True)

    runs = pd.DataFrame(run_rows)
    oof = pd.concat(all_oof, ignore_index=True)
    seeds_df = pd.DataFrame(seed_summaries).T
    n_models = train.model.nunique()
    summary = {
        "primary_metric": "within_lab_error_share",
        "per_seed": {str(k): v for k, v in seed_summaries.items()},
        "mean_over_seeds": seeds_df.mean().to_dict(),
        "std_over_seeds": seeds_df.std().to_dict(),
        "fold_balanced_accuracy": {
            "mean": float(runs.balanced_accuracy.mean()),
            "std": float(runs.balanced_accuracy.std()),
            "min": float(runs.balanced_accuracy.min()),
            "max": float(runs.balanced_accuracy.max()),
            "n_folds": len(runs),
        },
        "chance": {
            "balanced_accuracy": 1 / n_models,
            "within_lab_error_share": within_lab_error_share_chance(train.model),
        },
        "shuffled_label_control": control,
    }
    manifest = {
        "experiment": "E002",
        "issue": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).as_posix(), *sys.argv[1:]]),
        "code_revision": git_revision(),
        "data": {"path": train_path.as_posix(), "sha256": sha256(train_path),
                 "rows": len(train), "prompts": int(train.prompt_id.nunique()),
                 "models": n_models},
        "validation": {"scheme": "GroupKFold(shuffle=True)", "group": "prompt_id",
                       "n_splits": args.n_splits, "seeds": args.seeds},
        "model_config": BASELINE_CONFIG,
        "control": {"labels": "permuted with numpy default_rng(0)",
                    "fold_seed": args.seeds[0]},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "runtime_seconds": round(time.time() - started, 1),
    }

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    pd.concat(all_folds).to_csv(args.out_dir / "folds.csv", index=False)
    oof.to_csv(args.out_dir / "oof.csv", index=False)
    per_class_recall(oof.model, oof.pred).to_csv(args.out_dir / "per_class_recall.csv")
    confusion(oof.model, oof.pred).to_csv(args.out_dir / "confusion.csv")
    accuracy_by_length(oof.model, oof.pred, oof.n_words).to_csv(
        args.out_dir / "length_buckets.csv", index=False)

    print(json.dumps(summary["mean_over_seeds"], indent=2))
    print(f"Outputs written to {args.out_dir} in {manifest['runtime_seconds']}s")


if __name__ == "__main__":
    main()
