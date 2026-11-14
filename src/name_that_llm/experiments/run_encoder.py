"""Fine-tuned encoders vs the char TF-IDF baseline (issue #5, experiment E003).

Usage:
    python -m name_that_llm.experiments.run_encoder --phase PHASE [--arms A B]
        [--data-dir data] [--out-dir kb/research/data/E003]
        [--folds-csv kb/research/data/E002/folds.csv]
        [--folds 0 1 2 3 4] [--n-prompts N]   (smoke tests only, need another --out-dir)

Phases, in order:
    preflight  folds match E002, no truncation, both arms memorize 64 passages
    search     fold seed 0, per arm and lr: 10 epochs scored after each one -> search.csv
    eval       chosen (lr, epoch) per arm, fold seeds 1 and 2: trained on the same
               10-epoch schedule and stopped after the chosen epoch -> runs.csv
    control    arm A, chosen config, shuffled training labels  -> runs.csv
    timing     inference time for 4,000 passages               -> timing.json
    summarize  decision rules of H003 and H004                 -> summary.json + tables

Every training run writes its out-of-fold predictions to `oof/` and then a row
to its CSV, so an interrupted phase resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

from name_that_llm.decisions import (
    BASELINE_BALANCED_ACCURACY,
    MIN_RUN_BALANCED_ACCURACY,
    h003_status,
    h004_status,
    pick_config,
    run_is_stable,
)
from name_that_llm.encoder import (
    ARMS,
    EncoderConfig,
    MeanPoolClassifier,
    backbone_factory,
    count_truncated,
    load_tokenizer,
    predict_proba,
    train_classifier,
)
from name_that_llm.experiments.provenance import git_revision, sha256
from name_that_llm.metrics import (
    accuracy_by_length,
    balanced_accuracy,
    confusion,
    per_class_recall,
    summarize,
)
from name_that_llm.validation import make_folds

SEARCH_LRS = (1e-5, 3e-5, 5e-5)
MAX_EPOCHS = 10  # length of the LR schedule; the search scores epochs 1..10
SEARCH_SEED = 0
EVAL_SEEDS = (1, 2)
CONTROL_ARM, CONTROL_SEED = "A", 1
N_SPLITS = 5
MAX_LENGTH = 512

# Guardrails fixed in E003.
CONTROL_MAX_BALANCED_ACCURACY = 0.08
TIMING_N_TEXTS = 4000
TIMING_MAX_SECONDS = 600
PREFLIGHT_N_TEXTS = 64
PREFLIGHT_MIN_TRAIN_ACCURACY = 0.95
PREFLIGHT_CONFIG = {"lr": 5e-5, "epochs": 30}
STOP_IF_SEARCH_BELOW = 0.20

DEFAULT_OUT_DIR = Path("kb/research/data/E003")
ALL_FOLDS = tuple(range(N_SPLITS))

# One training run = one fold. Search runs have epochs = MAX_EPOCHS and store
# predictions for every epoch; eval and control runs store their last epoch.
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
    return {_key(row) for row in done[RUN_KEY].to_dict("records")}


def _key(row: dict) -> tuple:
    return (row["phase"], row["arm"], float(row["lr"]), int(row["epochs"]),
            int(row["seed"]), int(row["fold"]))


def append_row(csv_path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def oof_path(out_dir: Path, key: tuple) -> Path:
    phase, arm, lr, epochs, seed, fold = key
    return out_dir / "oof" / f"{phase}_{arm}_lr{lr:g}_ep{epochs}_s{seed}_f{fold}.parquet"


# ---------------------------------------------------------------- training


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; E003 runs on the GPU.")
    return torch.device("cuda")


def oof_frame(probs: np.ndarray, val: pd.DataFrame, classes: list[str], epoch: int) -> pd.DataFrame:
    oof = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
    oof.insert(0, "id", val.id.to_numpy())
    oof.insert(1, "model", val.model.to_numpy())
    oof.insert(2, "pred", [classes[i] for i in probs.argmax(axis=1)])
    oof.insert(3, "epoch", epoch)
    return oof


def run_fold(
    arm: str, cfg: EncoderConfig, train: pd.DataFrame, y_train: np.ndarray,
    folds: pd.Series, fold: int, tokenizer, classes: list[str], every_epoch: bool,
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
        probs = predict_proba(model, tokenizer, val.text.tolist(), MAX_LENGTH, device=device())
        frames.append(oof_frame(probs, val, classes, epoch))

    model, history = train_classifier(
        train.text[~is_val].tolist(), y_train[~is_val], len(classes), cfg,
        tokenizer, backbone_factory(ARMS[arm]), device(),
        on_epoch_end=score if every_epoch else None,
    )
    if not every_epoch:
        score(cfg.epochs, model)
    del model
    torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True), history, time.perf_counter() - t0


def train_and_record(
    phase: str, arm: str, lr: float, epochs: int, seed: int, train: pd.DataFrame,
    folds: pd.Series, classes: list[str], csv_path: Path, out_dir: Path,
    fold_ids: tuple[int, ...], shuffle: bool = False, every_epoch: bool = False,
) -> None:
    """Folds `fold_ids` of one (phase, arm, lr, epochs, seed); skips folds already recorded.

    Every run uses the MAX_EPOCHS schedule and trains `epochs` epochs of it.
    """
    done = completed_keys(csv_path)
    y = train.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    tokenizer = load_tokenizer(ARMS[arm])
    cfg = EncoderConfig(lr=lr, epochs=epochs, schedule_epochs=MAX_EPOCHS,
                        max_length=MAX_LENGTH, seed=seed)
    for fold in fold_ids:
        key = (phase, arm, float(lr), int(epochs), int(seed), int(fold))
        if key in done:
            continue
        y_train = control_train_labels(y, folds, fold, 1000 * seed + fold) if shuffle else y
        oof, history, seconds = run_fold(arm, cfg, train, y_train, folds, fold,
                                         tokenizer, classes, every_epoch)
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
        print(f"{phase} {arm} lr={lr:g} ep={epochs} seed={seed} fold={fold}: "
              f"bal_acc={by_epoch[-1]:.3f}{curve} loss {history[0]:.2f}->{history[-1]:.3f} "
              f"({seconds:.0f}s)", flush=True)


def read_oof(out_dir: Path, keys: list[tuple]) -> pd.DataFrame:
    """OOF predictions of the given runs, tagged with their fold seed and fold."""
    return pd.concat(
        [pd.read_parquet(oof_path(out_dir, k)).assign(seed=k[4], fold=k[5]) for k in keys],
        ignore_index=True,
    )


def search_curves(out_dir: Path, arm: str, fold_ids: tuple[int, ...]) -> pd.DataFrame:
    """Pooled OOF balanced accuracy of seed 0 per (lr, epoch) for one arm."""
    search = pd.read_csv(out_dir / "search.csv")
    rows = []
    for lr in SEARCH_LRS:
        runs = search[(search.arm == arm) & np.isclose(search.lr, lr)
                      & search.fold.isin(fold_ids)]
        if sorted(runs.fold) != sorted(fold_ids):
            raise SystemExit(f"Search for arm {arm} lr={lr:g} is incomplete.")
        oof = read_oof(out_dir, [_key(r) for r in runs[RUN_KEY].to_dict("records")])
        for epoch in range(1, MAX_EPOCHS + 1):
            at = oof[oof.epoch == epoch]
            rows.append({"arm": arm, "lr": lr, "epochs": epoch,
                         "balanced_accuracy": balanced_accuracy(at.model, at.pred)})
    return pd.DataFrame(rows)


def chosen_configs(out_dir: Path, arms: list[str], fold_ids: tuple[int, ...]) -> dict[str, dict]:
    """Search winner (lr, epoch) per arm."""
    return {arm: pick_config(search_curves(out_dir, arm, fold_ids)) for arm in arms}


# ---------------------------------------------------------------- phases


def phase_preflight(train, folds_csv, classes, args) -> None:
    report = {"cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    report["folds_match_e002"] = {
        str(seed): bool(load_folds(train, folds_csv, seed).equals(
            make_folds(train, n_splits=N_SPLITS, seed=seed)))
        for seed in (SEARCH_SEED, *EVAL_SEEDS)
    }
    sample = train.sample(PREFLIGHT_N_TEXTS, random_state=0)
    y = sample.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    report["arms"] = {}
    for arm in args.arms:
        tokenizer = load_tokenizer(ARMS[arm])
        weights_dtype = str(next(backbone_factory(ARMS[arm])().parameters()).dtype)
        cfg = EncoderConfig(**PREFLIGHT_CONFIG, max_length=MAX_LENGTH, seed=0)
        model, history = train_classifier(sample.text.tolist(), y, len(classes), cfg,
                                          tokenizer, backbone_factory(ARMS[arm]), device())
        preds = predict_proba(model, tokenizer, sample.text.tolist(), device=device()).argmax(1)
        del model
        torch.cuda.empty_cache()
        report["arms"][arm] = {
            "model": ARMS[arm].name, "revision": ARMS[arm].revision,
            "weights_dtype": weights_dtype,
            "n_truncated": count_truncated(tokenizer, train.text.tolist(), MAX_LENGTH),
            "max_tokens": max(len(t) for t in tokenizer(train.text.tolist())["input_ids"]),
            "memorize_train_accuracy": float((preds == y).mean()),
            "memorize_loss": [history[0], history[-1]],
        }
    report["passed"] = bool(
        report["cuda"]
        and all(report["folds_match_e002"].values())
        and all(a["n_truncated"] == 0 for a in report["arms"].values())
        and all(a["weights_dtype"] == "torch.float32" for a in report["arms"].values())
        and all(a["memorize_train_accuracy"] >= PREFLIGHT_MIN_TRAIN_ACCURACY
                for a in report["arms"].values())
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def phase_search(train, folds_csv, classes, args) -> None:
    folds = load_folds(train, folds_csv, SEARCH_SEED)
    for arm, lr in product(args.arms, SEARCH_LRS):
        train_and_record("search", arm, lr, MAX_EPOCHS, SEARCH_SEED, train, folds, classes,
                         args.out_dir / "search.csv", args.out_dir, args.folds,
                         every_epoch=True)
    pd.concat([search_curves(args.out_dir, arm, args.folds) for arm in args.arms]).to_csv(
        args.out_dir / "search_curves.csv", index=False)
    chosen = chosen_configs(args.out_dir, args.arms, args.folds)
    for arm, choice in chosen.items():
        print(f"arm {arm}: chosen lr={choice['lr']:g} epochs={choice['epochs']} "
              f"(seed 0 OOF balanced accuracy {choice['balanced_accuracy']:.3f})")
    if not passes_stop_rule(chosen):
        print(f"STOP RULE: best search score < {STOP_IF_SEARCH_BELOW}. "
              "Review the implementation before running the eval phase.")


def passes_stop_rule(chosen: dict[str, dict]) -> bool:
    return max(c["balanced_accuracy"] for c in chosen.values()) >= STOP_IF_SEARCH_BELOW


def phase_eval(train, folds_csv, classes, args) -> None:
    chosen = chosen_configs(args.out_dir, args.arms, args.folds)
    if not passes_stop_rule(chosen) and not args.ignore_stop_rule:
        raise SystemExit(f"Stop rule: the best search score is below {STOP_IF_SEARCH_BELOW}. "
                         "Review the implementation; eval phase not run.")
    for arm, seed in product(args.arms, EVAL_SEEDS):
        c = chosen[arm]
        train_and_record("eval", arm, c["lr"], c["epochs"], seed, train,
                         load_folds(train, folds_csv, seed), classes,
                         args.out_dir / "runs.csv", args.out_dir, args.folds)


def phase_control(train, folds_csv, classes, args) -> None:
    c = chosen_configs(args.out_dir, [CONTROL_ARM], args.folds)[CONTROL_ARM]
    train_and_record("control", CONTROL_ARM, c["lr"], c["epochs"], CONTROL_SEED, train,
                     load_folds(train, folds_csv, CONTROL_SEED), classes,
                     args.out_dir / "runs.csv", args.out_dir, args.folds, shuffle=True)


def phase_timing(train, folds_csv, classes, args) -> None:
    """Load + predict 4,000 passages. Head weights do not change the cost."""
    texts = (train.text.tolist() * (TIMING_N_TEXTS // len(train) + 1))[:TIMING_N_TEXTS]
    report = {}
    for arm in args.arms:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokenizer = load_tokenizer(ARMS[arm])
        model = MeanPoolClassifier(backbone_factory(ARMS[arm])(), len(classes)).to(device())
        predict_proba(model, tokenizer, texts, MAX_LENGTH, device=device())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        report[arm] = {"n_texts": len(texts), "seconds": round(seconds, 1),
                       "passed": seconds < TIMING_MAX_SECONDS}
        del model
        torch.cuda.empty_cache()
    (args.out_dir / "timing.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def phase_summarize(train, folds_csv, classes, args) -> None:
    out = args.out_dir
    arms = args.arms
    preflight = json.loads((out / "preflight.json").read_text())
    timing = json.loads((out / "timing.json").read_text())
    runs = pd.read_csv(out / "runs.csv")
    chosen = chosen_configs(out, arms, args.folds)
    baseline_runs = pd.read_csv(args.folds_csv.parent / "runs.csv")
    n_words = train.set_index("id").text.str.split().str.len()

    ev = runs[runs.phase == "eval"]
    per_arm, fold_scores, pooled = {}, {}, {}
    for arm in arms:
        arm_runs = ev[ev.arm == arm].sort_values(["seed", "fold"])
        if len(arm_runs) < len(EVAL_SEEDS) * len(args.folds):
            raise SystemExit(f"Eval phase for arm {arm} is incomplete.")
        oof = read_oof(out, [_key(r) for r in arm_runs[RUN_KEY].to_dict("records")])
        pooled[arm] = oof
        per_seed = {str(s): summarize(g.model, g.pred) for s, g in oof.groupby("seed")}
        fold_scores[arm] = arm_runs.balanced_accuracy.to_numpy()
        merged = arm_runs.merge(baseline_runs[["seed", "fold", "balanced_accuracy"]],
                                on=["seed", "fold"], suffixes=("", "_baseline"))
        per_arm[arm] = {
            "config": {k: chosen[arm][k] for k in ("lr", "epochs")},
            "search_balanced_accuracy_seed0": chosen[arm]["balanced_accuracy"],
            "per_seed": per_seed,
            "mean_over_seeds": pd.DataFrame(per_seed).T.mean().to_dict(),
            "fold_balanced_accuracy": arm_runs.balanced_accuracy.round(4).tolist(),
            "fold_delta_vs_baseline": (merged.balanced_accuracy
                                       - merged.balanced_accuracy_baseline).round(4).tolist(),
        }

    best_arm = max(arms, key=lambda a: per_arm[a]["mean_over_seeds"]["balanced_accuracy"])
    best_mean = per_arm[best_arm]["mean_over_seeds"]["balanced_accuracy"]

    control_runs = runs[runs.phase == "control"]
    control_oof = read_oof(out, [_key(r) for r in control_runs[RUN_KEY].to_dict("records")])
    control_bal_acc = balanced_accuracy(control_oof.model, control_oof.pred)

    # Stability is judged on the decision runs. Search runs record it for information:
    # an undertrained grid point (e.g. lr 1e-5, 4 epochs) is simply not chosen.
    loss_ok = all(run_is_stable(json.loads(h)) for h in ev.loss_history)
    eval_floor_ok = bool((ev.balanced_accuracy >= MIN_RUN_BALANCED_ACCURACY).all())
    guardrails = {
        "preflight_passed": preflight["passed"],
        "no_truncation": all(a["n_truncated"] == 0 for a in preflight["arms"].values()),
        "control_balanced_accuracy": control_bal_acc,
        "control_passed": control_bal_acc <= CONTROL_MAX_BALANCED_ACCURACY,
        "all_runs_loss_halved": loss_ok,
        "eval_runs_above_floor": eval_floor_ok,
        "timing_passed": all(t["passed"] for t in timing.values()),
    }
    method_valid = all(v for k, v in guardrails.items() if isinstance(v, bool))
    stable = loss_ok and eval_floor_ok

    h003 = h003_status(best_mean, all_stable=stable and method_valid)
    h004 = (h004_status(fold_scores["A"], fold_scores["B"], all_stable=stable and method_valid)
            if {"A", "B"} <= set(arms) else None)

    summary = {
        "primary_metric": "balanced_accuracy (pooled OOF, mean of fold seeds 1 and 2)",
        "baseline": {"name": "E002 char TF-IDF + LR",
                     "balanced_accuracy": BASELINE_BALANCED_ACCURACY,
                     "seeds_1_2": float(baseline_runs[baseline_runs.seed.isin(EVAL_SEEDS)]
                                        .groupby("seed").balanced_accuracy.mean().mean())},
        "arms": per_arm,
        "best_arm": best_arm,
        "delta_best_vs_baseline": best_mean - BASELINE_BALANCED_ACCURACY,
        "guardrails": guardrails,
        "method_valid": method_valid,
        "H003": {"status": h003, "best_mean": best_mean},
        "H004": h004,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    recalls, buckets = [], []
    for arm, oof in pooled.items():
        recalls.append(per_class_recall(oof.model, oof.pred).rename(arm))
        b = accuracy_by_length(oof.model, oof.pred, oof.id.map(n_words))
        buckets.append(b.assign(arm=arm))
        confusion(oof.model, oof.pred).to_csv(out / f"confusion_{arm}.csv")
    pd.concat(recalls, axis=1).to_csv(out / "per_class_recall.csv")
    pd.concat(buckets).to_csv(out / "length_buckets.csv", index=False)
    print(json.dumps({k: summary[k] for k in ("best_arm", "H003", "H004", "guardrails")},
                     indent=2, default=float))


PHASES = {"preflight": phase_preflight, "search": phase_search, "eval": phase_eval,
          "control": phase_control, "timing": phase_timing, "summarize": phase_summarize}


def update_manifest(args, train_path: Path, train: pd.DataFrame, seconds: float) -> None:
    path = args.out_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update({
        "experiment": "E003",
        "hypotheses": ["H003", "H004"],
        "issue": 5,
        "data": {"path": train_path.as_posix(), "sha256": sha256(train_path),
                 "rows": len(train), "prompts": int(train.prompt_id.nunique()),
                 "models": int(train.model.nunique())},
        "validation": {"folds_csv": args.folds_csv.as_posix(), "n_splits": N_SPLITS,
                       "search_seed": SEARCH_SEED, "eval_seeds": list(EVAL_SEEDS),
                       "control": {"arm": CONTROL_ARM, "seed": CONTROL_SEED}},
        "arms": {k: {"model": v.name, "revision": v.revision} for k, v in ARMS.items()},
        "recipe": EncoderConfig(lr=0.0, epochs=0, max_length=MAX_LENGTH).to_dict()
                  | {"pooling": "masked mean", "head": "dropout + linear",
                     "search_lrs": list(SEARCH_LRS), "max_epochs": MAX_EPOCHS,
                     "schedule": "linear warmup 10% + linear decay over max_epochs; "
                                 "eval and control stop after the chosen epoch"},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                     else None},
    })
    manifest.setdefault("phases", {})[args.phase] = {
        "command": " ".join(["python -m", __spec__.name if __spec__ else "run_encoder",
                             *sys.argv[1:]]),
        "code_revision": git_revision(),
        "folds": list(args.folds), "n_prompts": args.n_prompts,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(seconds, 1),
    }
    path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("kb/research/data/E003"))
    parser.add_argument("--folds-csv", type=Path,
                        default=Path("kb/research/data/E002/folds.csv"))
    parser.add_argument("--folds", type=int, nargs="+", choices=ALL_FOLDS, default=ALL_FOLDS,
                        help="smoke tests only: run a subset of folds")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="smoke tests only: keep the first N prompts")
    parser.add_argument("--ignore-stop-rule", action="store_true",
                        help="smoke tests only: run eval even if the search is below the floor")
    args = parser.parse_args()
    args.folds = tuple(sorted(set(args.folds)))
    is_smoke = args.folds != ALL_FOLDS or args.n_prompts is not None or args.ignore_stop_rule
    if is_smoke and args.out_dir.resolve() == DEFAULT_OUT_DIR.resolve():
        parser.error("--folds/--n-prompts/--ignore-stop-rule are for smoke tests: "
                     "pass a different --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.csv"
    train = pd.read_csv(train_path)
    if args.n_prompts is not None:
        keep = sorted(train.prompt_id.unique())[:args.n_prompts]
        train = train[train.prompt_id.isin(keep)].reset_index(drop=True)
    folds_csv = pd.read_csv(args.folds_csv)
    classes = sorted(train.model.unique())

    started = time.perf_counter()
    PHASES[args.phase](train, folds_csv, classes, args)
    update_manifest(args, train_path, train, time.perf_counter() - started)


if __name__ == "__main__":
    main()
