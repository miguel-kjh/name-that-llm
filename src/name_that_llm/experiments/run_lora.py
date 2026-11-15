"""LoRA-tuned embedding encoders vs gte and the baseline (issue #7, experiment E004).

Usage:
    python -m name_that_llm.experiments.run_lora --phase PHASE [--arms G Q0 Q4 Q8]
        [--data-dir data] [--out-dir kb/research/data/E004]
        [--folds-csv kb/research/data/E002/folds.csv]
        [--folds 0 1 2 3 4] [--n-prompts N]   (smoke tests only, need another --out-dir)

Phases, in order:
    preflight  folds match E002, no truncation (prompt included), Qwen ends in
               <|endoftext|> exactly once,
               bf16 base weights, trainable parameters, every arm memorizes 64 passages
    pilot      fold 0 of seed 0, one epoch per arm: time and peak memory, and the
               projected hours of the whole experiment             -> pilot.json
    search     fold seed 0, 6-epoch schedule scored after each epoch -> search.csv
               G and Q0 search lr {1e-4, 3e-4}; Q4 and Q8 reuse Q0's lr and only
               pick the stopping epoch (reduced protocol, agreed with the user)
    eval       chosen (lr, epoch) per arm, fold seeds 1 and 2      -> runs.csv
    control    arm Q0, chosen config, shuffled training labels     -> runs.csv
    timing     load + inference time for 4,000 passages            -> timing.json
    summarize  decision rule of H005 and secondary comparisons     -> summary.json + tables

Every training run writes its out-of-fold predictions to `oof/` and then a row
to its CSV, so an interrupted phase resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import peft
import sklearn
import torch
import transformers

from name_that_llm.decisions import (
    GTE_BALANCED_ACCURACY,
    MIN_RUN_BALANCED_ACCURACY,
    h005_status,
    run_is_stable,
)
from name_that_llm.encoder import (
    EncoderArm,
    EncoderConfig,
    LoraSpec,
    PooledClassifier,
    TextFormat,
    backbone_factory,
    count_truncated,
    end_token_id,
    load_tokenizer,
    predict_proba,
    tokenize,
    train_classifier,
)
from name_that_llm.experiments import runs
from name_that_llm.experiments.provenance import git_revision, sha256
from name_that_llm.experiments.runs import (
    RUN_KEY,
    device,
    load_folds,
    read_oof,
    run_key,
)
from name_that_llm.metrics import (
    accuracy_by_length,
    balanced_accuracy,
    confusion,
    per_class_recall,
    summarize,
)
from name_that_llm.validation import make_folds

# ---------------------------------------------------------------- E004 design

LORA = LoraSpec(r=16, alpha=32, dropout=0.05, target_modules="all-linear")
GEMMA_FORMAT = TextFormat(prompt="task: classification | query: ")
QWEN_FORMAT = TextFormat(
    prompt="Instruct: Identify which language model wrote this passage\nQuery:",
    end_token="<|endoftext|>")

# Every arm: LoRA on a frozen bf16 base, native pooling, L2-normalized embedding
# (as both models' sentence-transformers pipelines do), native task prompt.
_COMMON = {"normalize": True, "dtype": torch.bfloat16, "lora": LORA}
ARMS = {
    "G": EncoderArm("google/embeddinggemma-2", "f9972ac22e696a105c482202646979e33e4b5a1e",
                    pooling="mean", text_format=GEMMA_FORMAT, **_COMMON,
                    load_kwargs={"vision_config": None, "audio_config": None}),
    "Q0": EncoderArm("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                     pooling="last", text_format=QWEN_FORMAT, **_COMMON),
    "Q4": EncoderArm("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b",
                     pooling="last", text_format=QWEN_FORMAT, **_COMMON,
                     gradient_checkpointing=True),
    "Q8": EncoderArm("Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
                     pooling="last", text_format=QWEN_FORMAT, **_COMMON,
                     gradient_checkpointing=True),
}
ARM_ORDER = ("G", "Q0", "Q4", "Q8")  # Q4 and Q8 need Q0's search first
LARGE_ARMS = ("Q4", "Q8")  # they decide H005
LR_SEARCH_ARMS = ("G", "Q0")
LR_SOURCE = {"Q4": "Q0", "Q8": "Q0"}  # reduced protocol: inherit Q0's chosen lr

# Effective batch is always 16; these only split it to fit in memory (set by the pilot).
MICRO_BATCH = {"G": 16, "Q0": 16, "Q4": 16, "Q8": 4}
PREDICT_BATCH = {"G": 64, "Q0": 64, "Q4": 32, "Q8": 16}

SEARCH_LRS = (1e-4, 3e-4)
MAX_EPOCHS = 6  # length of the LR schedule; the search scores epochs 1..6
SEARCH_SEED = 0
EVAL_SEEDS = (1, 2)
CONTROL_ARM, CONTROL_SEED = "Q0", 1
N_SPLITS = 5
MAX_LENGTH = 512

# Guardrails and budget fixed in E004.
CONTROL_MAX_BALANCED_ACCURACY = 0.08
TIMING_N_TEXTS = 4000
TIMING_MAX_SECONDS = 600
KAGGLE_SLOWDOWN = 10  # conservative 4090 -> Kaggle T4 factor, for information only
KAGGLE_LIMIT_SECONDS = 9 * 3600
PREFLIGHT_N_TEXTS = 64
PREFLIGHT_MIN_TRAIN_ACCURACY = 0.95
PREFLIGHT_CONFIG = {"lr": 3e-4, "epochs": 30}
STOP_IF_SEARCH_BELOW = 0.20  # on Q0
ESTIMATED_HOURS = 12.0
MAX_OVERRUN = 3.0

DEFAULT_OUT_DIR = Path("kb/research/data/E004")
ALL_FOLDS = tuple(range(N_SPLITS))


def base_config(arm: str) -> EncoderConfig:
    return EncoderConfig(lr=0.0, epochs=1, schedule_epochs=MAX_EPOCHS, batch_size=16,
                         micro_batch_size=MICRO_BATCH[arm], max_length=MAX_LENGTH)


# ---------------------------------------------------------------- pure helpers


def ordered(arms) -> list[str]:
    return [a for a in ARM_ORDER if a in set(arms)]


def search_lrs(arm: str, out_dir: Path, fold_ids: tuple[int, ...]) -> tuple[float, ...]:
    """The lr grid of `arm`: searched for G and Q0, inherited from Q0 for Q4 and Q8.

    Raises SystemExit if the source arm's search is incomplete.
    """
    if arm in LR_SEARCH_ARMS:
        return SEARCH_LRS
    source = LR_SOURCE[arm]
    return (chosen_config(out_dir, source, fold_ids)["lr"],)


def chosen_config(out_dir: Path, arm: str, fold_ids: tuple[int, ...]) -> dict:
    """Search winner (lr, epoch) of one arm."""
    return runs.best_config(out_dir, arm, fold_ids, search_lrs(arm, out_dir, fold_ids),
                            MAX_EPOCHS)


def chosen_configs(out_dir: Path, arms, fold_ids: tuple[int, ...]) -> dict[str, dict]:
    return {arm: chosen_config(out_dir, arm, fold_ids) for arm in ordered(arms)}


def passes_stop_rule(q0_choice: dict) -> bool:
    return q0_choice["balanced_accuracy"] >= STOP_IF_SEARCH_BELOW


def projected_hours(epoch_seconds: dict[str, float], val_seconds: dict[str, float],
                    load_seconds: dict[str, float]) -> dict:
    """Projected GPU hours of the whole experiment from one measured epoch per arm.

    Worst case for the decision runs (k* = MAX_EPOCHS). Search runs score the
    validation fold after every epoch; eval and control runs score it once.
    """
    per_arm = {}
    for arm, epoch_s in epoch_seconds.items():
        n_lrs = len(SEARCH_LRS) if arm in LR_SEARCH_ARMS else 1
        search_runs = n_lrs * N_SPLITS
        eval_runs = len(EVAL_SEEDS) * N_SPLITS + (N_SPLITS if arm == CONTROL_ARM else 0)
        seconds = (search_runs * (MAX_EPOCHS * (epoch_s + val_seconds[arm]) + load_seconds[arm])
                   + eval_runs * (MAX_EPOCHS * epoch_s + val_seconds[arm] + load_seconds[arm]))
        per_arm[arm] = seconds / 3600
    total = sum(per_arm.values())
    return {"per_arm_hours": per_arm, "total_hours": total,
            "estimated_hours": ESTIMATED_HOURS,
            "within_budget": total <= MAX_OVERRUN * ESTIMATED_HOURS}


def fold_scores(runs_df: pd.DataFrame, phase: str, arm: str) -> pd.DataFrame:
    """Balanced accuracy per (seed, fold) of one arm in one phase."""
    return (runs_df[(runs_df.phase == phase) & (runs_df.arm == arm)]
            .sort_values(["seed", "fold"])[["seed", "fold", "balanced_accuracy"]]
            .reset_index(drop=True))


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Mean difference a - b and wins over the folds both have."""
    m = a.merge(b, on=["seed", "fold"], suffixes=("_a", "_b"))
    diff = m.balanced_accuracy_a - m.balanced_accuracy_b
    return {"mean_diff": float(diff.mean()), "wins_a": int((diff > 0).sum()),
            "wins_b": int((diff < 0).sum()), "n_folds": int(len(m)),
            "fold_diff": diff.round(4).tolist()}


# ---------------------------------------------------------------- phases


def _arm_report(arm: str, tokenizer, texts: list[str], backbone) -> dict:
    spec = ARMS[arm]
    ids = tokenize(tokenizer, texts, MAX_LENGTH * 4, spec.text_format)  # untruncated lengths
    frozen = [p for p in backbone.parameters() if not p.requires_grad]
    report = {
        "model": spec.name, "revision": spec.revision,
        "base_weights_dtype": str(frozen[0].dtype) if frozen else None,
        "trainable_parameters": int(sum(p.numel() for p in backbone.parameters()
                                        if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in backbone.parameters())),
        "n_truncated": count_truncated(tokenizer, texts, MAX_LENGTH, spec.text_format),
        "max_tokens": max(len(x) for x in ids),
    }
    if spec.text_format.end_token is not None:
        end = end_token_id(tokenizer, spec.text_format)
        formatted = tokenize(tokenizer, texts, MAX_LENGTH, spec.text_format)
        report["end_token"] = spec.text_format.end_token
        report["ends_with_end_token"] = all(x[-1] == end for x in formatted)
        report["end_token_appears_once"] = all(x.count(end) == 1 for x in formatted)
    return report


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
    for arm in ordered(args.arms):
        spec = ARMS[arm]
        tokenizer = load_tokenizer(spec)
        backbone = backbone_factory(spec)()
        arm_report = _arm_report(arm, tokenizer, train.text.tolist(), backbone)
        del backbone
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = replace(base_config(arm), **PREFLIGHT_CONFIG, schedule_epochs=None, seed=0)
        model, history = train_classifier(sample.text.tolist(), y, len(classes), cfg, tokenizer,
                                          backbone_factory(spec), device(),
                                          pooling=spec.pooling, text_format=spec.text_format,
                                          normalize=spec.normalize)
        preds = predict_proba(model, tokenizer, sample.text.tolist(), MAX_LENGTH,
                              batch_size=PREDICT_BATCH[arm], device=device()).argmax(1)
        del model
        torch.cuda.empty_cache()
        report["arms"][arm] = arm_report | {
            "memorize_train_accuracy": float((preds == y).mean()),
            "memorize_loss": [history[0], history[-1]],
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        print(arm, json.dumps(report["arms"][arm]), flush=True)
    arms = report["arms"].values()
    report["passed"] = bool(
        report["cuda"]
        and all(report["folds_match_e002"].values())
        and all(a["n_truncated"] == 0 for a in arms)
        and all(a.get("ends_with_end_token", True) and a.get("end_token_appears_once", True)
                for a in arms)
        and all(a["base_weights_dtype"] == "torch.bfloat16" for a in arms)
        and all(a["memorize_train_accuracy"] >= PREFLIGHT_MIN_TRAIN_ACCURACY for a in arms)
    )
    (args.out_dir / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def phase_pilot(train, folds_csv, classes, args) -> None:
    """One epoch on fold 0 of the search seed per arm. Not a result: timing only."""
    folds = load_folds(train, folds_csv, SEARCH_SEED)
    is_val = (folds == args.folds[0]).to_numpy()
    y = train.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    report = {"arms": {}}
    for arm in ordered(args.arms):
        spec = ARMS[arm]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tokenizer = load_tokenizer(spec)
        make = backbone_factory(spec)
        loaded = {}

        def timed_make():
            t = time.perf_counter()
            backbone = make()
            loaded["seconds"] = time.perf_counter() - t
            return backbone

        cfg = replace(base_config(arm), lr=SEARCH_LRS[0], epochs=1, seed=SEARCH_SEED)
        model, history = train_classifier(train.text[~is_val].tolist(), y[~is_val],
                                          len(classes), cfg, tokenizer, timed_make, device(),
                                          pooling=spec.pooling, text_format=spec.text_format,
                                          normalize=spec.normalize)
        torch.cuda.synchronize()
        train_s = time.perf_counter() - t0 - loaded["seconds"]
        t1 = time.perf_counter()
        predict_proba(model, tokenizer, train.text[is_val].tolist(), MAX_LENGTH,
                      batch_size=PREDICT_BATCH[arm], device=device())
        torch.cuda.synchronize()
        val_s = time.perf_counter() - t1
        del model
        report["arms"][arm] = {
            "load_seconds": round(loaded["seconds"], 1), "epoch_seconds": round(train_s, 1),
            "val_seconds": round(val_s, 1), "loss": history,
            "micro_batch_size": MICRO_BATCH[arm],
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        print(arm, json.dumps(report["arms"][arm]), flush=True)
    a = report["arms"]
    report["projection"] = projected_hours({k: v["epoch_seconds"] for k, v in a.items()},
                                           {k: v["val_seconds"] for k, v in a.items()},
                                           {k: v["load_seconds"] for k, v in a.items()})
    (args.out_dir / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["projection"], indent=2))
    if not report["projection"]["within_budget"]:
        print(f"BUDGET: projection above {MAX_OVERRUN}x the {ESTIMATED_HOURS} h estimate. "
              "Stop and re-plan with the user before the search.")


def phase_search(train, folds_csv, classes, args) -> None:
    folds = load_folds(train, folds_csv, SEARCH_SEED)
    curves = []
    for arm in ordered(args.arms):
        lrs = search_lrs(arm, args.out_dir, args.folds)  # Q4/Q8: refuses without Q0
        for lr in lrs:
            runs.train_and_record("search", arm, ARMS[arm], base_config(arm), lr, MAX_EPOCHS,
                                  SEARCH_SEED, train, folds, classes,
                                  args.out_dir / "search.csv", args.out_dir, args.folds,
                                  every_epoch=True, predict_batch_size=PREDICT_BATCH[arm])
        curves.append(runs.search_curves(args.out_dir, arm, args.folds, lrs, MAX_EPOCHS))
        choice = chosen_config(args.out_dir, arm, args.folds)
        print(f"arm {arm}: chosen lr={choice['lr']:g} epochs={choice['epochs']} "
              f"(seed 0 OOF balanced accuracy {choice['balanced_accuracy']:.3f})", flush=True)
        if arm == "Q0" and not passes_stop_rule(choice):
            print(f"STOP RULE: Q0 search score < {STOP_IF_SEARCH_BELOW}. "
                  "Review the implementation before going on.")
            break
    path = args.out_dir / "search_curves.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.concat(curves)
    if not old.empty:
        old = old[~old.arm.isin(new.arm)]
    pd.concat([old, new]).to_csv(path, index=False)


def phase_eval(train, folds_csv, classes, args) -> None:
    chosen = chosen_configs(args.out_dir, args.arms, args.folds)
    if "Q0" in chosen and not passes_stop_rule(chosen["Q0"]) and not args.ignore_stop_rule:
        raise SystemExit(f"Stop rule: Q0's search score is below {STOP_IF_SEARCH_BELOW}.")
    for arm, seed in product(ordered(args.arms), EVAL_SEEDS):
        c = chosen[arm]
        runs.train_and_record("eval", arm, ARMS[arm], base_config(arm), c["lr"], c["epochs"],
                              seed, train, load_folds(train, folds_csv, seed), classes,
                              args.out_dir / "runs.csv", args.out_dir, args.folds,
                              predict_batch_size=PREDICT_BATCH[arm])


def phase_control(train, folds_csv, classes, args) -> None:
    c = chosen_config(args.out_dir, CONTROL_ARM, args.folds)
    runs.train_and_record("control", CONTROL_ARM, ARMS[CONTROL_ARM], base_config(CONTROL_ARM),
                          c["lr"], c["epochs"], CONTROL_SEED, train,
                          load_folds(train, folds_csv, CONTROL_SEED), classes,
                          args.out_dir / "runs.csv", args.out_dir, args.folds, shuffle=True,
                          predict_batch_size=PREDICT_BATCH[CONTROL_ARM])


def phase_timing(train, folds_csv, classes, args) -> None:
    """Load base + LoRA + head and predict 4,000 passages. Weights do not change the cost."""
    texts = (train.text.tolist() * (TIMING_N_TEXTS // len(train) + 1))[:TIMING_N_TEXTS]
    report = {}
    for arm in ordered(args.arms):
        spec = ARMS[arm]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokenizer = load_tokenizer(spec)
        model = PooledClassifier(backbone_factory(spec)(), len(classes), pooling=spec.pooling,
                                 text_format=spec.text_format,
                                 normalize=spec.normalize).to(device())
        predict_proba(model, tokenizer, texts, MAX_LENGTH, batch_size=PREDICT_BATCH[arm],
                      device=device())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        report[arm] = {"n_texts": len(texts), "seconds": round(seconds, 1),
                       "passed": seconds < TIMING_MAX_SECONDS,
                       "kaggle_estimate_seconds": round(seconds * KAGGLE_SLOWDOWN, 1),
                       "fits_kaggle": seconds * KAGGLE_SLOWDOWN < KAGGLE_LIMIT_SECONDS / 2}
        del model
        torch.cuda.empty_cache()
    (args.out_dir / "timing.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# The inference guardrail is per arm (E004 design): an arm that fails it can still
# confirm or reject H005, but it is not a submission candidate.
PER_ARM_GUARDRAILS = ("timing_passed",)


def is_method_valid(guardrails: dict) -> bool:
    """Every experiment-wide guardrail holds; per-arm ones only gate submission."""
    return all(v for k, v in guardrails.items()
               if isinstance(v, bool) and k not in PER_ARM_GUARDRAILS)


def submission_candidates(timing: dict) -> list[str]:
    """Arms fast enough on the 4090 and projected to fit the Kaggle limit."""
    return [a for a in ordered(timing) if timing[a]["passed"] and timing[a]["fits_kaggle"]]


def phase_summarize(train, folds_csv, classes, args) -> None:
    out = args.out_dir
    arms = ordered(args.arms)
    preflight = json.loads((out / "preflight.json").read_text())
    timing = json.loads((out / "timing.json").read_text())
    runs_df = pd.read_csv(out / "runs.csv")
    chosen = chosen_configs(out, arms, args.folds)
    baseline_runs = pd.read_csv(args.folds_csv.parent / "runs.csv")
    baseline = baseline_runs[baseline_runs.seed.isin(EVAL_SEEDS)][
        ["seed", "fold", "balanced_accuracy"]]
    gte = fold_scores(pd.read_csv(args.gte_runs), "eval", "B")
    n_words = train.set_index("id").text.str.split().str.len()

    ev = runs_df[runs_df.phase == "eval"]
    per_arm, pooled, scores = {}, {}, {}
    for arm in arms:
        arm_runs = ev[ev.arm == arm].sort_values(["seed", "fold"])
        if len(arm_runs) < len(EVAL_SEEDS) * len(args.folds):
            raise SystemExit(f"Eval phase for arm {arm} is incomplete.")
        oof = read_oof(out, [run_key(r) for r in arm_runs[RUN_KEY].to_dict("records")])
        pooled[arm] = oof
        scores[arm] = fold_scores(runs_df, "eval", arm)
        per_seed = {str(s): summarize(g.model, g.pred) for s, g in oof.groupby("seed")}
        per_arm[arm] = {
            "model": ARMS[arm].name,
            "config": {k: chosen[arm][k] for k in ("lr", "epochs")},
            "search_balanced_accuracy_seed0": chosen[arm]["balanced_accuracy"],
            "per_seed": per_seed,
            "mean_over_seeds": pd.DataFrame(per_seed).T.mean().to_dict(),
            "fold_balanced_accuracy": arm_runs.balanced_accuracy.round(4).tolist(),
            "vs_baseline": paired(scores[arm], baseline),
            "vs_gte": paired(scores[arm], gte),
        }

    mean = {a: per_arm[a]["mean_over_seeds"]["balanced_accuracy"] for a in arms}
    large = [a for a in LARGE_ARMS if a in arms]
    best_large = max(large, key=mean.get) if large else None

    control_runs = runs_df[runs_df.phase == "control"]
    control_oof = read_oof(out, [run_key(r) for r in control_runs[RUN_KEY].to_dict("records")])
    control_bal_acc = balanced_accuracy(control_oof.model, control_oof.pred)

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
    method_valid = is_method_valid(guardrails)
    stable = loss_ok and eval_floor_ok

    scaling = {}
    for a, b in (("Q4", "Q0"), ("Q8", "Q4"), ("Q8", "Q0")):
        if a in scores and b in scores:
            scaling[f"{a}_minus_{b}"] = paired(scores[a], scores[b])

    summary = {
        "primary_metric": "balanced_accuracy (pooled OOF, mean of fold seeds 1 and 2)",
        "baseline": {"name": "E002 char TF-IDF + LR",
                     "seeds_1_2": float(baseline.groupby("seed").balanced_accuracy.mean().mean())},
        "gte": {"name": "E003 arm B gte-modernbert-base", "seeds_1_2": GTE_BALANCED_ACCURACY},
        "arms": per_arm,
        "mean_balanced_accuracy": mean,
        "scaling": scaling,
        "best_large_arm": best_large,
        "guardrails": guardrails,
        "kaggle_fit": {a: timing[a]["fits_kaggle"] for a in timing},
        "submission_candidates": submission_candidates(timing),
        "method_valid": method_valid,
        "H005": ({"status": h005_status(mean[best_large], all_stable=stable and method_valid),
                  "best_large_mean": mean[best_large]} if best_large else None),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    recalls, buckets = [], []
    for arm, oof in pooled.items():
        recalls.append(per_class_recall(oof.model, oof.pred).rename(arm))
        buckets.append(accuracy_by_length(oof.model, oof.pred, oof.id.map(n_words))
                       .assign(arm=arm))
        confusion(oof.model, oof.pred).to_csv(out / f"confusion_{arm}.csv")
    pd.concat(recalls, axis=1).to_csv(out / "per_class_recall.csv")
    pd.concat(buckets).to_csv(out / "length_buckets.csv", index=False)
    print(json.dumps({k: summary[k] for k in ("mean_balanced_accuracy", "best_large_arm",
                                              "H005", "guardrails")}, indent=2, default=float))


PHASES = {"preflight": phase_preflight, "pilot": phase_pilot, "search": phase_search,
          "eval": phase_eval, "control": phase_control, "timing": phase_timing,
          "summarize": phase_summarize}


def update_manifest(args, train_path: Path, train: pd.DataFrame, seconds: float) -> None:
    path = args.out_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update({
        "experiment": "E004",
        "hypotheses": ["H005"],
        "issue": 7,
        "data": {"path": train_path.as_posix(), "sha256": sha256(train_path),
                 "rows": len(train), "prompts": int(train.prompt_id.nunique()),
                 "models": int(train.model.nunique())},
        "validation": {"folds_csv": args.folds_csv.as_posix(), "n_splits": N_SPLITS,
                       "search_seed": SEARCH_SEED, "eval_seeds": list(EVAL_SEEDS),
                       "control": {"arm": CONTROL_ARM, "seed": CONTROL_SEED}},
        "arms": {k: {"model": v.name, "revision": v.revision, "pooling": v.pooling, "normalize": v.normalize,
                     "prompt": v.text_format.prompt, "end_token": v.text_format.end_token,
                     "dtype": str(v.dtype), "gradient_checkpointing": v.gradient_checkpointing,
                     "micro_batch_size": MICRO_BATCH[k], "load_kwargs": dict(v.load_kwargs)}
                 for k, v in ARMS.items()},
        "recipe": {**EncoderConfig(lr=0.0, epochs=0, max_length=MAX_LENGTH).to_dict(),
                   "lora": asdict(LORA),
                   "head": "dropout + linear", "search_lrs": list(SEARCH_LRS),
                   "lr_search_arms": list(LR_SEARCH_ARMS), "lr_source": LR_SOURCE,
                   "max_epochs": MAX_EPOCHS,
                   "schedule": "linear warmup 10% + linear decay over max_epochs; "
                               "eval and control stop after the chosen epoch"},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                     else None},
    })
    manifest.setdefault("phases", {})[args.phase] = {
        "command": " ".join(["python -m", __spec__.name if __spec__ else "run_lora",
                             *sys.argv[1:]]),
        "code_revision": git_revision(),
        "arms": ordered(args.arms), "folds": list(args.folds), "n_prompts": args.n_prompts,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(seconds, 1),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARM_ORDER, default=list(ARM_ORDER))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds-csv", type=Path,
                        default=Path("kb/research/data/E002/folds.csv"))
    parser.add_argument("--gte-runs", type=Path, default=Path("kb/research/data/E003/runs.csv"),
                        help="E003 runs.csv, for the per-fold comparison with gte")
    parser.add_argument("--folds", type=int, nargs="+", choices=ALL_FOLDS, default=ALL_FOLDS,
                        help="smoke tests only: run a subset of folds")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="smoke tests only: keep the first N prompts")
    parser.add_argument("--ignore-stop-rule", action="store_true",
                        help="smoke tests only: run eval even if Q0's search is below the floor")
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
