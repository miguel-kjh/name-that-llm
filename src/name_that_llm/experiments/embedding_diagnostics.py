"""Geometry of the embeddings the E004 head sees (diagnostic annex to E004, issue #7).

Usage:
    python -m name_that_llm.experiments.embedding_diagnostics --phase extract   (GPU)
    python -m name_that_llm.experiments.embedding_diagnostics --phase analyze   (CPU)

One fold: fold seed 1, fold 0 (train on 4 folds, ~2,160 passages; validate on ~540).
Variants, for Q0 and Q8:
    {arm}_base  pretrained encoder, no LoRA (same prompt, pooling and normalization)
    {arm}_lora  retrained with E004's chosen config; adapter and head are saved

extract  embeddings of every passage (both splits) per variant -> artifacts/
         + checks (fold accuracy vs E004, head(embed) == predict_proba, the saved
         adapter reloads to the same predictions)          -> checks.json
analyze  t-SNE and PCA by lab and by model, cosine histograms by pair type,
         centring, separation by model, lab and prompt                   -> png + geometry.json

Embeddings and adapters are large and stay out of git, in `artifacts/`; only figures
and metrics go to `kb/`. Each variant is skipped when its files already exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from name_that_llm.embedding_geometry import (  # noqa: E402
    CROSS_LAB,
    PAIR_TYPES,
    SAME_LAB,
    SAME_MODEL,
    geometry_report,
    pair_cosines,
    pair_types,
    unit,
)
from name_that_llm.encoder import (  # noqa: E402
    PooledClassifier,
    backbone_factory,
    load_tokenizer,
    predict_embeddings,
    predict_proba,
    train_classifier,
)
from name_that_llm.experiments.eda_length import INK, LAB_COLORS, MUTED  # noqa: E402
from name_that_llm.experiments.provenance import git_revision  # noqa: E402
from name_that_llm.experiments.run_lora import (  # noqa: E402
    ARMS,
    MAX_LENGTH,
    PREDICT_BATCH,
    base_config,
    chosen_config,
)
from name_that_llm.experiments.runs import device, load_folds  # noqa: E402
from name_that_llm.labels import lab_of  # noqa: E402
from name_that_llm.metrics import balanced_accuracy  # noqa: E402

ARMS_USED = ("Q0", "Q8")
VARIANTS = tuple(f"{a}_{k}" for a in ARMS_USED for k in ("base", "lora"))
FOLD_SEED, FOLD = 1, 0
SPLITS = ("train", "val")
E004_DIR = Path("kb/research/data/E004")
DEFAULT_OUT_DIR = E004_DIR / "embeddings"
DEFAULT_ARTIFACTS = Path("artifacts/E004-embeddings")
# GPU nondeterminism between identical runs (E004 regression check): ~0.02 per fold.
FOLD_TOLERANCE = 0.03
PROBA_TOLERANCE = 1e-2  # bf16 autocast vs a float32 head on the same embeddings
TSNE_PERPLEXITY, TSNE_SEED, PCA_BEFORE_TSNE = 30, 0, 50
MARKERS = ("o", "s", "^", "D", "v")  # models within a lab; at most 5 per lab


# ---------------------------------------------------------------- extract (GPU)


def _classifier(spec, n_classes: int) -> PooledClassifier:
    return PooledClassifier(backbone_factory(spec)(), n_classes, pooling=spec.pooling,
                            text_format=spec.text_format, normalize=spec.normalize)


def _base_spec(arm: str):
    return replace(ARMS[arm], lora=None, gradient_checkpointing=False)


def head_probabilities(emb: np.ndarray, head_state: dict) -> np.ndarray:
    """Softmax of the linear head applied to `emb` (the head is a plain nn.Linear)."""
    w = head_state["weight"].float().cpu().numpy()
    b = head_state["bias"].float().cpu().numpy()
    logits = emb @ w.T + b
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def reference_fold_score(runs_csv: Path, arm: str) -> float:
    runs = pd.read_csv(runs_csv)
    row = runs[(runs.phase == "eval") & (runs.arm == arm) & (runs.seed == FOLD_SEED)
               & (runs.fold == FOLD)]
    return float(row.balanced_accuracy.iloc[0])


def extract_base(arm, texts, classes, tokenizer, art: Path) -> None:
    path = art / f"{arm}_base" / "embeddings.npy"
    if path.exists():
        return
    model = _classifier(_base_spec(arm), len(classes)).to(device())
    emb = predict_embeddings(model, tokenizer, texts, MAX_LENGTH,
                             batch_size=PREDICT_BATCH[arm], device=device())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    del model
    torch.cuda.empty_cache()


def extract_lora(arm, train, y, is_val, classes, tokenizer, art: Path, args) -> dict | None:
    """Retrain the arm on the training folds, save adapter + head + embeddings, run checks."""
    out = art / f"{arm}_lora"
    if (out / "embeddings.npy").exists():
        return None
    spec = ARMS[arm]
    c = chosen_config(E004_DIR, arm, tuple(range(5)))
    cfg = replace(base_config(arm), lr=c["lr"], epochs=c["epochs"], seed=FOLD_SEED)
    texts = train.text.tolist()
    val_texts = train.text[is_val].tolist()
    t0 = time.perf_counter()
    model, history = train_classifier(
        train.text[~is_val].tolist(), y[~is_val], len(classes), cfg, tokenizer,
        backbone_factory(spec), device(), pooling=spec.pooling, text_format=spec.text_format,
        normalize=spec.normalize)
    train_seconds = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out / "adapter")
    head_state = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
    torch.save(head_state, out / "head.pt")
    batch = PREDICT_BATCH[arm]
    probs = predict_proba(model, tokenizer, val_texts, MAX_LENGTH, batch_size=batch,
                          device=device())
    emb = predict_embeddings(model, tokenizer, texts, MAX_LENGTH, batch_size=batch,
                             device=device())
    del model
    torch.cuda.empty_cache()

    # Reload from disk: fresh base weights + saved adapter + saved head.
    from peft import PeftModel

    base = backbone_factory(_base_spec(arm))()
    reloaded = PooledClassifier(PeftModel.from_pretrained(base, out / "adapter"), len(classes),
                                pooling=spec.pooling, text_format=spec.text_format,
                                normalize=spec.normalize)
    reloaded.head.load_state_dict(head_state)
    reloaded_probs = predict_proba(reloaded.to(device()), tokenizer, val_texts, MAX_LENGTH,
                                   batch_size=batch, device=device())
    del reloaded, base
    torch.cuda.empty_cache()
    np.save(out / "embeddings.npy", emb)

    truth = train.model[is_val].to_numpy()
    score = balanced_accuracy(truth, [classes[i] for i in probs.argmax(axis=1)])
    reference = reference_fold_score(E004_DIR / "runs.csv", arm)
    head_diff = float(np.abs(head_probabilities(emb[is_val], head_state) - probs).max())
    reload_diff = float(np.abs(reloaded_probs - probs).max())
    reload_agree = float((reloaded_probs.argmax(1) == probs.argmax(1)).mean())
    return {
        "config": {"lr": c["lr"], "epochs": c["epochs"], "fold_seed": FOLD_SEED, "fold": FOLD},
        "loss_history": [round(v, 5) for v in history],
        "train_seconds": round(train_seconds, 1),
        "fold_balanced_accuracy": score,
        "e004_fold_balanced_accuracy": reference,
        "fold_matches_e004": abs(score - reference) <= FOLD_TOLERANCE,
        "head_on_embeddings_max_abs_diff": head_diff,
        "head_on_embeddings_matches": head_diff <= PROBA_TOLERANCE,
        "reloaded_adapter_max_abs_diff": reload_diff,
        "reloaded_adapter_argmax_agreement": reload_agree,
        "reloaded_adapter_matches": reload_diff <= PROBA_TOLERANCE,
    }


def phase_extract(train, folds, classes, args) -> None:
    is_val = (folds == FOLD).to_numpy()
    y = train.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    checks_path = args.out_dir / "checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    for arm in ARMS_USED:
        tokenizer = load_tokenizer(ARMS[arm])
        extract_base(arm, train.text.tolist(), classes, tokenizer, args.artifacts)
        print(f"{arm}_base: embeddings done", flush=True)
        result = extract_lora(arm, train, y, is_val, classes, tokenizer, args.artifacts, args)
        if result is not None:
            checks[f"{arm}_lora"] = result
            checks_path.write_text(json.dumps(checks, indent=2))
            print(f"{arm}_lora: {json.dumps(result)}", flush=True)


# ---------------------------------------------------------------- analyze (CPU)


def model_markers(models: list[str]) -> dict[str, str]:
    """A marker per model, distinct within its lab (models in sorted order)."""
    markers, used = {}, {}
    for m in sorted(models):
        lab = lab_of(m)
        markers[m] = MARKERS[used.get(lab, 0)]
        used[lab] = used.get(lab, 0) + 1
    return markers


def tsne_2d(x: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    reduced = PCA(n_components=min(PCA_BEFORE_TSNE, *x.shape), random_state=0).fit_transform(
        unit(x))
    return TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init="pca",
                learning_rate="auto", random_state=TSNE_SEED).fit_transform(reduced)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(unit(x)), pca.explained_variance_ratio_.round(4).tolist()


def _clean(ax, title: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#d9d8d3")


def plot_projection(coords: dict[str, np.ndarray], meta: pd.DataFrame, title: str,
                    path: Path, subtitles: dict[str, str] | None = None) -> None:
    """Rows: coloured by lab; lab colour + a marker per model. Columns: variants."""
    variants = list(coords)
    markers = model_markers(meta.model.unique().tolist())
    fig, axes = plt.subplots(2, len(variants), figsize=(4.2 * len(variants), 8.6),
                             squeeze=False)
    for col, v in enumerate(variants):
        xy = coords[v]
        sub = f" ({subtitles[v]})" if subtitles else ""
        ax = axes[0, col]
        for lab, color in LAB_COLORS.items():
            sel = (meta.lab == lab).to_numpy()
            ax.scatter(xy[sel, 0], xy[sel, 1], s=9, color=color, alpha=0.75, linewidths=0,
                       label=lab)
        _clean(ax, f"{v} · por laboratorio{sub}")
        ax = axes[1, col]
        for m in sorted(markers):
            sel = (meta.model == m).to_numpy()
            ax.scatter(xy[sel, 0], xy[sel, 1], s=14, color=LAB_COLORS[lab_of(m)],
                       marker=markers[m], alpha=0.75, linewidths=0.3, edgecolors="white",
                       label=m)
        _clean(ax, f"{v} · por modelo")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    axes[1, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=7,
                       ncol=1)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


PAIR_COLORS = dict(zip(PAIR_TYPES, list(LAB_COLORS.values())[:3]))  # fixed slots 1-3
PAIR_LABELS = {SAME_MODEL: "mismo modelo", SAME_LAB: "mismo laboratorio, otro modelo",
               CROSS_LAB: "distinto laboratorio"}


def plot_cosines(cos: dict, title: str, path: Path) -> None:
    """Rows: raw and train-centred cosines. Columns: variants. One curve per pair type."""
    variants = list(cos)
    fig, axes = plt.subplots(2, len(variants), figsize=(4.2 * len(variants), 6.4),
                             squeeze=False)
    for col, v in enumerate(variants):
        for row, kind in enumerate(("raw", "centred")):
            ax = axes[row, col]
            values, types = cos[v][kind]
            bins = np.linspace(-1, 1, 161)
            for t in PAIR_TYPES:
                ax.hist(values[types == t], bins=bins, density=True, histtype="step",
                        linewidth=2, color=PAIR_COLORS[t], label=PAIR_LABELS[t])
            ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
            ax.set_xlim(-0.6, 1.0)
            ax.set_yticks([])
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#d9d8d3")
            label = "sin centrar" if kind == "raw" else "centrado (media del train)"
            ax.set_title(f"{v} · {label}", fontsize=10, color=INK, loc="left")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("similitud coseno", fontsize=9, color=MUTED)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def phase_analyze(train, folds, classes, args) -> None:
    meta = pd.DataFrame({"id": train.id, "model": train.model,
                         "lab": train.model.map(lab_of), "prompt_id": train.prompt_id,
                         "split": np.where(folds == FOLD, "val", "train")})
    emb = {v: np.load(args.artifacts / v / "embeddings.npy") for v in VARIANTS}
    for v, x in emb.items():
        if len(x) != len(meta):
            raise SystemExit(f"{v}: {len(x)} embeddings for {len(meta)} passages")
    report, cos = {}, {s: {} for s in SPLITS}
    for v, x in emb.items():
        is_train = (meta.split == "train").to_numpy()
        center = unit(x[is_train]).mean(axis=0)
        report[v] = {"dim": int(x.shape[1]),
                     "norm_mean": float(np.linalg.norm(x, axis=1).mean())}
        for s in SPLITS:
            sel = (meta.split == s).to_numpy()
            m = meta[sel]
            report[v][s] = {
                "raw": geometry_report(x[sel], m.model, m.lab, prompts=m.prompt_id),
                "centred": geometry_report(x[sel], m.model, m.lab, center=center,
                                           prompts=m.prompt_id)}
            types = pair_types(m.model, m.lab)
            cos[s][v] = {"raw": (pair_cosines(unit(x[sel])), types),
                         "centred": (pair_cosines(unit(x[sel]) - center), types)}

    for s in SPLITS:
        sel = (meta.split == s).to_numpy()
        m = meta[sel].reset_index(drop=True)
        n = f"{s} (n = {sel.sum()})"
        tsne = {v: tsne_2d(emb[v][sel]) for v in VARIANTS}
        plot_projection(tsne, m, f"t-SNE de los embeddings, {n}",
                        args.out_dir / f"tsne_{s}.png")
        pca, ratios = {}, {}
        for v in VARIANTS:
            pca[v], ratios[v] = pca_2d(emb[v][sel])
            report[v][s]["pca_explained_variance"] = ratios[v]
        plot_projection(pca, m, f"PCA de los embeddings, {n}", args.out_dir / f"pca_{s}.png",
                        subtitles={v: f"PC1 {ratios[v][0]:.0%}, PC2 {ratios[v][1]:.0%}"
                                   for v in VARIANTS})
        plot_cosines(cos[s], f"Similitud coseno por tipo de par, {n}",
                     args.out_dir / f"cosine_{s}.png")
    (args.out_dir / "geometry.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({v: {s: {k: round(report[v][s]["raw"][k], 3) for k in
                              ("mean_vector_norm", "auc_same_vs_other_model",
                               "auc_same_vs_other_prompt")}
                          for s in SPLITS} for v in VARIANTS}, indent=2))


# ---------------------------------------------------------------- main


PHASES = {"extract": phase_extract, "analyze": phase_analyze}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--folds-csv", type=Path,
                        default=Path("kb/research/data/E002/folds.csv"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    folds = load_folds(train, pd.read_csv(args.folds_csv), FOLD_SEED)
    classes = sorted(train.model.unique())
    t0 = time.perf_counter()
    PHASES[args.phase](train, folds, classes, args)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"experiment": "E004", "diagnostic": "embedding geometry",
                     "fold_seed": FOLD_SEED, "fold": FOLD, "variants": list(VARIANTS),
                     "artifacts": args.artifacts.as_posix(),
                     "tsne": {"perplexity": TSNE_PERPLEXITY, "seed": TSNE_SEED,
                              "pca_before": PCA_BEFORE_TSNE}})
    manifest.setdefault("phases", {})[args.phase] = {
        "command": " ".join(["python -m name_that_llm.experiments.embedding_diagnostics",
                             *sys.argv[1:]]),
        "code_revision": git_revision(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
