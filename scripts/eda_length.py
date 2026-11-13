"""Length EDA (issue #1): length by model/lab and within-prompt comparison.

Usage:
    uv run python scripts/eda_length.py [--data-dir data] [--out-dir kb/research/data/eda_length]

Writes CSV tables and PNG figures to --out-dir and prints the headline numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from name_that_llm.eda_length import (
    LENGTH_FEATURES,
    add_length_features,
    center_by_prompt,
    distribution_shift,
    friedman_within_prompt,
    kruskal_with_effect,
    lab_of,
    length_summary,
)

# Length range the competition states for passages.
DECLARED_MIN_WORDS, DECLARED_MAX_WORDS = 40, 250

# Fixed lab -> colour (validated categorical slots 1-6, light mode).
LAB_COLORS = {
    "Anthropic": "#2a78d6",
    "Google": "#eb6834",
    "OpenAI": "#1baf7a",
    "ZAI": "#eda100",
    "Moonshot AI": "#e87ba4",
    "Qwen": "#008300",
}
INK, MUTED = "#1f1f1e", "#6b6a64"


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (feature, test, grouping) with statistic, p-value and effect size."""
    rows = []
    for feature in LENGTH_FEATURES:
        for by in ("model", "lab"):
            kw = kruskal_with_effect(df, feature, by=by)
            rows.append(
                {
                    "feature": feature,
                    "test": "kruskal",
                    "by": by,
                    "statistic": kw.statistic,
                    "p_value": kw.p_value,
                    "effect": "epsilon_sq",
                    "effect_value": kw.epsilon_sq,
                    "effect_null": kw.epsilon_sq_null,
                    "n": kw.n,
                }
            )
        fr = friedman_within_prompt(df, feature)
        rows.append(
            {
                "feature": feature,
                "test": "friedman",
                "by": "model",
                "statistic": fr.statistic,
                "p_value": fr.p_value,
                "effect": "kendall_w",
                "effect_value": fr.kendall_w,
                "effect_null": 1 / fr.n_blocks,
                "n": fr.n_blocks,
                "n_blocks_dropped": fr.n_blocks_dropped,
            }
        )
    return pd.DataFrame(rows)


def out_of_range_by_model(df: pd.DataFrame) -> pd.DataFrame:
    below = df["n_words"] < DECLARED_MIN_WORDS
    above = df["n_words"] > DECLARED_MAX_WORDS
    table = (
        df.assign(below=below, above=above)
        .groupby("model")[["below", "above"]]
        .sum()
        .assign(total=lambda t: t.below + t.above)
    )
    return table.sort_values("total", ascending=False)


def plot_by_model(df: pd.DataFrame, col: str, title: str, path: Path) -> None:
    """Horizontal boxplots per model, sorted by median, coloured by lab."""
    order = df.groupby("model")[col].median().sort_values().index
    data = [df.loc[df.model == m, col].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(8, 7))
    boxes = ax.boxplot(
        data,
        orientation="horizontal",
        tick_labels=order,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
        flierprops={"marker": "o", "markersize": 3, "markeredgecolor": MUTED},
    )
    for patch, model in zip(boxes["boxes"], order):
        patch.set_facecolor(LAB_COLORS[lab_of(model)])
        patch.set_edgecolor("white")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LAB_COLORS.values()]
    ax.legend(handles, LAB_COLORS.keys(), loc="lower right", frameon=False, fontsize=8)
    _style(ax, title, col)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_train_vs_test(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    bins = range(0, int(max(train.n_words.max(), test.n_words.max())) + 10, 10)
    fig, ax = plt.subplots(figsize=(8, 4))
    for df, label, color in ((train, "train", "#2a78d6"), (test, "test", "#eb6834")):
        ax.hist(df.n_words, bins=bins, density=True, histtype="step", linewidth=2,
                color=color, label=f"{label} (n={len(df)})")
    for x in (DECLARED_MIN_WORDS, DECLARED_MAX_WORDS):
        ax.axvline(x, color=MUTED, linestyle=":", linewidth=1)
    ax.legend(frameon=False)
    _style(ax, "Word count: train vs test (dotted: declared 40–250 range)", "n_words")
    ax.set_ylabel("density", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("kb/research/data/eda_length"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = add_length_features(pd.read_csv(args.data_dir / "train.csv"))
    train["lab"] = train["model"].map(lab_of)
    train = center_by_prompt(train, list(LENGTH_FEATURES))
    test = add_length_features(pd.read_csv(args.data_dir / "test.csv"))

    tests = run_tests(train)
    friedman_words = friedman_within_prompt(train, "n_words")
    shift = distribution_shift(train.n_words, test.n_words)

    tests.to_csv(args.out_dir / "tests.csv", index=False)
    length_summary(train, "n_words", by="model").to_csv(args.out_dir / "words_by_model.csv")
    length_summary(train, "n_words", by="lab").to_csv(args.out_dir / "words_by_lab.csv")
    length_summary(train, "n_words_centered", by="model").to_csv(
        args.out_dir / "words_centered_by_model.csv")
    friedman_words.mean_ranks.rename("mean_rank").to_csv(
        args.out_dir / "words_friedman_mean_ranks.csv")
    out_of_range_by_model(train).to_csv(args.out_dir / "words_out_of_range_by_model.csv")
    pd.DataFrame([asdict(shift)]).to_csv(args.out_dir / "words_train_vs_test.csv", index=False)

    plot_by_model(train, "n_words", "Word count by model", args.out_dir / "words_by_model.png")
    plot_by_model(train, "n_words_centered", "Word count minus prompt mean, by model",
                  args.out_dir / "words_centered_by_model.png")
    plot_train_vs_test(train, test, args.out_dir / "words_train_vs_test.png")

    print(f"train: {len(train)} rows, {train.prompt_id.nunique()} prompts, "
          f"{train.model.nunique()} models; test: {len(test)} rows")
    print(train[list(LENGTH_FEATURES)].describe().round(1).to_string())
    print("\nTests:")
    print(tests.round(4).to_string(index=False))
    print(f"\nTrain vs test n_words: KS={shift.ks_statistic:.3f} p={shift.p_value:.3g} "
          f"median {shift.median_a:.0f} vs {shift.median_b:.0f}")
    print(f"\nOutputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
