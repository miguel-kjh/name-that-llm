"""Length features and statistical tests for the length EDA (issue #1).

All functions are pure: they take DataFrames and return new objects, with no I/O.
The training data has a fully crossed design (each prompt answered by every model),
which is what makes the within-prompt Friedman test possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from scipy import stats

LENGTH_FEATURES = ("n_words", "n_chars", "n_sentences")

# Model-name prefix -> lab. Order does not matter: prefixes do not overlap.
_LAB_BY_PREFIX = {
    "claude": "Anthropic",
    "gemini": "Google",
    "gpt": "OpenAI",
    "glm": "ZAI",
    "kimi": "Moonshot AI",
    "qwen": "Qwen",
}

# A sentence ends at one or more of .!? followed by whitespace or end of text.
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


def lab_of(model: str) -> str:
    """Return the lab that trained `model`; raise on an unknown model name."""
    for prefix, lab in _LAB_BY_PREFIX.items():
        if model.startswith(prefix):
            return lab
    raise ValueError(f"Unknown model, no lab mapping: {model!r}")


def count_sentences(text: str) -> int:
    """Count sentences; a non-empty text without a terminator counts as one."""
    n_ends = len(_SENTENCE_END.findall(text))
    return max(n_ends, 1 if text.strip() else 0)


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with word, character and sentence counts of `text`."""
    out = df.copy()
    out["n_words"] = out["text"].str.split().str.len()
    out["n_chars"] = out["text"].str.len()
    out["n_sentences"] = out["text"].map(count_sentences)
    return out


def length_summary(df: pd.DataFrame, col: str, by: str) -> pd.DataFrame:
    """Descriptive statistics of `col` per group, sorted by median."""
    grouped = df.groupby(by)[col]
    summary = pd.DataFrame(
        {
            "count": grouped.count(),
            "mean": grouped.mean(),
            "std": grouped.std(),
            "min": grouped.min(),
            "q25": grouped.quantile(0.25),
            "median": grouped.median(),
            "q75": grouped.quantile(0.75),
            "max": grouped.max(),
        }
    )
    return summary.sort_values("median")


@dataclass(frozen=True)
class KruskalResult:
    statistic: float
    p_value: float
    # Share of rank variance explained by the groups. It is not 0 under the null:
    # E[H] = k - 1, so its expected value with no effect is `epsilon_sq_null`.
    # Its maximum is below 1 unless groups are internally tied (1 - 1/k² for k
    # equal-sized, perfectly separated groups).
    epsilon_sq: float
    epsilon_sq_null: float
    n: int
    n_groups: int


def kruskal_with_effect(df: pd.DataFrame, col: str, by: str) -> KruskalResult:
    """Kruskal-Wallis test of `col` across `by` groups, with ε² = H / (n - 1)."""
    samples = [group[col].to_numpy() for _, group in df.groupby(by)]
    h, p = stats.kruskal(*samples)
    n, k = len(df), len(samples)
    return KruskalResult(
        statistic=float(h),
        p_value=float(p),
        epsilon_sq=float(h / (n - 1)),
        epsilon_sq_null=(k - 1) / (n - 1),
        n=n,
        n_groups=k,
    )


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    kendall_w: float  # agreement of within-block rankings, in [0, 1]
    n_blocks: int
    n_blocks_dropped: int  # blocks missing at least one treatment
    mean_ranks: pd.Series  # mean within-block rank per treatment (1 = smallest)


def friedman_within_prompt(
    df: pd.DataFrame, col: str, by: str = "model", block: str = "prompt_id"
) -> FriedmanResult:
    """Friedman test of `col` across `by`, ranking within each `block`.

    Requires at most one row per (block, treatment). Blocks that lack any
    treatment are dropped, because Friedman needs complete blocks.
    """
    if df.duplicated([block, by]).any():
        raise ValueError(f"More than one row per ({block}, {by}) pair")

    wide = df.pivot(index=block, columns=by, values=col)
    complete = wide.dropna()
    n_blocks, k = complete.shape
    chi2, p = stats.friedmanchisquare(*(complete[c].to_numpy() for c in complete))
    mean_ranks = complete.rank(axis=1).mean().sort_values()
    return FriedmanResult(
        statistic=float(chi2),
        p_value=float(p),
        kendall_w=float(chi2 / (n_blocks * (k - 1))),
        n_blocks=n_blocks,
        n_blocks_dropped=len(wide) - n_blocks,
        mean_ranks=mean_ranks,
    )


def center_by_prompt(
    df: pd.DataFrame, cols: list[str], block: str = "prompt_id"
) -> pd.DataFrame:
    """Return a copy with `<col>_centered` = col minus its block mean, for each col."""
    out = df.copy()
    block_means = out.groupby(block)[cols].transform("mean")
    for col in cols:
        out[f"{col}_centered"] = out[col] - block_means[col]
    return out


@dataclass(frozen=True)
class ShiftResult:
    ks_statistic: float
    p_value: float
    median_a: float
    median_b: float


def distribution_shift(a: pd.Series, b: pd.Series) -> ShiftResult:
    """Two-sample Kolmogorov-Smirnov comparison of two distributions."""
    ks, p = stats.ks_2samp(a.to_numpy(), b.to_numpy())
    return ShiftResult(
        ks_statistic=float(ks),
        p_value=float(p),
        median_a=float(a.median()),
        median_b=float(b.median()),
    )
