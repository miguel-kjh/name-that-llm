"""Geometry of a set of embeddings: centring, cosine similarity by pair type, separation.

Pure numpy/sklearn, no GPU. Used by the E004 embedding diagnostic on the vectors the
classifier head sees. Every pair (i < j) of a set is used; a set is at most a few
thousand passages.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

SAME_MODEL, SAME_LAB, CROSS_LAB = "same_model", "same_lab_other_model", "cross_lab"
PAIR_TYPES = (SAME_MODEL, SAME_LAB, CROSS_LAB)


def unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to norm 1 (float64)."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mean_vector_norm(x: np.ndarray) -> float:
    """Norm of the mean unit vector: 0 if the directions are spread evenly, 1 if all equal.

    It is also the square root of the mean cosine over all ordered pairs, self-pairs included.
    """
    return float(np.linalg.norm(unit(x).mean(axis=0)))


def pair_cosines(x: np.ndarray) -> np.ndarray:
    """Cosine similarity of every pair i < j, in `np.triu_indices` order."""
    u = unit(x)
    return (u @ u.T)[np.triu_indices(len(u), k=1)]  # n x n, not n^2 x dim


def pair_types(models: Sequence[str], labs: Sequence[str]) -> np.ndarray:
    """Pair type of every pair i < j, in the same order as `pair_cosines`."""
    models, labs = np.asarray(models), np.asarray(labs)
    i, j = np.triu_indices(len(models), k=1)
    same_model = models[i] == models[j]
    same_lab = labs[i] == labs[j]
    return np.where(same_model, SAME_MODEL, np.where(same_lab, SAME_LAB, CROSS_LAB))


def describe(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0, "mean": None, "median": None, "p5": None, "p95": None}
    return {"n": int(len(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95))}


def separation_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """P(a random positive pair is more similar than a random negative one); 0.5 = none.

    None when either group is empty.
    """
    if len(positive) == 0 or len(negative) == 0:
        return None
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    return float(roc_auc_score(labels, scores))


def geometry_report(x: np.ndarray, models: Sequence[str], labs: Sequence[str],
                    center: np.ndarray | None = None,
                    prompts: Sequence | None = None) -> dict:
    """Centring and cosine structure of `x`.

    With `center`, the unit vectors are shifted by `-center` (the mean unit vector of a
    reference set, e.g. the training passages) before the cosines are taken.
    With `prompts`, also how much the vectors group by prompt (topic) rather than by model.
    """
    u = unit(x)
    if center is not None:
        u = u - center
    cos, kind = pair_cosines(u), pair_types(models, labs)
    groups = {t: cos[kind == t] for t in PAIR_TYPES}
    other_model = cos[kind != SAME_MODEL]
    return {
        "n": int(len(u)),
        "mean_vector_norm": mean_vector_norm(u),
        "all_pairs": describe(cos),
        **{t: describe(v) for t, v in groups.items()},
        "same_minus_other_model": (float(groups[SAME_MODEL].mean() - other_model.mean())
                                   if len(groups[SAME_MODEL]) and len(other_model) else None),
        "auc_same_vs_other_model": separation_auc(groups[SAME_MODEL], other_model),
        "auc_same_lab_vs_cross_lab": separation_auc(groups[SAME_LAB], groups[CROSS_LAB]),
        **({} if prompts is None else prompt_report(cos, models, prompts)),
    }


def prompt_report(cos: np.ndarray, models: Sequence[str], prompts: Sequence) -> dict:
    """Grouping by prompt, and by model once same-prompt pairs are left out."""
    models, prompts = np.asarray(models), np.asarray(prompts)
    i, j = np.triu_indices(len(models), k=1)
    same_prompt, same_model = prompts[i] == prompts[j], models[i] == models[j]
    return {
        "same_prompt": describe(cos[same_prompt]),
        "other_prompt": describe(cos[~same_prompt]),
        "auc_same_vs_other_prompt": separation_auc(cos[same_prompt], cos[~same_prompt]),
        "auc_same_vs_other_model_across_prompts": separation_auc(
            cos[same_model & ~same_prompt], cos[~same_model & ~same_prompt]),
    }
