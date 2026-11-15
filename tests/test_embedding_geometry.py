"""Embedding geometry on synthetic vectors whose answer is known."""

import numpy as np
import pytest

from name_that_llm.embedding_geometry import (
    CROSS_LAB,
    SAME_LAB,
    SAME_MODEL,
    geometry_report,
    mean_vector_norm,
    pair_cosines,
    pair_types,
    separation_auc,
    unit,
)


def test_orthogonal_vectors_are_centred():
    x = np.eye(6) * 3.0
    assert np.allclose(pair_cosines(x), 0)
    assert mean_vector_norm(np.vstack([x, -x])) == pytest.approx(0)


def test_identical_directions_have_mean_norm_one():
    x = np.tile([1.0, 2.0, 2.0], (5, 1)) * np.arange(1, 6)[:, None]  # norm does not matter
    assert mean_vector_norm(x) == pytest.approx(1)
    assert np.allclose(pair_cosines(x), 1)


def test_a_shifted_cloud_has_positive_cosines_that_centring_removes():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(400, 50))
    x = noise + 4.0 * np.eye(50)[0]  # every vector pushed along one direction
    raw = geometry_report(x, ["m"] * 400, ["L"] * 400)
    assert raw["mean_vector_norm"] > 0.4 and raw["all_pairs"]["mean"] > 0.2
    # Identity: n^2 |mean|^2 = n + sum over i != j of cos, so the mean over i < j is exact.
    n, norm = 400, raw["mean_vector_norm"]
    assert raw["all_pairs"]["mean"] == pytest.approx((n * n * norm ** 2 - n) / (n * (n - 1)))
    centred = geometry_report(x, ["m"] * 400, ["L"] * 400, center=unit(x).mean(axis=0))
    assert abs(centred["all_pairs"]["mean"]) < 0.02
    assert centred["mean_vector_norm"] < 0.05


def test_pair_types_follow_triu_order():
    models = ["a1", "a1", "a2", "b1"]
    labs = ["A", "A", "A", "B"]
    # pairs: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    assert pair_types(models, labs).tolist() == [
        SAME_MODEL, SAME_LAB, CROSS_LAB, SAME_LAB, CROSS_LAB, CROSS_LAB]
    x = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=float)
    assert pair_cosines(x).tolist() == pytest.approx([1, 0, -1, 0, -1, 0])


def test_separated_classes_have_auc_one_and_mixed_ones_half():
    rng = np.random.default_rng(1)
    centres = np.eye(20)[:4] * 10
    models = np.repeat(["m0", "m1", "m2", "m3"], 30)
    labs = np.repeat(["A", "A", "B", "B"], 30)
    x = centres[np.repeat(np.arange(4), 30)] + rng.normal(size=(120, 20))
    report = geometry_report(x, models, labs)
    assert report["auc_same_vs_other_model"] > 0.99
    assert report["same_minus_other_model"] > 0.5
    mixed = geometry_report(rng.normal(size=(120, 20)), models, labs)
    assert abs(mixed["auc_same_vs_other_model"] - 0.5) < 0.05


def test_separation_auc_extremes():
    assert separation_auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert separation_auc(np.array([0.1]), np.array([0.9])) == 0.0


def test_model_markers_are_distinct_within_each_lab():
    pytest.importorskip("matplotlib")
    from name_that_llm.experiments.embedding_diagnostics import model_markers

    models = ["claude-a", "claude-b", "gpt-a", "gpt-b", "gpt-c", "kimi-k3"]
    m = model_markers(models)
    assert len({m["claude-a"], m["claude-b"]}) == 2
    assert len({m["gpt-a"], m["gpt-b"], m["gpt-c"]}) == 3


def test_topic_clusters_are_seen_as_prompt_not_model_grouping():
    rng = np.random.default_rng(2)
    prompts = np.repeat(np.arange(6), 12)  # 6 prompts x 12 passages
    models = np.tile(np.repeat(["m0", "m1", "m2"], 4), 6)  # every model on every prompt
    labs = np.where(models == "m0", "A", "B")
    x = np.eye(30)[prompts] * 10 + rng.normal(size=(72, 30))  # vectors encode only the prompt
    r = geometry_report(x, models, labs, prompts=prompts)
    assert r["auc_same_vs_other_prompt"] > 0.99
    assert abs(r["auc_same_vs_other_model_across_prompts"] - 0.5) < 0.1
    assert "same_prompt" not in geometry_report(x, models, labs)
