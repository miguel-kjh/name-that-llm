import numpy as np
import pandas as pd
import pytest

from name_that_llm.eda_length import (
    add_length_features,
    center_by_prompt,
    count_sentences,
    friedman_within_prompt,
    kruskal_with_effect,
    lab_of,
    length_summary,
)

MODELS = [f"m{i}" for i in range(6)]
N_PROMPTS = 60


def crossed_design(model_shift=None, prompt_effect=0.0, seed=0):
    """One length per (prompt, model): noise + optional prompt and model effects."""
    rng = np.random.default_rng(seed)
    model_shift = model_shift or {}
    rows = []
    for p in range(N_PROMPTS):
        prompt_offset = prompt_effect * rng.normal()
        for m in MODELS:
            value = 100 + prompt_offset + model_shift.get(m, 0) + rng.normal(0, 10)
            rows.append({"prompt_id": p, "model": m, "length": value})
    return pd.DataFrame(rows)


def test_no_model_effect_gives_high_p_and_tiny_effect():
    df = crossed_design()
    kw = kruskal_with_effect(df, "length", by="model")
    fr = friedman_within_prompt(df, "length")
    # With no effect, ε² sits near its null expectation (k-1)/(n-1), not at 0.
    assert kw.p_value > 0.05
    assert kw.epsilon_sq_null == pytest.approx(5 / 359)
    assert kw.epsilon_sq < 3 * kw.epsilon_sq_null
    assert fr.p_value > 0.05 and fr.kendall_w < 0.05


def test_shifted_model_is_detected_and_ranked_highest():
    df = crossed_design(model_shift={"m3": 30})
    kw = kruskal_with_effect(df, "length", by="model")
    fr = friedman_within_prompt(df, "length")
    assert kw.p_value < 1e-6
    assert fr.p_value < 1e-6
    assert fr.mean_ranks.idxmax() == "m3"


def test_prompt_effect_is_not_mistaken_for_model_effect():
    df = crossed_design(prompt_effect=50)
    assert kruskal_with_effect(df, "length", by="model").p_value > 0.05
    assert friedman_within_prompt(df, "length").p_value > 0.05


def test_friedman_removes_prompt_effect_that_hides_model_shift():
    # A shift of 5 is invisible across prompts (sd 50) but clear within prompt.
    df = crossed_design(model_shift={"m0": 5}, prompt_effect=50)
    kw = kruskal_with_effect(df, "length", by="model")
    fr = friedman_within_prompt(df, "length")
    assert fr.p_value < kw.p_value
    assert fr.mean_ranks.idxmax() == "m0"


def test_centering_makes_every_prompt_mean_zero():
    df = center_by_prompt(crossed_design(prompt_effect=50), ["length"])
    prompt_means = df.groupby("prompt_id")["length_centered"].mean()
    assert np.allclose(prompt_means, 0)
    assert "length" in df  # original column is kept


def test_incomplete_prompt_is_dropped_and_reported():
    df = crossed_design()
    df = df.drop(df[(df.prompt_id == 0) & (df.model == "m2")].index)
    fr = friedman_within_prompt(df, "length")
    assert fr.n_blocks == N_PROMPTS - 1
    assert fr.n_blocks_dropped == 1


def test_duplicate_prompt_model_pair_is_rejected():
    df = crossed_design()
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="More than one row"):
        friedman_within_prompt(df, "length")


def test_epsilon_sq_for_perfect_separation_is_one_minus_inverse_k_squared():
    # Two equal groups, fully separated: within-group rank spread remains,
    # so ε² tends to 1 - 1/k² = 0.75 rather than 1.
    df = pd.DataFrame({"g": ["a"] * 20 + ["b"] * 20, "x": range(40)})
    eps = kruskal_with_effect(df, "x", by="g").epsilon_sq
    assert eps == pytest.approx(0.75, abs=0.01)


@pytest.mark.parametrize(
    "model, lab",
    [
        ("claude-opus-4-6", "Anthropic"),
        ("gemini-3.6-flash", "Google"),
        ("gpt-5.6-luna", "OpenAI"),
        ("glm-5.3-flash", "ZAI"),
        ("kimi-k3", "Moonshot AI"),
        ("qwen3.8-max", "Qwen"),
    ],
)
def test_lab_of_known_models(model, lab):
    assert lab_of(model) == lab


def test_lab_of_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        lab_of("llama-5")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("One. Two! Three?", 3),
        ("No terminator", 1),
        ("Version 3.5 is out. Done.", 2),  # decimal point is not a sentence end
        ("Wait... what?!", 2),
        ("", 0),
    ],
)
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected


def test_add_length_features_counts_and_keeps_input():
    df = pd.DataFrame({"text": ["Hello there. Bye.", "one  two\nthree"]})
    out = add_length_features(df)
    assert out["n_words"].tolist() == [3, 3]
    assert out["n_chars"].tolist() == [17, 14]
    assert out["n_sentences"].tolist() == [2, 1]
    assert list(df.columns) == ["text"]


def test_length_summary_matches_pandas():
    df = crossed_design()
    summary = length_summary(df, "length", by="model")
    expected = df.groupby("model")["length"].median().sort_values()
    assert summary.index.tolist() == expected.index.tolist()
    assert np.allclose(summary["median"], expected)
    assert (summary["count"] == N_PROMPTS).all()
