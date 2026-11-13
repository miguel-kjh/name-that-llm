"""Encoder tests on CPU with a tiny random ModernBERT and an in-memory tokenizer.

Nothing is downloaded: the real models are checked by `run_encoder --phase preflight`.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast  # noqa: E402

import name_that_llm.encoder as encoder  # noqa: E402
from name_that_llm.encoder import (  # noqa: E402
    EncoderArm,
    EncoderConfig,
    MeanPoolClassifier,
    backbone_factory,
    count_truncated,
    mean_pool,
    predict_proba,
    train_classifier,
)

WORDS = [f"w{i}" for i in range(60)]
N_CLASSES = 4


@pytest.fixture(scope="module")
def tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")


def tiny_backbone():
    config = ModernBertConfig(
        vocab_size=len(WORDS) + 2, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, global_attn_every_n_layers=1,
        local_attention=16, max_position_embeddings=64, pad_token_id=0,
        bos_token_id=None, eos_token_id=None, cls_token_id=None, sep_token_id=None,
    )
    return ModernBertModel(config)


def synthetic(n_per_class=8, seed=0):
    """Each class has its own keyword, so the texts are perfectly separable."""
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for label in range(N_CLASSES):
        for _ in range(n_per_class):
            filler = rng.choice(WORDS[N_CLASSES:], size=rng.integers(3, 10)).tolist()
            texts.append(" ".join([WORDS[label], *filler]))
            labels.append(label)
    return texts, np.array(labels)


def test_mean_pool_ignores_padding():
    hidden = torch.randn(1, 3, 8)
    padded = torch.cat([hidden, torch.randn(1, 4, 8) * 100], dim=1)
    short = mean_pool(hidden, torch.ones(1, 3))
    long = mean_pool(padded, torch.tensor([[1, 1, 1, 0, 0, 0, 0]]))
    assert torch.allclose(short, long)


def test_mean_pool_is_the_mean_of_unmasked_rows():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert mean_pool(hidden, torch.tensor([[1, 1, 0]])).item() == pytest.approx(2.0)


def test_classifier_output_shape(tokenizer):
    model = MeanPoolClassifier(tiny_backbone(), n_classes=18)
    batch = tokenizer(["w1 w2 w3", "w4"], padding=True, return_tensors="pt")
    assert model(batch["input_ids"], batch["attention_mask"]).shape == (2, 18)


def test_prediction_does_not_depend_on_batch_padding(tokenizer):
    torch.manual_seed(0)
    model = MeanPoolClassifier(tiny_backbone(), n_classes=N_CLASSES)
    texts = ["w5 w6", "w7 w8 w9 w10 w11 w12 w13 w14 w15"]
    together = predict_proba(model, tokenizer, texts, batch_size=2)
    alone = np.vstack([predict_proba(model, tokenizer, [t], batch_size=1) for t in texts])
    np.testing.assert_allclose(together, alone, atol=1e-5)


def test_predict_proba_rows_sum_to_one_and_keep_order(tokenizer):
    texts, labels = synthetic()
    model, _ = train_classifier(texts, labels, N_CLASSES, EncoderConfig(lr=3e-3, epochs=8),
                                tokenizer, tiny_backbone)
    # Shuffle lengths so the internal length sort would scramble a naive implementation.
    order = np.random.default_rng(1).permutation(len(texts))
    probs = predict_proba(model, tokenizer, [texts[i] for i in order], batch_size=5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    single = predict_proba(model, tokenizer, [texts[order[3]]])
    np.testing.assert_allclose(probs[3], single[0], atol=1e-5)


def test_tiny_model_memorizes_separable_texts(tokenizer):
    texts, labels = synthetic()
    model, history = train_classifier(texts, labels, N_CLASSES, EncoderConfig(lr=3e-3, epochs=20),
                                      tokenizer, tiny_backbone)
    preds = predict_proba(model, tokenizer, texts).argmax(axis=1)
    assert (preds == labels).mean() == 1.0
    assert history[-1] < 0.5 * history[0]
    assert len(history) == 1 + 20


def test_same_seed_same_history_other_seed_differs(tokenizer):
    texts, labels = synthetic()

    def run(seed):
        cfg = EncoderConfig(lr=1e-3, epochs=2, batch_size=4, seed=seed)
        return train_classifier(texts, labels, N_CLASSES, cfg, tokenizer, tiny_backbone)[1]

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_count_truncated(tokenizer):
    assert count_truncated(tokenizer, ["w1 w2 w3", "w1"], max_length=2) == 1
    assert count_truncated(tokenizer, ["w1 w2 w3", "w1"], max_length=3) == 0


def test_backbone_factory_always_loads_float32_weights(monkeypatch):
    # A float16 checkpoint (like gte-modernbert-base) must not be trained in float16.
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return "backbone"

    monkeypatch.setattr(encoder, "AutoModel", FakeAutoModel)
    make = backbone_factory(EncoderArm("org/model", "abc123"))
    assert make() == "backbone"
    assert calls == [("org/model", {"revision": "abc123", "dtype": torch.float32})]
