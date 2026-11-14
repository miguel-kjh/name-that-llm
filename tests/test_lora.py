"""E004 encoder features on CPU: last-token pooling, text formats, LoRA, micro-batches.

A tiny random Qwen3 (causal, like Qwen3-Embedding) and an in-memory tokenizer.
Nothing is downloaded: the real models are checked by `run_lora --phase preflight`.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model  # noqa: E402

import name_that_llm.encoder as encoder  # noqa: E402
from name_that_llm.encoder import (  # noqa: E402
    ARMS,
    EncoderArm,
    EncoderConfig,
    LoraSpec,
    PooledClassifier,
    TextFormat,
    accumulate_gradients,
    backbone_factory,
    count_truncated,
    output_size,
    last_token_pool,
    predict_proba,
    tokenize,
    train_classifier,
    wrap_backbone,
)

WORDS = [f"w{i}" for i in range(60)]
N_CLASSES = 4
PAD, UNK, EOS = 0, 1, 2
FORMAT = TextFormat(prompt="w59 w58 ", end_token="[EOS]")


def _tokenizer(add_eos: bool) -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": PAD, "[UNK]": UNK, "[EOS]": EOS, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if add_eos:  # like a tokenizer whose post-processor already ends with EOS
        tok.post_processor = processors.TemplateProcessing(
            single="$A [EOS]", special_tokens=[("[EOS]", EOS)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   eos_token="[EOS]")


@pytest.fixture(scope="module")
def tokenizer():
    return _tokenizer(add_eos=False)


def tiny_qwen():
    config = Qwen3Config(
        vocab_size=len(WORDS) + 3, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, pad_token_id=PAD, attention_dropout=0.0,
    )
    return Qwen3Model(config)


LORA_ARM = EncoderArm("tiny", "rev", pooling="last", text_format=FORMAT,
                      lora=LoraSpec(r=4, alpha=8, dropout=0.0))


def tiny_lora():
    return wrap_backbone(tiny_qwen(), LORA_ARM)


def synthetic(n_per_class=8, seed=0):
    """Each class has its own keyword, so the texts are perfectly separable."""
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for label in range(N_CLASSES):
        for _ in range(n_per_class):
            filler = rng.choice(WORDS[N_CLASSES:50], size=rng.integers(3, 10)).tolist()
            texts.append(" ".join([WORDS[label], *filler]))
            labels.append(label)
    return texts, np.array(labels)


def train_lora(texts, labels, cfg, **kwargs):
    return train_classifier(texts, labels, N_CLASSES, cfg, kwargs.pop("tok"), tiny_lora,
                            pooling="last", text_format=FORMAT, **kwargs)


# ---------------------------------------------------------------- pooling


def test_last_token_pool_with_right_and_left_padding():
    hidden = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert last_token_pool(hidden, right).flatten().tolist() == [1.0, 7.0]
    assert last_token_pool(hidden, left).flatten().tolist() == [3.0, 7.0]


def test_output_size_of_plain_projected_and_multimodal_configs():
    class Text:
        hidden_size = 512

    class Projected(Text):
        embedding_dim = 768  # EmbeddingGemma 2: states projected 512 -> 768

    class Multimodal:
        def __init__(self, text):
            self.text_config = text

    assert output_size(tiny_qwen().config) == 32
    assert output_size(Multimodal(Text())) == 512
    assert output_size(Multimodal(Projected())) == 768


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError):
        PooledClassifier(tiny_qwen(), N_CLASSES, pooling="cls")


def test_last_token_prediction_does_not_depend_on_batch_padding(tokenizer):
    # Right padding + causal attention: real tokens never see the padding, so a
    # passage scores the same alone or next to a much longer one.
    torch.manual_seed(0)
    model = PooledClassifier(tiny_lora(), N_CLASSES, pooling="last", text_format=FORMAT)
    texts = ["w5 w6", "w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17"]
    together = predict_proba(model, tokenizer, texts, batch_size=2)
    alone = np.vstack([predict_proba(model, tokenizer, [t], batch_size=1) for t in texts])
    np.testing.assert_allclose(together, alone, atol=1e-5)


# ---------------------------------------------------------------- text format


@pytest.mark.parametrize("tokenizer_adds_eos", [False, True])
def test_prompt_is_prepended_and_end_token_appended_exactly_once(tokenizer_adds_eos):
    tok = _tokenizer(tokenizer_adds_eos)
    ids = tokenize(tok, ["w1 w2"], max_length=512, text_format=FORMAT)[0]
    assert ids == tok.convert_tokens_to_ids(["w59", "w58", "w1", "w2"]) + [EOS]


def test_truncation_keeps_the_eos_at_the_end(tokenizer):
    ids = tokenize(tokenizer, ["w1 w2 w3 w4 w5"], max_length=4, text_format=FORMAT)[0]
    assert len(ids) == 4 and ids[-1] == EOS
    assert ids[:2] == tokenizer.convert_tokens_to_ids(["w59", "w58"])


def test_end_token_is_the_named_one_not_the_tokenizers_eos():
    # Like Qwen3-Embedding: the tokenizer appends [EOS] but its eos_token is another
    # token. The sequence must end in the named token, once, with nothing after it.
    tok = _tokenizer(add_eos=True)
    tok.eos_token = "w57"
    ids = tokenize(tok, ["w1 w2"], max_length=512, text_format=FORMAT)[0]
    assert ids[-1] == EOS and ids.count(EOS) == 1
    assert tok.convert_tokens_to_ids("w57") not in ids


def test_unknown_end_token_is_rejected(tokenizer):
    with pytest.raises(ValueError):
        tokenize(tokenizer, ["w1"], max_length=512, text_format=TextFormat(end_token="<nope>"))


def test_count_truncated_includes_prompt_and_eos(tokenizer):
    # "w1 w2" is 2 tokens alone, 5 with the 2-token prompt and EOS.
    assert count_truncated(tokenizer, ["w1 w2"], max_length=4) == 0
    assert count_truncated(tokenizer, ["w1 w2"], max_length=4, text_format=FORMAT) == 1


def test_default_format_is_the_plain_e003_tokenization(tokenizer):
    plain = tokenizer(["w1 w2 w3"], truncation=True, max_length=2)["input_ids"]
    assert tokenize(tokenizer, ["w1 w2 w3"], max_length=2) == plain


def test_prediction_uses_the_models_own_text_format(tokenizer):
    torch.manual_seed(0)
    with_format = PooledClassifier(tiny_qwen(), N_CLASSES, pooling="last", text_format=FORMAT)
    plain = PooledClassifier(tiny_qwen(), N_CLASSES, pooling="last",
                             text_format=TextFormat(end_token="[EOS]"))
    plain.load_state_dict(with_format.state_dict())
    texts = ["w5 w6", "w7"]
    np.testing.assert_allclose(
        predict_proba(with_format, tokenizer, texts),
        predict_proba(plain, tokenizer, [FORMAT.prompt + t for t in texts]), atol=1e-6)


# ---------------------------------------------------------------- LoRA


def test_lora_trains_only_adapters_and_head(tokenizer):
    texts, labels = synthetic(n_per_class=2)
    model, _ = train_lora(texts, labels, EncoderConfig(lr=1e-2, epochs=1, batch_size=4),
                          tok=tokenizer)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or n.startswith("head.") for n in trainable)
    assert {"head.weight", "head.bias"} <= trainable


def test_lora_base_weights_do_not_move_during_training(tokenizer):
    texts, labels = synthetic(n_per_class=2)
    snapshots = []

    def snapshot(epoch, model):
        snapshots.append({n: p.detach().clone() for n, p in model.backbone.named_parameters()})

    train_lora(texts, labels, EncoderConfig(lr=1e-2, epochs=3, batch_size=4),
               tok=tokenizer, on_epoch_end=snapshot)
    first, last = snapshots[0], snapshots[-1]
    for name in first:
        if "lora_" in name:
            continue
        assert torch.equal(first[name], last[name]), name
    assert any(not torch.equal(first[n], last[n]) for n in first if "lora_" in n)


def test_tiny_lora_model_memorizes_separable_texts(tokenizer):
    texts, labels = synthetic()
    model, history = train_lora(texts, labels, EncoderConfig(lr=1e-2, epochs=25, batch_size=8),
                                tok=tokenizer)
    preds = predict_proba(model, tokenizer, texts).argmax(axis=1)
    assert (preds == labels).mean() == 1.0
    assert history[-1] < 0.5 * history[0]


# ---------------------------------------------------------------- micro-batches


def test_micro_batches_give_the_same_gradient_as_the_whole_batch(tokenizer):
    # Compare gradients, not weights after AdamW: LoRA B starts at zero and Adam's
    # first step is ~lr * sign(grad), which turns 1e-9 rounding noise into full steps.
    texts, labels = synthetic(n_per_class=4)
    ids = tokenize(tokenizer, texts, 512, FORMAT)
    y = torch.tensor(labels)
    batch = list(range(len(texts)))
    torch.manual_seed(0)
    model = PooledClassifier(tiny_lora(), N_CLASSES, dropout=0.0, pooling="last",
                             text_format=FORMAT)
    loss_fn = torch.nn.CrossEntropyLoss()

    def grads(micro):
        model.zero_grad(set_to_none=True)
        loss = accumulate_gradients(model, ids, y, batch, micro, PAD, loss_fn,
                                    torch.device("cpu"), bf16=False)
        return loss, {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}

    loss_whole, g_whole = grads(len(batch))
    loss_split, g_split = grads(4)
    assert loss_split == pytest.approx(loss_whole, rel=1e-6)
    assert g_whole.keys() == g_split.keys() and len(g_whole) > 2
    for name in g_whole:
        torch.testing.assert_close(g_split[name], g_whole[name], atol=1e-6, rtol=1e-5, msg=name)


def test_micro_batches_train_to_the_same_loss_curve(tokenizer):
    texts, labels = synthetic(n_per_class=4)
    base = dict(lr=1e-2, epochs=2, batch_size=8, dropout=0.0)
    _, h_whole = train_lora(texts, labels, EncoderConfig(**base), tok=tokenizer)
    _, h_split = train_lora(texts, labels, EncoderConfig(**base, micro_batch_size=2),
                            tok=tokenizer)
    np.testing.assert_allclose(h_split, h_whole, rtol=1e-3)


def test_micro_batch_larger_than_batch_is_rejected(tokenizer):
    texts, labels = synthetic(n_per_class=1)
    with pytest.raises(ValueError):
        train_lora(texts, labels, EncoderConfig(lr=1e-2, epochs=1, batch_size=4,
                                                micro_batch_size=8), tok=tokenizer)


# ---------------------------------------------------------------- early stop on the schedule


def test_lora_stopping_at_epoch_k_on_the_6_epoch_schedule_reproduces_epoch_k(tokenizer):
    # What E004 relies on: eval runs train k* epochs of the 6-epoch schedule.
    texts, labels = synthetic()
    probe = texts[:6]
    snapshots = {}

    def snapshot(epoch, model):
        snapshots[epoch] = predict_proba(model, tokenizer, probe)

    _, long_history = train_lora(texts, labels, EncoderConfig(lr=1e-2, epochs=6, batch_size=4),
                                 tok=tokenizer, on_epoch_end=snapshot)
    short, short_history = train_lora(
        texts, labels, EncoderConfig(lr=1e-2, epochs=3, schedule_epochs=6, batch_size=4),
        tok=tokenizer)
    np.testing.assert_allclose(predict_proba(short, tokenizer, probe), snapshots[3], atol=1e-6)
    assert short_history == long_history[:4]


# ---------------------------------------------------------------- loading


def test_backbone_factory_passes_dtype_and_load_kwargs(monkeypatch):
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return "backbone"

    monkeypatch.setattr(encoder, "AutoModel", FakeAutoModel)
    arm = EncoderArm("org/model", "abc", dtype=torch.bfloat16,
                     load_kwargs={"vision_config": None, "audio_config": None})
    assert backbone_factory(arm)() == "backbone"
    assert calls == [("org/model", {"revision": "abc", "dtype": torch.bfloat16,
                                    "vision_config": None, "audio_config": None})]


def test_gradient_checkpointing_is_enabled_only_when_asked():
    calls = []

    class FakeBackbone:
        def gradient_checkpointing_enable(self, **kwargs):
            calls.append(kwargs)

    wrap_backbone(FakeBackbone(), EncoderArm("m", "r"))
    assert calls == []
    wrap_backbone(FakeBackbone(), EncoderArm("m", "r", gradient_checkpointing=True))
    assert calls == [{"gradient_checkpointing_kwargs": {"use_reentrant": False}}]


def test_e003_arms_keep_their_original_recipe():
    for arm in ARMS.values():
        assert arm.pooling == "mean" and not arm.normalize
        assert arm.text_format == TextFormat()
        assert arm.dtype == torch.float32
        assert arm.lora is None and not arm.gradient_checkpointing
        assert dict(arm.load_kwargs) == {}


# ---------------------------------------------------------------- normalization


def test_normalized_embedding_has_norm_sqrt_dim_and_bounded_logits(tokenizer):
    torch.manual_seed(0)
    model = PooledClassifier(tiny_qwen(), N_CLASSES, dropout=0.0, pooling="last",
                             text_format=FORMAT, normalize=True)
    captured = {}
    model.head.register_forward_hook(lambda m, inp, out: captured.update(x=inp[0]))
    ids = torch.tensor(tokenize(tokenizer, ["w5 w6 w7"], 512, FORMAT))
    logits = model(ids, torch.ones_like(ids))
    dim = captured["x"].shape[-1]
    torch.testing.assert_close(captured["x"].norm(dim=-1), torch.full((1,), dim ** 0.5))
    bound = dim ** 0.5 * model.head.weight.norm(dim=1) + model.head.bias.abs()  # Cauchy-Schwarz
    assert (logits.abs()[0] <= bound + 1e-6).all()


def test_normalization_is_off_by_default_and_passed_through_training(tokenizer):
    assert not PooledClassifier(tiny_qwen(), N_CLASSES).normalize
    texts, labels = synthetic(n_per_class=1)
    model, _ = train_lora(texts, labels, EncoderConfig(lr=1e-2, epochs=1, batch_size=4),
                          tok=tokenizer, normalize=True)
    assert model.normalize
