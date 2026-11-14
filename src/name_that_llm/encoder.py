"""Fine-tuned transformer encoder for model attribution (E003 #5, E004 #7).

Recipe from L001 (Sun et al., 2025): a pretrained encoder, pooling over its
token states, dropout and a linear head. E003 trains everything with masked
mean pooling; E004 adds last-token pooling, task prompts and LoRA adapters.
The defaults of `EncoderArm` reproduce E003 exactly.
No I/O beyond loading pretrained weights from the Hugging Face cache.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup


@dataclass(frozen=True)
class LoraSpec:
    """LoRA adapters on the backbone; the backbone itself stays frozen."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | tuple[str, ...] = "all-linear"


@dataclass(frozen=True)
class TextFormat:
    """How a raw passage becomes token ids. Stored on the classifier, so training
    and prediction always format text the same way."""

    prompt: str = ""  # prepended to every passage
    # Token every sequence must end with, exactly once (last-token pooling). Named
    # explicitly: Qwen3-Embedding pools on <|endoftext|>, not on its eos_token <|im_end|>.
    end_token: str | None = None


@dataclass(frozen=True)
class EncoderArm:
    name: str
    revision: str  # pinned hub commit, for reproducibility
    pooling: str = "mean"  # "mean" or "last" (last real token)
    # L2-normalize the pooled embedding (as embedding models do), rescaled to norm sqrt(dim)
    normalize: bool = False
    text_format: TextFormat = TextFormat()
    dtype: torch.dtype = torch.float32  # weights as loaded; bf16 compute is via autocast
    lora: LoraSpec | None = None  # None: full fine-tuning
    gradient_checkpointing: bool = False
    load_kwargs: Mapping = field(default_factory=dict)  # extra from_pretrained kwargs


ARMS = {
    "A": EncoderArm("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
    "B": EncoderArm("Alibaba-NLP/gte-modernbert-base", "e7f32e3c00f91d699e8c43b53106206bcc72bb22"),
}


@dataclass(frozen=True)
class EncoderConfig:
    """Training recipe. `epochs` is how long to train; `schedule_epochs` (default:
    `epochs`) sets the length of the warmup + linear-decay schedule, so a run can
    stop early on the same schedule as a longer one.
    """

    lr: float
    epochs: int
    schedule_epochs: int | None = None
    batch_size: int = 16  # examples per optimizer step
    micro_batch_size: int | None = None  # examples per forward pass; None: batch_size
    max_length: int = 512
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    seed: int = 0
    bf16: bool = True  # only used on CUDA

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `hidden` (batch, seq, dim) over positions where `mask` is 1."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """State of the last position where `mask` is 1, for left or right padding."""
    last = mask.shape[1] - 1 - mask.flip(dims=[1]).argmax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


POOLING = {"mean": mean_pool, "last": last_token_pool}


def output_size(config) -> int:
    """Width of the backbone's `last_hidden_state`.

    Usually `hidden_size`. EmbeddingGemma 2 projects its 512-wide text states to
    `text_config.embedding_dim` (768) before returning them; multimodal configs
    keep both fields under `text_config`.
    """
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is not None and getattr(cfg, "embedding_dim", None) is not None:
            return cfg.embedding_dim
    size = getattr(config, "hidden_size", None)
    return size if size is not None else config.text_config.hidden_size


class PooledClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, dropout: float = 0.1,
                 pooling: str = "mean", text_format: TextFormat = TextFormat(),
                 normalize: bool = False):
        super().__init__()
        if pooling not in POOLING:
            raise ValueError(f"unknown pooling {pooling!r}")
        self.backbone = backbone
        self.pooling = pooling
        self.normalize = normalize
        self.text_format = text_format
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(output_size(backbone.config), n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = POOLING[self.pooling](hidden, attention_mask)
        if self.normalize:
            # Direction only, at RMS 1 per dimension (the scale of a LayerNorm output).
            # Unit norm alone caps the logits at the head's weight norm: in E004's
            # check the loss stayed at ~2.5 (chance 2.89) after 30 epochs.
            pooled = nn.functional.normalize(pooled.float(), dim=-1) * pooled.shape[-1] ** 0.5
        return self.head(self.dropout(pooled))


MeanPoolClassifier = PooledClassifier  # E003 name; mean pooling is the default


def load_tokenizer(arm: EncoderArm) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(arm.name, revision=arm.revision)


def backbone_factory(arm: EncoderArm) -> Callable[[], nn.Module]:
    """A function that returns fresh pretrained weights of `arm` on each call.

    Weights load in `arm.dtype` (float32 by default). Without an explicit dtype,
    transformers loads the checkpoint's own: gte-modernbert-base is stored in
    float16, and AdamW on float16 weights diverged to NaN in the E003 preflight.
    With `arm.lora`, the frozen base can be bf16 because only the float32
    adapters and head are optimized.
    """
    def make() -> nn.Module:
        backbone = AutoModel.from_pretrained(arm.name, revision=arm.revision, dtype=arm.dtype,
                                             **arm.load_kwargs)
        return wrap_backbone(backbone, arm)
    return make


def wrap_backbone(backbone, arm: EncoderArm):
    """Gradient checkpointing and LoRA adapters, as `arm` asks; unchanged otherwise."""
    if arm.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if arm.lora is None:
        return backbone
    from peft import LoraConfig, get_peft_model  # GPU-group dependency, E004 only

    spec = arm.lora
    target = spec.target_modules if isinstance(spec.target_modules, str) else list(spec.target_modules)
    config = LoraConfig(r=spec.r, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                        target_modules=target, bias="none")
    return get_peft_model(backbone, config)  # adapters are float32 (peft default)


def _encode(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str],
            text_format: TextFormat, max_length: int | None) -> list:
    """Token ids per formatted text; truncated to `max_length` (end token included) if given."""
    texts = [text_format.prompt + t for t in texts]
    if text_format.end_token is None:
        kwargs = {"truncation": True, "max_length": max_length} if max_length else {}
        return tokenizer(texts, **kwargs)["input_ids"]
    end = end_token_id(tokenizer, text_format)
    ids = [x[:-1] if x and x[-1] == end else x for x in tokenizer(texts)["input_ids"]]
    if max_length:
        ids = [x[:max_length - 1] for x in ids]
    return [x + [end] for x in ids]  # exactly once, whether or not the tokenizer adds it


def end_token_id(tokenizer: PreTrainedTokenizerBase, text_format: TextFormat) -> int:
    token_id = tokenizer.convert_tokens_to_ids(text_format.end_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"end token {text_format.end_token!r} is not in the vocabulary")
    return token_id


def tokenize(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], max_length: int,
             text_format: TextFormat = TextFormat()) -> list:
    """Token ids per text, formatted and truncated to `max_length`, without padding."""
    return _encode(tokenizer, texts, text_format, max_length)


def count_truncated(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], max_length: int,
                    text_format: TextFormat = TextFormat()) -> int:
    lengths = [len(ids) for ids in _encode(tokenizer, texts, text_format, None)]
    return int(sum(n > max_length for n in lengths))


def _batches(
    ids: list, order: Sequence[int], batch_size: int, pad_id: int
) -> Iterator[tuple[list[int], torch.Tensor, torch.Tensor]]:
    """Dynamically padded batches of `ids` taken in `order`."""
    for start in range(0, len(order), batch_size):
        idx = list(order[start:start + batch_size])
        width = max(len(ids[i]) for i in idx)
        input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            input_ids[row, :len(ids[i])] = torch.tensor(ids[i])
            mask[row, :len(ids[i])] = 1
        yield idx, input_ids, mask


def _autocast(device: torch.device, bf16: bool):
    """bf16 autocast on CUDA when requested; a no-op elsewhere."""
    enabled = device.type == "cuda" and bf16
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def accumulate_gradients(
    model: nn.Module, ids: list, y: torch.Tensor, batch: Sequence[int], micro_batch_size: int,
    pad_id: int, loss_fn: nn.Module, device: torch.device, bf16: bool,
) -> float:
    """Backward pass of one batch, in micro-batches; returns the batch's mean loss.

    Each micro-batch loss is weighted by its share of the batch, so the summed
    gradient equals the gradient of the whole-batch mean loss.
    """
    batch_loss = 0.0
    for idx, input_ids, mask in _batches(ids, batch, micro_batch_size, pad_id):
        with _autocast(device, bf16):
            logits = model(input_ids.to(device), mask.to(device))
        loss = loss_fn(logits.float(), y[idx].to(device))
        if len(idx) < len(batch):
            loss = loss * (len(idx) / len(batch))
        loss.backward()
        batch_loss += loss.item()
    return batch_loss


def train_classifier(
    texts: Sequence[str],
    labels: Sequence[int],
    n_classes: int,
    config: EncoderConfig,
    tokenizer: PreTrainedTokenizerBase,
    make_backbone: Callable[[], nn.Module],
    device: torch.device | str = "cpu",
    on_epoch_end: Callable[[int, PooledClassifier], None] | None = None,
    pooling: str = "mean",
    text_format: TextFormat = TextFormat(),
    normalize: bool = False,
) -> tuple[PooledClassifier, list[float]]:
    """Train a fresh classifier. Returns it with its loss history.

    `loss_history[0]` is the loss of the first batch before any update; the
    rest are the mean training loss of each epoch. `on_epoch_end(epoch, model)`
    is called after each epoch (1-based), e.g. to score a validation set; it
    does not change the training run.

    Only parameters with `requires_grad` are optimized and clipped: all of them
    in full fine-tuning, the adapters and the head with LoRA. A batch split into
    micro-batches gets the same gradient as the whole batch (up to padding and
    dropout noise), so `micro_batch_size` only trades speed for memory.
    """
    device = torch.device(device)
    set_seed(config.seed)
    model = PooledClassifier(make_backbone(), n_classes, config.dropout, pooling,
                             text_format, normalize).to(device)
    ids = tokenize(tokenizer, texts, config.max_length, text_format)
    y = torch.tensor(np.asarray(labels), dtype=torch.long)
    micro = config.micro_batch_size or config.batch_size
    if micro > config.batch_size:
        raise ValueError("micro_batch_size must be at most batch_size")

    steps_per_epoch = -(-len(ids) // config.batch_size)
    if config.schedule_epochs is not None and config.schedule_epochs < config.epochs:
        raise ValueError("schedule_epochs must be at least epochs")
    total_steps = steps_per_epoch * (config.schedule_epochs or config.epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_ratio * total_steps), total_steps
    )
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)

    history: list[float] = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(ids), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = accumulate_gradients(model, ids, y, batch, micro, tokenizer.pad_token_id,
                                              loss_fn, device, config.bf16)
            if not history:
                history.append(batch_loss)
            nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(batch_loss)
        history.append(float(np.mean(epoch_losses)))
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
            model.train()
    return model, history


@torch.no_grad()
def predict_proba(
    model: PooledClassifier,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int = 512,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    bf16: bool = True,
) -> np.ndarray:
    """Class probabilities (n_texts, n_classes), rows in the order of `texts`.

    Texts are formatted with the model's own `text_format`, as in training.
    """
    device = torch.device(device)
    model.eval()
    ids = tokenize(tokenizer, texts, max_length, model.text_format)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # less padding
    probs = np.zeros((len(ids), model.head.out_features), dtype=np.float32)
    for idx, input_ids, mask in _batches(ids, order, batch_size, tokenizer.pad_token_id):
        with _autocast(device, bf16):
            logits = model(input_ids.to(device), mask.to(device))
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs
