"""Fine-tuned transformer encoder for model attribution (E003, issue #5).

Recipe from L001 (Sun et al., 2025): a pretrained encoder, masked mean pooling
over its token states, dropout and a linear head, all trained end to end.
No I/O beyond loading pretrained weights from the Hugging Face cache.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup


@dataclass(frozen=True)
class EncoderArm:
    name: str
    revision: str  # pinned hub commit, for reproducibility


ARMS = {
    "A": EncoderArm("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
    "B": EncoderArm("Alibaba-NLP/gte-modernbert-base", "e7f32e3c00f91d699e8c43b53106206bcc72bb22"),
}


@dataclass(frozen=True)
class EncoderConfig:
    lr: float
    epochs: int
    batch_size: int = 16
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


class MeanPoolClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.config.hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(mean_pool(hidden, attention_mask)))


def load_tokenizer(arm: EncoderArm) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(arm.name, revision=arm.revision)


def backbone_factory(arm: EncoderArm) -> Callable[[], nn.Module]:
    """A function that returns fresh pretrained weights of `arm` on each call."""
    return lambda: AutoModel.from_pretrained(arm.name, revision=arm.revision)


def tokenize(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], max_length: int) -> list:
    """Token ids per text, truncated to `max_length`, without padding."""
    return tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]


def count_truncated(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], max_length: int) -> int:
    lengths = [len(ids) for ids in tokenizer(list(texts))["input_ids"]]
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


def train_classifier(
    texts: Sequence[str],
    labels: Sequence[int],
    n_classes: int,
    config: EncoderConfig,
    tokenizer: PreTrainedTokenizerBase,
    make_backbone: Callable[[], nn.Module],
    device: torch.device | str = "cpu",
) -> tuple[MeanPoolClassifier, list[float]]:
    """Fine-tune a fresh classifier. Returns it with its loss history.

    `loss_history[0]` is the loss of the first batch before any update; the
    rest are the mean training loss of each epoch.
    """
    device = torch.device(device)
    set_seed(config.seed)
    model = MeanPoolClassifier(make_backbone(), n_classes, config.dropout).to(device)
    ids = tokenize(tokenizer, texts, config.max_length)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    steps_per_epoch = -(-len(ids) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_ratio * total_steps), total_steps
    )
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)

    history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(ids), generator=generator).tolist()
        epoch_losses = []
        for idx, input_ids, mask in _batches(ids, order, config.batch_size, tokenizer.pad_token_id):
            with _autocast(device, config.bf16):
                logits = model(input_ids.to(device), mask.to(device))
            loss = loss_fn(logits.float(), y[idx].to(device))
            if not history:
                history.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))
    return model, history


@torch.no_grad()
def predict_proba(
    model: MeanPoolClassifier,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int = 512,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    bf16: bool = True,
) -> np.ndarray:
    """Class probabilities (n_texts, n_classes), rows in the order of `texts`."""
    device = torch.device(device)
    model.eval()
    ids = tokenize(tokenizer, texts, max_length)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # less padding
    probs = np.zeros((len(ids), model.head.out_features), dtype=np.float32)
    for idx, input_ids, mask in _batches(ids, order, batch_size, tokenizer.pad_token_id):
        with _autocast(device, bf16):
            logits = model(input_ids.to(device), mask.to(device))
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs
