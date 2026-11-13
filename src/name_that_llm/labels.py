"""Model and lab labels of the competition."""

from __future__ import annotations

# Model-name prefix -> lab. Order does not matter: prefixes do not overlap.
_LAB_BY_PREFIX = {
    "claude": "Anthropic",
    "gemini": "Google",
    "gpt": "OpenAI",
    "glm": "ZAI",
    "kimi": "Moonshot AI",
    "qwen": "Qwen",
}


def lab_of(model: str) -> str:
    """Return the lab that trained `model`; raise on an unknown model name."""
    for prefix, lab in _LAB_BY_PREFIX.items():
        if model.startswith(prefix):
            return lab
    raise ValueError(f"Unknown model, no lab mapping: {model!r}")
