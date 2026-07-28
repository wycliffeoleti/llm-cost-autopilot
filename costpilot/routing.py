from __future__ import annotations

from pathlib import Path

import yaml

from costpilot.domain import ModelConfig
from costpilot.providers.fake import FAKE_MODELS


def load_routing_config(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text())
    return dict(raw)


def route(tier: str, config: dict[str, str]) -> ModelConfig:
    if tier not in config:
        raise ValueError(f"No routing entry for tier {tier!r}")
    model_id = config[tier]
    if model_id not in FAKE_MODELS:
        raise ValueError(f"Routing config maps {tier!r} to unknown model {model_id!r}")
    return FAKE_MODELS[model_id]
