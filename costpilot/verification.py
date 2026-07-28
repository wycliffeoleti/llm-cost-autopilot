from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from costpilot.domain import ModelConfig
from costpilot.providers.fake import FAKE_MODELS


@dataclass(frozen=True)
class VerificationConfig:
    reference_model: ModelConfig
    default_threshold: float


def load_verification_config(path: Path) -> VerificationConfig:
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("Verification config must be a mapping")

    model_id = raw.get("reference_model_id")
    if not isinstance(model_id, str) or model_id not in FAKE_MODELS:
        raise ValueError("Verification config contains an unknown reference model")

    thresholds = raw.get("thresholds")
    threshold = thresholds.get("default") if isinstance(thresholds, dict) else None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("Verification default threshold must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Verification default threshold must be between 0.0 and 1.0")

    return VerificationConfig(
        reference_model=FAKE_MODELS[model_id], default_threshold=threshold
    )
