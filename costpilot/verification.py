from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from costpilot.domain import ModelConfig, Request, Response
from costpilot.ports import Provider
from costpilot.providers.fake import FAKE_MODELS, FakeProvider


@dataclass(frozen=True)
class VerificationConfig:
    reference_model: ModelConfig
    default_threshold: float


@dataclass(frozen=True)
class VerificationResult:
    original_model_id: str
    reference_model_id: str
    quality_score: float
    threshold: float
    passed: bool
    simulated: Literal[True]
    original_cost_usd: float
    reference_cost_usd: float
    escalation_cost_delta_usd: float


def load_verification_config(path: Path) -> VerificationConfig:
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("Verification config must be a mapping")

    model_id = raw.get("reference_model_id")
    if not isinstance(model_id, str) or model_id not in FAKE_MODELS:
        raise ValueError("Verification config contains an unknown reference model")
    if FAKE_MODELS[model_id].quality_tier != "high":
        raise ValueError("Verification reference model must have high quality tier")

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


def simulated_agreement_score(original_text: str, reference_text: str) -> float:
    def normalize(text: str) -> str:
        return re.sub(r"^\[[^]]+\]\s*", "", text).strip()

    return 1.0 if normalize(original_text) == normalize(reference_text) else 0.0


def verify_response(
    request: Request,
    original_response: Response,
    reference_model: ModelConfig,
    provider: Provider,
    threshold: float,
) -> VerificationResult:
    """Compare an existing fake response with one explicit fake reference run."""
    _validate_threshold(threshold)
    _validate_fake_provider(provider)
    _validate_reference_model(reference_model)
    if not original_response.simulated:
        raise ValueError("Verification requires a simulated original response")
    if original_response.model_id not in FAKE_MODELS:
        raise ValueError("Original response contains an unknown fake model")
    reference_response = provider.send(request, reference_model)
    if reference_response.model_id != reference_model.model_id:
        raise ValueError("Reference response model ID does not match reference model")

    quality_score = simulated_agreement_score(
        original_response.output_text, reference_response.output_text
    )
    return VerificationResult(
        original_model_id=original_response.model_id,
        reference_model_id=reference_response.model_id,
        quality_score=quality_score,
        threshold=threshold,
        passed=quality_score >= threshold,
        simulated=True,
        original_cost_usd=original_response.cost_usd,
        reference_cost_usd=reference_response.cost_usd,
        escalation_cost_delta_usd=reference_response.cost_usd - original_response.cost_usd,
    )


def should_escalate(result: VerificationResult) -> bool:
    """Return whether the caller should explicitly request a reference rerun."""
    return not result.passed


def rerun_with_reference(
    request: Request, reference_model: ModelConfig, provider: Provider
) -> Response:
    """Execute one explicit reference-model run without retry or persistence."""
    _validate_fake_provider(provider)
    _validate_reference_model(reference_model)
    return provider.send(request, reference_model)


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("Verification threshold must be numeric")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Verification threshold must be between 0.0 and 1.0")


def _validate_fake_provider(provider: Provider) -> None:
    if not isinstance(provider, FakeProvider):
        raise TypeError("Verification requires a FakeProvider")


def _validate_reference_model(reference_model: ModelConfig) -> None:
    if FAKE_MODELS.get(reference_model.model_id) is not reference_model:
        raise ValueError("Reference model must come from the canonical fake registry")
    if reference_model.quality_tier != "high":
        raise ValueError("Verification reference model must have high quality tier")
