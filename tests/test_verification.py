from dataclasses import replace

import pytest

from costpilot.domain import ModelConfig, Request, Response
from costpilot.providers.fake import FAKE_MODELS, FakeProvider
from costpilot.verification import simulated_agreement_score, verify_response


class SpyProvider:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.calls: list[tuple[Request, ModelConfig]] = []

    def send(self, request: Request, model: ModelConfig) -> Response:
        self.calls.append((request, model))
        return self.response


def test_simulated_agreement_ignores_fake_model_identity():
    original = "[claude-haiku] simulated response (digest=abcd1234, input_tokens=5)"
    reference = "[gpt-4o] simulated response (digest=abcd1234, input_tokens=5)"
    assert simulated_agreement_score(original, reference) == 1.0


def test_simulated_agreement_returns_zero_for_different_fake_payloads():
    original = "[claude-haiku] simulated response (digest=abcd1234, input_tokens=5)"
    reference = "[gpt-4o] simulated response (digest=deadbeef, input_tokens=5)"
    assert simulated_agreement_score(original, reference) == 0.0


def test_verify_response_calls_reference_once_and_accounts_for_costs():
    request = Request(prompt="Summarize this quarterly report.", request_id="request-1")
    original = FakeProvider().send(request, FAKE_MODELS["claude-haiku"])
    reference = FakeProvider().send(request, FAKE_MODELS["gpt-4o"])
    provider = SpyProvider(reference)

    result = verify_response(
        request, original, FAKE_MODELS["gpt-4o"], provider, threshold=1.0
    )

    assert provider.calls == [(request, FAKE_MODELS["gpt-4o"])]
    assert result.original_model_id == "claude-haiku"
    assert result.reference_model_id == "gpt-4o"
    assert result.quality_score == 1.0
    assert result.passed is True
    assert result.simulated is True
    assert result.original_cost_usd == original.cost_usd
    assert result.reference_cost_usd == reference.cost_usd
    assert result.escalation_cost_delta_usd == pytest.approx(
        reference.cost_usd - original.cost_usd
    )
    assert original == FakeProvider().send(request, FAKE_MODELS["claude-haiku"])


def test_verify_response_reports_failed_comparison_for_divergent_fake_output():
    request = Request(prompt="Summarize this quarterly report.", request_id="request-1")
    original = FakeProvider().send(request, FAKE_MODELS["claude-haiku"])
    reference = replace(
        FakeProvider().send(request, FAKE_MODELS["gpt-4o"]),
        output_text="[gpt-4o] simulated response (digest=divergent, input_tokens=5)",
    )

    result = verify_response(
        request, original, FAKE_MODELS["gpt-4o"], SpyProvider(reference), threshold=1.0
    )

    assert result.quality_score == 0.0
    assert result.passed is False


def test_verify_response_rejects_invalid_threshold_before_reference_execution():
    request = Request(prompt="Summarize this quarterly report.", request_id="request-1")
    original = FakeProvider().send(request, FAKE_MODELS["claude-haiku"])
    reference = FakeProvider().send(request, FAKE_MODELS["gpt-4o"])
    provider = SpyProvider(reference)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        verify_response(request, original, FAKE_MODELS["gpt-4o"], provider, threshold=1.1)

    assert provider.calls == []
