from pathlib import Path

from costpilot.classifier import load_dataset, train_classifier
from costpilot.domain import ModelConfig, Request, Response
from costpilot.providers.fake import FakeProvider
from costpilot.routing import classify_and_route, load_routing_config
from costpilot.verification import (
    load_verification_config,
    rerun_with_reference,
    should_escalate,
    verify_response,
)

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"
ROUTING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"
VERIFICATION_CONFIG_PATH = Path(__file__).parent.parent / "config" / "verification.yaml"


class TrackingFakeProvider(FakeProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[Request, ModelConfig]] = []

    def send(self, request: Request, model: ModelConfig) -> Response:
        self.calls.append((request, model))
        return super().send(request, model)


def test_phase_2_route_to_phase_3_explicit_offline_escalation_flow():
    classifier = train_classifier(load_dataset(DATASET_PATH).examples)
    routing_config = load_routing_config(ROUTING_CONFIG_PATH)
    verification_config = load_verification_config(VERIFICATION_CONFIG_PATH)
    request = Request(
        prompt="What is the capital of Germany please answer concisely today now soon?",
        request_id="integration-1",
    )
    provider = TrackingFakeProvider()

    original_model = classify_and_route(request.prompt, classifier, routing_config)
    original_response = provider.send(request, original_model)
    result = verify_response(
        request,
        original_response,
        verification_config.reference_model,
        provider,
        verification_config.default_threshold,
    )

    assert result.simulated is True
    assert result.original_model_id == original_model.model_id
    assert result.reference_model_id == verification_config.reference_model.model_id
    assert len(provider.calls) == 2
    assert should_escalate(result) is True

    rerun = rerun_with_reference(request, verification_config.reference_model, provider)
    assert rerun.model_id == verification_config.reference_model.model_id
    assert len(provider.calls) == 3
