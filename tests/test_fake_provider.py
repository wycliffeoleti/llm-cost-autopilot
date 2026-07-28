from costpilot.domain import Request
from costpilot.ports import Provider
from costpilot.providers.fake import FAKE_MODELS, FakeProvider


def test_fake_models_registry_has_five_expected_models():
    assert set(FAKE_MODELS.keys()) == {
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet",
        "claude-haiku",
        "llama-local",
    }


def test_fake_provider_satisfies_provider_protocol():
    assert isinstance(FakeProvider(), Provider)


def test_send_returns_well_formed_response():
    provider = FakeProvider()
    request = Request(prompt="Summarize this in one sentence.", request_id="req-1")
    model = FAKE_MODELS["gpt-4o-mini"]

    response = provider.send(request, model)

    assert response.model_id == "gpt-4o-mini"
    assert response.output_text
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms == model.avg_latency_ms
    assert response.cost_usd >= 0
    assert response.simulated is True


def test_send_is_deterministic_for_same_request_and_model():
    provider = FakeProvider()
    request = Request(prompt="What is the capital of France?", request_id="req-2")
    model = FAKE_MODELS["claude-sonnet"]

    first = provider.send(request, model)
    second = provider.send(request, model)

    assert first == second


def test_send_produces_different_costs_across_models_for_same_prompt():
    provider = FakeProvider()
    request = Request(
        prompt="Explain quantum entanglement in two sentences.", request_id="req-3"
    )

    costs = {
        model_id: provider.send(request, model).cost_usd
        for model_id, model in FAKE_MODELS.items()
    }

    assert len(set(costs.values())) > 1
