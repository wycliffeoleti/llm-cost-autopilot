from costpilot.domain import ModelConfig, Request, Response


def test_model_config_holds_all_required_fields():
    config = ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        cost_per_input_token=0.0000025,
        cost_per_output_token=0.00001,
        avg_latency_ms=1100.0,
        quality_tier="high",
    )
    assert config.provider == "openai"
    assert config.model_id == "gpt-4o"
    assert config.cost_per_input_token == 0.0000025
    assert config.cost_per_output_token == 0.00001
    assert config.avg_latency_ms == 1100.0
    assert config.quality_tier == "high"


def test_request_holds_prompt_and_id():
    request = Request(prompt="Hello", request_id="req-1")
    assert request.prompt == "Hello"
    assert request.request_id == "req-1"


def test_response_holds_all_required_fields():
    response = Response(
        output_text="hi",
        input_tokens=3,
        output_tokens=5,
        latency_ms=1100.0,
        cost_usd=0.00012,
        model_id="gpt-4o",
    )
    assert response.output_text == "hi"
    assert response.input_tokens == 3
    assert response.output_tokens == 5
    assert response.latency_ms == 1100.0
    assert response.cost_usd == 0.00012
    assert response.model_id == "gpt-4o"
