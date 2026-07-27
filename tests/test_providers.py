import json
from pathlib import Path

import pytest

from costpilot.domain import Request
from costpilot.providers.fake import FAKE_MODELS, FakeProvider

PROMPTS_PATH = Path(__file__).parent.parent / "evals" / "prompts" / "phase1_baseline.json"


def _load_prompts() -> list[dict[str, str]]:
    data = json.loads(PROMPTS_PATH.read_text())
    return data["prompts"]


def test_baseline_prompt_set_has_exactly_ten_unique_prompts():
    prompts = _load_prompts()
    assert len(prompts) == 10
    assert len({p["id"] for p in prompts}) == 10
    assert all(p["text"].strip() for p in prompts)


@pytest.mark.parametrize("model_id", sorted(FAKE_MODELS.keys()))
def test_every_provider_handles_every_baseline_prompt(model_id):
    provider = FakeProvider()
    model = FAKE_MODELS[model_id]
    prompts = _load_prompts()

    for entry in prompts:
        request = Request(prompt=entry["text"], request_id=entry["id"])
        response = provider.send(request, model)

        assert response.model_id == model_id
        assert response.output_text
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms == model.avg_latency_ms
        assert response.cost_usd >= 0
