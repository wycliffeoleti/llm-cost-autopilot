from pathlib import Path

from costpilot.classifier import load_dataset, train_classifier
from costpilot.providers.fake import FAKE_MODELS
from costpilot.routing import classify_and_route, load_routing_config

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"
ROUTING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"


def test_classify_and_route_end_to_end_per_tier():
    dataset = load_dataset(DATASET_PATH)
    model = train_classifier(dataset.examples)
    config = load_routing_config(ROUTING_CONFIG_PATH)

    assert classify_and_route("What is the capital of Germany?", model, config) == FAKE_MODELS[
        "claude-haiku"
    ]

    complex_prompt = (
        "Analyze and compare these two hiring strategies, considering at least "
        "three trade-offs, and recommend one with justification."
    )
    assert classify_and_route(complex_prompt, model, config) == FAKE_MODELS["gpt-4o"]
