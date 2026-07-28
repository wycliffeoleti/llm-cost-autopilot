from pathlib import Path

import pytest

from costpilot.providers.fake import FAKE_MODELS
from costpilot.routing import load_routing_config, route

ROUTING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"


def test_load_routing_config_has_all_three_tiers():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    assert set(config.keys()) == {"tier_1", "tier_2", "tier_3"}


def test_load_routing_config_maps_to_known_models():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    for model_id in config.values():
        assert model_id in FAKE_MODELS


def test_route_returns_the_correct_model_config_per_tier():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    assert route("tier_1", config) == FAKE_MODELS["claude-haiku"]
    assert route("tier_2", config) == FAKE_MODELS["gpt-4o-mini"]
    assert route("tier_3", config) == FAKE_MODELS["gpt-4o"]


def test_route_raises_for_unknown_tier():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    with pytest.raises(ValueError):
        route("tier_9", config)


def test_route_raises_for_unknown_model_id_in_config():
    with pytest.raises(ValueError):
        route("tier_1", {"tier_1": "not-a-real-model"})
