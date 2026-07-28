from pathlib import Path

from costpilot.providers.fake import FAKE_MODELS
from costpilot.routing import load_routing_config

ROUTING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"


def test_load_routing_config_has_all_three_tiers():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    assert set(config.keys()) == {"tier_1", "tier_2", "tier_3"}


def test_load_routing_config_maps_to_known_models():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    for model_id in config.values():
        assert model_id in FAKE_MODELS
