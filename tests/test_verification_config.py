from pathlib import Path

import pytest

from costpilot.providers.fake import FAKE_MODELS
from costpilot.verification import load_verification_config

CONFIG_PATH = Path(__file__).parent.parent / "config" / "verification.yaml"


def test_load_verification_config_returns_known_reference_model_and_default_threshold():
    config = load_verification_config(CONFIG_PATH)
    assert config.reference_model == FAKE_MODELS["gpt-4o"]
    assert config.default_threshold == 1.0


@pytest.mark.parametrize(
    "contents",
    [
        "reference_model_id: no-such-model\nthresholds:\n  default: 1.0\n",
        "reference_model_id: gpt-4o\nthresholds:\n  default: 1.1\n",
        "reference_model_id: gpt-4o\nthresholds:\n  default: not-a-number\n",
    ],
)
def test_load_verification_config_rejects_invalid_values(tmp_path, contents):
    path = tmp_path / "verification.yaml"
    path.write_text(contents)
    with pytest.raises((TypeError, ValueError)):
        load_verification_config(path)
