import json
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"
VALID_TIERS = {"tier_1", "tier_2", "tier_3"}


def _load_raw():
    return json.loads(DATASET_PATH.read_text())


def test_dataset_file_has_required_top_level_fields():
    raw = _load_raw()
    assert raw["status"] == "ai_drafted_pending_human_review"
    assert isinstance(raw["examples"], list)
    assert len(raw["examples"]) > 0


def test_dataset_examples_have_valid_tiers_and_unique_ids():
    raw = _load_raw()
    ids = [example["id"] for example in raw["examples"]]
    assert len(ids) == len(set(ids)), "duplicate example id found"
    for example in raw["examples"]:
        assert example["tier"] in VALID_TIERS, f"invalid tier for {example['id']}"
        assert example["text"].strip(), f"empty text for {example['id']}"


def test_dataset_has_at_least_210_examples_balanced_across_tiers():
    raw = _load_raw()
    assert len(raw["examples"]) >= 210
    counts = Counter(example["tier"] for example in raw["examples"])
    for tier in VALID_TIERS:
        assert 60 <= counts[tier] <= 80, f"{tier} has {counts[tier]} examples, expected ~70"
