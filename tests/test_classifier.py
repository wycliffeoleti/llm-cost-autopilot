from pathlib import Path

import pytest

from costpilot.classifier import load_dataset

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"


def test_load_dataset_returns_status_and_examples():
    dataset = load_dataset(DATASET_PATH)
    assert dataset.status == "ai_drafted_pending_human_review"
    assert len(dataset.examples) >= 210


def test_load_dataset_rejects_unknown_tier(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(
        '{"version": "1.0", "status": "ai_drafted_pending_human_review", '
        '"examples": [{"id": "x1", "text": "hello", "tier": "tier_9"}]}'
    )
    with pytest.raises(ValueError):
        load_dataset(bad_file)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    bad_file = tmp_path / "dup.json"
    bad_file.write_text(
        '{"version": "1.0", "status": "ai_drafted_pending_human_review", '
        '"examples": ['
        '{"id": "x1", "text": "hello", "tier": "tier_1"}, '
        '{"id": "x1", "text": "world", "tier": "tier_2"}'
        ']}'
    )
    with pytest.raises(ValueError):
        load_dataset(bad_file)
