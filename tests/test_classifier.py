from pathlib import Path

import pytest

from costpilot.classifier import load_dataset, train_test_split_dataset

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


def test_train_test_split_covers_every_example_exactly_once():
    dataset = load_dataset(DATASET_PATH)
    train, test = train_test_split_dataset(dataset)
    assert len(train) + len(test) == len(dataset.examples)
    assert {example.id for example in train} | {example.id for example in test} == {
        example.id for example in dataset.examples
    }
    assert {example.id for example in train} & {example.id for example in test} == set()


def test_train_test_split_is_deterministic_for_a_fixed_seed():
    dataset = load_dataset(DATASET_PATH)
    train_a, test_a = train_test_split_dataset(dataset, seed=42)
    train_b, test_b = train_test_split_dataset(dataset, seed=42)
    assert [example.id for example in train_a] == [example.id for example in train_b]
    assert [example.id for example in test_a] == [example.id for example in test_b]


def test_train_test_split_is_stratified_across_tiers():
    dataset = load_dataset(DATASET_PATH)
    _, test = train_test_split_dataset(dataset)
    assert {example.tier for example in test} == {"tier_1", "tier_2", "tier_3"}
