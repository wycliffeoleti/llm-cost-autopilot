from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from costpilot.features import FEATURE_KEYS, extract_features

TIER_LABELS = ("tier_1", "tier_2", "tier_3")
RANDOM_SEED = 42


@dataclass(frozen=True)
class LabeledExample:
    id: str
    text: str
    tier: str


@dataclass(frozen=True)
class Dataset:
    status: str
    examples: list[LabeledExample]


def load_dataset(path: Path) -> Dataset:
    raw = json.loads(Path(path).read_text())
    status = raw["status"]
    examples: list[LabeledExample] = []
    seen_ids: set[str] = set()
    for entry in raw["examples"]:
        if entry["tier"] not in TIER_LABELS:
            raise ValueError(f"Unknown tier {entry['tier']!r} for example {entry['id']!r}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate example id {entry['id']!r}")
        seen_ids.add(entry["id"])
        examples.append(
            LabeledExample(id=entry["id"], text=entry["text"], tier=entry["tier"])
        )
    return Dataset(status=status, examples=examples)


def train_test_split_dataset(
    dataset: Dataset, seed: int = RANDOM_SEED, test_size: float = 0.2
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    labels = [example.tier for example in dataset.examples]
    train, test = train_test_split(
        dataset.examples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return list(train), list(test)


def _feature_matrix(texts: list[str]) -> np.ndarray:
    rows = [[extract_features(text)[key] for key in FEATURE_KEYS] for text in texts]
    return np.array(rows, dtype=float)


def train_classifier(
    train_examples: list[LabeledExample], seed: int = RANDOM_SEED
) -> LogisticRegression:
    features = _feature_matrix([example.text for example in train_examples])
    labels = [example.tier for example in train_examples]
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(features, labels)
    return model
