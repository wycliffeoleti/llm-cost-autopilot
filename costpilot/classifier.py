from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
