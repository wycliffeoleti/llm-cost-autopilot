from __future__ import annotations

from pathlib import Path

import yaml


def load_routing_config(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text())
    return dict(raw)
