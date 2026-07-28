from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from costpilot.service import OfflineService


ROOT = Path(__file__).parent.parent


def _service(database_path: Path) -> OfflineService:
    return OfflineService(
        dataset_path=ROOT / "data" / "complexity_dataset.draft.json",
        routing_config_path=ROOT / "config" / "routing.yaml",
        verification_config_path=ROOT / "config" / "verification.yaml",
        audit_database_path=database_path,
    )


def test_complete_uses_supplied_utc_timestamp(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 28, 9, 0, tzinfo=UTC)
    service = _service(tmp_path / "audit.sqlite3")

    service.complete("Fixture-only prompt.", "fixed-time", timestamp=timestamp)

    assert service._store.read_all()[0].timestamp == timestamp
