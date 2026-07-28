from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from costpilot.phase6 import FIXED_START_TIMESTAMP, iter_seeded_lifecycles, main, run_seeded_demo
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


def test_seeded_lifecycles_replay_fixture_six_times_in_fixed_order() -> None:
    lifecycles = list(iter_seeded_lifecycles())

    assert len(lifecycles) == 60
    assert [lifecycle.request_id for lifecycle in lifecycles] == [
        f"phase6-{index:04d}" for index in range(1, 61)
    ]
    assert [lifecycle.fixture_id for lifecycle in lifecycles[:10]] == [
        f"p{index:02d}" for index in range(1, 11)
    ]
    assert [lifecycle.fixture_id for lifecycle in lifecycles[10:20]] == [
        f"p{index:02d}" for index in range(1, 11)
    ]
    assert lifecycles[0].timestamp == FIXED_START_TIMESTAMP
    assert lifecycles[-1].timestamp == datetime(2026, 7, 28, 9, 59, tzinfo=UTC)


def test_seeded_demo_is_aggregate_only_reconciled_and_byte_identical(tmp_path: Path) -> None:
    first_path = tmp_path / "first.html"
    second_path = tmp_path / "second.html"

    first = run_seeded_demo(first_path)
    second = run_seeded_demo(second_path)
    first_html = first_path.read_text(encoding="utf-8")

    assert first.event_count == 60
    assert first == second
    assert first.lifecycle_cost_microusd == sum(cost for _, cost in first.daily_costs)
    assert first.lifecycle_cost_microusd == sum(cost for _, cost in first.weekly_costs)
    assert first_html == second_path.read_text(encoding="utf-8")
    assert "Offline deterministic seeded demonstration." in first_html
    assert "Offline deterministic prototype data." in first_html
    assert "Summarize the following paragraph" not in first_html
    assert "simulated response" not in first_html
    assert "prompt_hash" not in first_html
    assert re.search(r"\b[a-f0-9]{64}\b", first_html) is None


def test_phase6_command_writes_a_static_report(tmp_path: Path) -> None:
    output_path = tmp_path / "report.html"

    assert main([str(output_path)]) == 0

    assert output_path.is_file()
