"""Reproducible aggregate-only Phase 6 offline seeded demonstration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from costpilot.reporting import AuditReport, build_report, render_html_report
from costpilot.service import OfflineService

ROOT = Path(__file__).parent.parent
FIXTURE_PATH = ROOT / "evals" / "prompts" / "phase1_baseline.json"
FIXED_START_TIMESTAMP = datetime(2026, 7, 28, 9, 0, tzinfo=UTC)
PHASE6_DISCLAIMER = (
    "Offline deterministic seeded demonstration. This artifact replays 60 request "
    "lifecycles from the repository's fixed synthetic prompt fixture using FakeProvider, "
    "fixed constants, and fixed timestamps. It makes no network or live-provider calls "
    "and incurs no provider spend. Any tokens, latency, verification results, USD figures, "
    "or deltas shown are deterministic simulated values, not measurements of real spend, "
    "answer quality, routing efficacy, throughput, reliability, or realized savings. The "
    "routing dataset remains ai_drafted_pending_human_review; its labels and derived "
    "metrics are not real-world ground truth."
)


@dataclass(frozen=True)
class SeededLifecycle:
    """A local-only fixture lifecycle; prompt text never enters public artifacts."""

    request_id: str
    fixture_id: str
    prompt: str
    timestamp: datetime


def iter_seeded_lifecycles() -> Iterator[SeededLifecycle]:
    """Yield the fixed fixture in its file order six times with fixed UTC times."""
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    prompts = fixture["prompts"]
    for index in range(60):
        item = prompts[index % len(prompts)]
        yield SeededLifecycle(
            request_id=f"phase6-{index + 1:04d}",
            fixture_id=str(item["id"]),
            prompt=str(item["text"]),
            timestamp=FIXED_START_TIMESTAMP + timedelta(minutes=index),
        )


def run_seeded_demo(output_path: Path) -> AuditReport:
    """Replay the fixed lifecycle locally and write only a static aggregate HTML report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="llm-cost-autopilot-phase6-") as temporary_directory:
        service = OfflineService(
            dataset_path=ROOT / "data" / "complexity_dataset.draft.json",
            routing_config_path=ROOT / "config" / "routing.yaml",
            verification_config_path=ROOT / "config" / "verification.yaml",
            audit_database_path=Path(temporary_directory) / "phase6.sqlite3",
        )
        for lifecycle in iter_seeded_lifecycles():
            service.complete(
                lifecycle.prompt,
                lifecycle.request_id,
                timestamp=lifecycle.timestamp,
            )
        report = build_report(service._store)
    output_path.write_text(
        render_html_report(
            report,
            title="Phase 6 — Offline Seeded Lifecycle Report",
            additional_disclaimer=PHASE6_DISCLAIMER,
        ),
        encoding="utf-8",
    )
    return report
