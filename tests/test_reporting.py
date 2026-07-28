from datetime import UTC, datetime

from costpilot.audit import SQLiteAuditStore
from costpilot.domain import Request
from costpilot.providers.fake import FAKE_MODELS, FakeProvider
from costpilot.reporting import build_report, render_html_report, render_text_report


def _append_event(store: SQLiteAuditStore, timestamp: datetime, request_id: str) -> None:
    request = Request(prompt="<script>private prompt</script>", request_id=request_id)
    response = FakeProvider().send(request, FAKE_MODELS["claude-haiku"])
    store.append(timestamp, request, "tier_1", response)


def test_report_aggregates_utc_day_and_week_and_simulated_deltas(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.sqlite3")
    _append_event(store, datetime(2026, 7, 27, 23, 30, tzinfo=UTC), "one")
    _append_event(store, datetime(2026, 7, 28, 0, 30, tzinfo=UTC), "two")

    report = build_report(store)

    assert report.event_count == 2
    assert list(report.daily_costs) == [("2026-07-27", report.daily_costs[0][1]), ("2026-07-28", report.daily_costs[1][1])]
    assert report.routing_distribution == [("anthropic/claude-haiku", 2)]
    assert report.routing_only_reduction_microusd == (
        report.direct_gpt4o_cost_microusd - report.routed_cost_microusd
    )
    assert report.end_to_end_delta_microusd == (
        report.direct_gpt4o_cost_microusd - report.lifecycle_cost_microusd
    )


def test_reports_are_deterministic_escaped_and_repeat_the_offline_banner(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.sqlite3")
    _append_event(store, datetime(2026, 7, 28, tzinfo=UTC), "<unsafe-id>")
    report = build_report(store)

    html_first = render_html_report(report, title="<Audit & report>")
    html_second = render_html_report(report, title="<Audit & report>")
    text = render_text_report(report)

    assert html_first == html_second
    assert "&lt;Audit &amp; report&gt;" in html_first
    assert "<script>" not in html_first
    assert html_first.count("Offline deterministic prototype data.") >= 2
    assert "not actual spend" in text
