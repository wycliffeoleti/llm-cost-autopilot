from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

from costpilot.audit import MICRODOLLARS_PER_USD, SQLiteAuditStore

OFFLINE_BANNER = (
    "Offline deterministic prototype data. All responses, tokens, latency, "
    "verification values, and USD figures are simulated from FakeProvider and "
    "fixed constants. No live provider was called and the figures are not actual "
    "spend, answer quality, routing efficacy, or realized savings."
)


@dataclass(frozen=True)
class AuditReport:
    event_count: int
    routed_cost_microusd: int
    lifecycle_cost_microusd: int
    direct_gpt4o_cost_microusd: int
    escalation_count: int
    daily_costs: list[tuple[str, int]]
    weekly_costs: list[tuple[str, int]]
    routing_distribution: list[tuple[str, int]]
    verification_distribution: list[tuple[str, int]]

    @property
    def verification_count(self) -> int:
        return sum(count for _, count in self.verification_distribution)

    @property
    def verification_pass_rate(self) -> float | None:
        if self.verification_count == 0:
            return None
        passed = dict(self.verification_distribution).get("passed", 0)
        return passed / self.verification_count

    @property
    def escalation_rate(self) -> float | None:
        return None if self.event_count == 0 else self.escalation_count / self.event_count

    @property
    def routing_only_reduction_microusd(self) -> int:
        return self.direct_gpt4o_cost_microusd - self.routed_cost_microusd

    @property
    def end_to_end_delta_microusd(self) -> int:
        return self.direct_gpt4o_cost_microusd - self.lifecycle_cost_microusd


def build_report(store: SQLiteAuditStore) -> AuditReport:
    """Build a deterministic UTC aggregate report from the local audit database."""
    aggregates = store.report_aggregates()
    summary = aggregates["summary"][0]
    return AuditReport(
        event_count=int(summary[0]),
        routed_cost_microusd=int(summary[1]),
        lifecycle_cost_microusd=int(summary[2]),
        direct_gpt4o_cost_microusd=int(summary[3]),
        escalation_count=int(summary[4]),
        daily_costs=_cost_rows(aggregates["daily"]),
        weekly_costs=_cost_rows(aggregates["weekly"]),
        routing_distribution=_count_rows(aggregates["routing"]),
        verification_distribution=_count_rows(aggregates["verification"]),
    )


def render_text_report(report: AuditReport) -> str:
    lines = [
        "LLM Cost Autopilot — Offline Audit Report",
        OFFLINE_BANNER,
        "",
        f"Lifecycle coverage: {report.event_count} request(s)",
        f"Simulated lifecycle cost: {_format_usd(report.lifecycle_cost_microusd)}",
        f"Direct GPT-4o fake-response baseline: {_format_usd(report.direct_gpt4o_cost_microusd)}",
        (
            "Routing-only simulated reduction versus direct GPT-4o fake response: "
            f"{_format_usd(report.routing_only_reduction_microusd)}"
        ),
        (
            "End-to-end simulated delta versus direct GPT-4o fake response: "
            f"{_format_usd(report.end_to_end_delta_microusd)}"
        ),
        f"Verification pass rate: {_format_rate(report.verification_pass_rate)}",
        f"Simulated escalation rate: {_format_rate(report.escalation_rate)}",
        "",
        "UTC daily simulated lifecycle cost:",
        *_cost_lines(report.daily_costs),
        "UTC weekly simulated lifecycle cost:",
        *_cost_lines(report.weekly_costs),
        "Routing distribution:",
        *_count_lines(report.routing_distribution),
        "Fake verification distribution:",
        *_count_lines(report.verification_distribution),
        "",
        OFFLINE_BANNER,
    ]
    return "\n".join(lines) + "\n"


def render_html_report(
    report: AuditReport,
    title: str = "Offline Audit Dashboard",
    additional_disclaimer: str | None = None,
) -> str:
    """Render a self-contained report with escaped text and no remote assets."""
    sections = "".join(
        (
            _html_table("UTC daily simulated lifecycle cost", report.daily_costs, "USD"),
            _html_table("UTC weekly simulated lifecycle cost", report.weekly_costs, "USD"),
            _html_table("Routing distribution", report.routing_distribution, "Requests"),
            _html_table(
                "Fake verification distribution", report.verification_distribution, "Requests"
            ),
        )
    )
    metrics = (
        ("Lifecycle coverage", str(report.event_count)),
        ("Simulated lifecycle cost", _format_usd(report.lifecycle_cost_microusd)),
        ("Direct GPT-4o fake-response baseline", _format_usd(report.direct_gpt4o_cost_microusd)),
        (
            "Routing-only simulated reduction versus direct GPT-4o fake response",
            _format_usd(report.routing_only_reduction_microusd),
        ),
        (
            "End-to-end simulated delta versus direct GPT-4o fake response",
            _format_usd(report.end_to_end_delta_microusd),
        ),
        ("Verification pass rate", _format_rate(report.verification_pass_rate)),
        ("Simulated escalation rate", _format_rate(report.escalation_rate)),
    )
    metric_html = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in metrics
    )
    escaped_title = escape(title)
    banner = escape(OFFLINE_BANNER)
    additional_banner = (
        f'<p class="banner">{escape(additional_disclaimer)}</p>'
        if additional_disclaimer is not None
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
body {{ font-family: system-ui, sans-serif; color: #172033; background: #f6f8fb; margin: 0; }}
main {{ max-width: 960px; margin: 0 auto; padding: 2rem; }}
.banner {{ background: #fff3cd; border-left: 4px solid #b7791f; padding: 1rem; }}
dl {{ display: grid; grid-template-columns: minmax(18rem, 2fr) 1fr; gap: .5rem 1rem; }}
dt {{ font-weight: 600; }} dd {{ margin: 0; text-align: right; }}
section {{ background: white; padding: 1rem; margin: 1rem 0; border-radius: .4rem; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ padding: .45rem; text-align: left; border-bottom: 1px solid #dce1ea; }}
td:last-child {{ text-align: right; }} footer {{ margin-top: 1rem; }}
</style>
</head>
<body><main>
<h1>{escaped_title}</h1>
<p class="banner">{banner}</p>
{additional_banner}
<section><h2>Simulated lifecycle summary</h2><dl>{metric_html}</dl></section>
{sections}
<footer class="banner">{banner}</footer>
</main></body>
</html>
"""


def _cost_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, int]]:
    return [(str(label), int(value)) for label, value in rows]


def _count_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, int]]:
    return [(str(label), int(value)) for label, value in rows]


def _format_usd(microusd: int) -> str:
    return f"${microusd / MICRODOLLARS_PER_USD:,.6f} simulated"


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.1%}"


def _cost_lines(rows: list[tuple[str, int]]) -> list[str]:
    return [f"  {label}: {_format_usd(cost)}" for label, cost in rows] or ["  none"]


def _count_lines(rows: list[tuple[str, int]]) -> list[str]:
    return [f"  {label}: {count}" for label, count in rows] or ["  none"]


def _html_table(heading: str, rows: list[tuple[str, int]], value_heading: str) -> str:
    rendered_rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            escape(label),
            escape(_format_usd(value) if value_heading == "USD" else str(value)),
        )
        for label, value in rows
    ) or "<tr><td colspan=\"2\">none</td></tr>"
    return (
        f"<section><h2>{escape(heading)}</h2><table><thead><tr>"
        f"<th>Group</th><th>{escape(value_heading)}</th></tr></thead>"
        f"<tbody>{rendered_rows}</tbody></table></section>"
    )
