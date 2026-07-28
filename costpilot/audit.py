from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from costpilot.domain import Request, Response
from costpilot.providers.fake import FAKE_MODELS, FakeProvider
from costpilot.verification import VerificationResult, simulated_agreement_score

MICRODOLLARS_PER_USD = 1_000_000
VALID_CLASSIFIER_TIERS = frozenset({"tier_1", "tier_2", "tier_3"})


def _microdollars(cost_usd: float) -> int:
    if isinstance(cost_usd, bool) or not isinstance(cost_usd, (int, float)):
        raise TypeError("Cost must be numeric")
    if not math.isfinite(cost_usd) or cost_usd < 0:
        raise ValueError("Cost must be finite and non-negative")
    return round(cost_usd * MICRODOLLARS_PER_USD)


def _validate_response(response: Response, field: str) -> None:
    if not response.simulated:
        raise ValueError(f"{field} response must be simulated")
    if response.model_id not in FAKE_MODELS:
        raise ValueError(f"{field} response model must be a canonical fake model")
    if (
        isinstance(response.input_tokens, bool)
        or isinstance(response.output_tokens, bool)
        or not isinstance(response.input_tokens, int)
        or not isinstance(response.output_tokens, int)
        or response.input_tokens < 0
        or response.output_tokens < 0
    ):
        raise ValueError(f"{field} response tokens must be non-negative integers")
    if (
        isinstance(response.latency_ms, bool)
        or not isinstance(response.latency_ms, (int, float))
        or not math.isfinite(response.latency_ms)
        or response.latency_ms < 0
    ):
        raise ValueError(f"{field} response latency must be finite and non-negative")
    _microdollars(response.cost_usd)


@dataclass(frozen=True)
class AuditEvent:
    """One append-only, fully simulated request lifecycle.

    Cost fields are rounded to integer microdollars for each simulated provider
    invocation before the lifecycle total is calculated. This makes persisted
    and aggregated totals exact without storing binary floating-point currency.
    """

    timestamp: datetime
    request_id: str
    prompt_hash: str
    classifier_tier: str
    routed_provider: str
    routed_model_id: str
    routed_input_tokens: int
    routed_output_tokens: int
    routed_latency_ms: float
    routed_cost_microusd: int
    verification_model_id: str | None
    verification_quality_score: float | None
    verification_threshold: float | None
    verification_passed: bool | None
    verification_cost_microusd: int
    escalated: bool
    rerun_model_id: str | None
    rerun_input_tokens: int | None
    rerun_output_tokens: int | None
    rerun_latency_ms: float | None
    rerun_cost_microusd: int
    lifecycle_cost_microusd: int
    direct_gpt4o_cost_microusd: int
    def __post_init__(self) -> None:
        offset = self.timestamp.utcoffset()
        if self.timestamp.tzinfo is None or offset is None:
            raise ValueError("Audit timestamp must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("Audit timestamp must be UTC")
        if not self.request_id:
            raise ValueError("Audit request ID must not be empty")
        if len(self.prompt_hash) != 64 or any(char not in "0123456789abcdef" for char in self.prompt_hash):
            raise ValueError("Audit prompt hash must be 64 lowercase hexadecimal characters")
        if self.classifier_tier not in VALID_CLASSIFIER_TIERS:
            raise ValueError("Audit classifier tier is invalid")
        model = FAKE_MODELS.get(self.routed_model_id)
        if model is None or self.routed_provider != model.provider:
            raise ValueError("Audit routed model/provider must be canonical")
        _validate_fields(
            self.routed_input_tokens,
            self.routed_output_tokens,
            self.routed_latency_ms,
            self.routed_cost_microusd,
            "routed",
        )
        if type(self.escalated) is not bool:
            raise ValueError("Audit escalated must be a bool")
        if self.verification_model_id is None:
            if any(
                value is not None
                for value in (
                    self.verification_quality_score,
                    self.verification_threshold,
                    self.verification_passed,
                )
            ) or self.verification_cost_microusd != 0:
                raise ValueError("Audit verification fields must be absent together")
        else:
            if self.verification_model_id not in FAKE_MODELS:
                raise ValueError("Audit verification model must be canonical")
            if FAKE_MODELS[self.verification_model_id].quality_tier != "high":
                raise ValueError("Audit verification model must have high quality tier")
            if not all(
                value is not None
                for value in (
                    self.verification_quality_score,
                    self.verification_threshold,
                    self.verification_passed,
                )
            ):
                raise ValueError("Audit verification fields must be present together")
            assert self.verification_quality_score is not None
            assert self.verification_threshold is not None
            assert self.verification_passed is not None
            if (
                isinstance(self.verification_quality_score, bool)
                or isinstance(self.verification_threshold, bool)
                or not math.isfinite(self.verification_quality_score)
                or not math.isfinite(self.verification_threshold)
                or not 0.0 <= self.verification_quality_score <= 1.0
                or not 0.0 <= self.verification_threshold <= 1.0
            ):
                raise ValueError("Audit verification values must be finite values from 0.0 to 1.0")
            if type(self.verification_passed) is not bool:
                raise ValueError("Audit verification passed must be a bool")
            if self.verification_passed != (
                self.verification_quality_score >= self.verification_threshold
            ):
                raise ValueError("Audit verification pass state must match its quality threshold")
            _validate_microdollars(self.verification_cost_microusd, "verification")
            if self.verification_cost_microusd == 0:
                raise ValueError("Audit verification cost must be positive when verification is present")
        if self.escalated != (self.rerun_model_id is not None):
            raise ValueError("Audit escalation state must match rerun fields")
        if self.rerun_model_id is None:
            if any(
                value is not None
                for value in (
                    self.rerun_input_tokens,
                    self.rerun_output_tokens,
                    self.rerun_latency_ms,
                )
            ) or self.rerun_cost_microusd != 0:
                raise ValueError("Audit rerun fields must be absent together")
        else:
            if self.rerun_model_id not in FAKE_MODELS:
                raise ValueError("Audit rerun model must be canonical")
            if self.verification_passed is not False:
                raise ValueError("Audit rerun requires a failed simulated verification")
            if self.rerun_model_id != self.verification_model_id:
                raise ValueError("Audit rerun model must match the verification model")
            assert self.rerun_input_tokens is not None
            assert self.rerun_output_tokens is not None
            assert self.rerun_latency_ms is not None
            _validate_fields(
                self.rerun_input_tokens,
                self.rerun_output_tokens,
                self.rerun_latency_ms,
                self.rerun_cost_microusd,
                "rerun",
            )
        _validate_microdollars(self.direct_gpt4o_cost_microusd, "direct gpt-4o")
        if self.lifecycle_cost_microusd != (
            self.routed_cost_microusd
            + self.verification_cost_microusd
            + self.rerun_cost_microusd
        ):
            raise ValueError("Audit lifecycle total must reconcile exactly")

    @classmethod
    def from_lifecycle(
        cls,
        timestamp: datetime,
        request: Request,
        classifier_tier: str,
        routed_response: Response,
        *,
        verification: VerificationResult | None = None,
        verification_response: Response | None = None,
        rerun_response: Response | None = None,
    ) -> AuditEvent:
        """Construct a validated event without persisting prompt or output text."""
        offset = timestamp.utcoffset()
        if timestamp.tzinfo is None or offset is None:
            raise ValueError("Audit timestamp must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("Audit timestamp must be UTC")
        _validate_deterministic_response(request, routed_response, "Routed")
        if rerun_response is not None:
            _validate_deterministic_response(request, rerun_response, "Rerun")
        if verification is None:
            if verification_response is not None or rerun_response is not None:
                raise ValueError("Audit verification data requires a verification result")
        else:
            if verification.simulated is not True:
                raise ValueError("Audit verification result must be simulated")
            if verification_response is None:
                raise ValueError("Audit verification response is required")
            _validate_deterministic_response(request, verification_response, "Verification")
            if (
                verification.original_model_id != routed_response.model_id
                or verification.reference_model_id != verification_response.model_id
                or _microdollars(verification.original_cost_usd)
                != _microdollars(routed_response.cost_usd)
                or _microdollars(verification.reference_cost_usd)
                != _microdollars(verification_response.cost_usd)
            ):
                raise ValueError("Audit verification provenance does not match the simulated responses")
            if FAKE_MODELS[verification_response.model_id].quality_tier != "high":
                raise ValueError("Audit verification model must have high quality tier")
            if type(verification.passed) is not bool:
                raise ValueError("Audit verification result passed state must be a bool")
            expected_score = simulated_agreement_score(
                routed_response.output_text, verification_response.output_text
            )
            if (
                verification.quality_score != expected_score
                or verification.passed != (expected_score >= verification.threshold)
                or _microdollars(verification.escalation_cost_delta_usd)
                != _microdollars(verification_response.cost_usd)
                - _microdollars(routed_response.cost_usd)
            ):
                raise ValueError("Audit verification result does not match simulated lifecycle data")
            if (
                rerun_response is not None
                and rerun_response.model_id != verification_response.model_id
            ):
                raise ValueError("Audit rerun must use the verification model")
        direct_response = FakeProvider().send(request, FAKE_MODELS["gpt-4o"])
        utc_timestamp = timestamp.astimezone(UTC)
        verification_cost = 0 if verification_response is None else _microdollars(verification_response.cost_usd)
        rerun_cost = 0 if rerun_response is None else _microdollars(rerun_response.cost_usd)
        event = cls(
            timestamp=utc_timestamp,
            request_id=request.request_id,
            prompt_hash=hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            classifier_tier=classifier_tier,
            routed_provider=FAKE_MODELS[routed_response.model_id].provider,
            routed_model_id=routed_response.model_id,
            routed_input_tokens=routed_response.input_tokens,
            routed_output_tokens=routed_response.output_tokens,
            routed_latency_ms=routed_response.latency_ms,
            routed_cost_microusd=_microdollars(routed_response.cost_usd),
            verification_model_id=None if verification_response is None else verification_response.model_id,
            verification_quality_score=None if verification is None else verification.quality_score,
            verification_threshold=None if verification is None else verification.threshold,
            verification_passed=None if verification is None else verification.passed,
            verification_cost_microusd=verification_cost,
            escalated=rerun_response is not None,
            rerun_model_id=None if rerun_response is None else rerun_response.model_id,
            rerun_input_tokens=None if rerun_response is None else rerun_response.input_tokens,
            rerun_output_tokens=None if rerun_response is None else rerun_response.output_tokens,
            rerun_latency_ms=None if rerun_response is None else rerun_response.latency_ms,
            rerun_cost_microusd=rerun_cost,
            lifecycle_cost_microusd=(
                _microdollars(routed_response.cost_usd) + verification_cost + rerun_cost
            ),
            direct_gpt4o_cost_microusd=_microdollars(direct_response.cost_usd),
        )
        return event


def _validate_microdollars(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Audit {field} cost must be a non-negative integer microdollar amount")


def _validate_deterministic_response(request: Request, response: Response, field: str) -> None:
    _validate_response(response, field)
    expected = FakeProvider().send(request, FAKE_MODELS[response.model_id])
    if response != expected:
        raise ValueError(f"{field} response does not match the supplied request and canonical fake model")


def _validate_fields(
    input_tokens: int, output_tokens: int, latency_ms: float, cost_microusd: int, field: str
) -> None:
    if (
        isinstance(input_tokens, bool)
        or isinstance(output_tokens, bool)
        or not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or input_tokens < 0
        or output_tokens < 0
    ):
        raise ValueError(f"Audit {field} tokens must be non-negative integers")
    if (
        isinstance(latency_ms, bool)
        or not isinstance(latency_ms, (int, float))
        or not math.isfinite(latency_ms)
        or latency_ms < 0
    ):
        raise ValueError(f"Audit {field} latency must be finite and non-negative")
    _validate_microdollars(cost_microusd, field)


class SQLiteAuditStore:
    """Local SQLite event store with database-enforced append-only records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    insertion_id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    request_id TEXT NOT NULL UNIQUE,
                    prompt_hash TEXT NOT NULL,
                    classifier_tier TEXT NOT NULL,
                    routed_provider TEXT NOT NULL,
                    routed_model_id TEXT NOT NULL,
                    routed_input_tokens INTEGER NOT NULL,
                    routed_output_tokens INTEGER NOT NULL,
                    routed_latency_ms REAL NOT NULL,
                    routed_cost_microusd INTEGER NOT NULL,
                    verification_model_id TEXT,
                    verification_quality_score REAL,
                    verification_threshold REAL,
                    verification_passed INTEGER,
                    verification_cost_microusd INTEGER NOT NULL,
                    escalated INTEGER NOT NULL,
                    rerun_model_id TEXT,
                    rerun_input_tokens INTEGER,
                    rerun_output_tokens INTEGER,
                    rerun_latency_ms REAL,
                    rerun_cost_microusd INTEGER NOT NULL,
                    lifecycle_cost_microusd INTEGER NOT NULL,
                    direct_gpt4o_cost_microusd INTEGER NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events BEGIN
                    SELECT RAISE(ABORT, 'audit_events is append-only: UPDATE rejected');
                END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events BEGIN
                    SELECT RAISE(ABORT, 'audit_events is append-only: DELETE rejected');
                END;
                """
            )

    def append(
        self,
        timestamp: datetime,
        request: Request,
        classifier_tier: str,
        routed_response: Response,
        *,
        verification: VerificationResult | None = None,
        verification_response: Response | None = None,
        rerun_response: Response | None = None,
    ) -> None:
        """Validate lifecycle inputs and persist the resulting audit event.

        The store deliberately does not accept a caller-constructed ``AuditEvent``.
        It establishes provenance by reconstructing the deterministic lifecycle
        immediately before the insert.
        """
        event = AuditEvent.from_lifecycle(
            timestamp,
            request,
            classifier_tier,
            routed_response,
            verification=verification,
            verification_response=verification_response,
            rerun_response=rerun_response,
        )
        values = _event_values(event)
        try:
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO audit_events ({', '.join(values)}) "
                    f"VALUES ({', '.join('?' for _ in values)})",
                    tuple(values.values()),
                )
        except sqlite3.IntegrityError as error:
            if "request_id" in str(error):
                raise ValueError("Audit event for this request ID already exists") from error
            raise

    def read_all(self) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY timestamp, insertion_id"
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def report_aggregates(self) -> dict[str, list[tuple[Any, ...]]]:
        """Return aggregate-only rows for a report without exposing event text."""
        with self._connect() as connection:
            return {
                "summary": connection.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(routed_cost_microusd), 0),
                           COALESCE(SUM(lifecycle_cost_microusd), 0),
                           COALESCE(SUM(direct_gpt4o_cost_microusd), 0),
                           COALESCE(SUM(escalated), 0)
                    FROM audit_events
                    """
                ).fetchall(),
                "daily": connection.execute(
                    """
                    SELECT substr(timestamp, 1, 10), SUM(lifecycle_cost_microusd)
                    FROM audit_events GROUP BY substr(timestamp, 1, 10) ORDER BY 1
                    """
                ).fetchall(),
                "weekly": connection.execute(
                    """
                    SELECT strftime('%Y-W%W', timestamp), SUM(lifecycle_cost_microusd)
                    FROM audit_events GROUP BY strftime('%Y-W%W', timestamp) ORDER BY 1
                    """
                ).fetchall(),
                "routing": connection.execute(
                    """
                    SELECT routed_provider || '/' || routed_model_id, COUNT(*) FROM audit_events
                    GROUP BY routed_provider, routed_model_id ORDER BY routed_provider, routed_model_id
                    """
                ).fetchall(),
                "verification": connection.execute(
                    """
                    SELECT CASE verification_passed WHEN 1 THEN 'passed' ELSE 'failed' END,
                           COUNT(*)
                    FROM audit_events WHERE verification_passed IS NOT NULL
                    GROUP BY verification_passed ORDER BY verification_passed DESC
                    """
                ).fetchall(),
            }


def _event_values(event: AuditEvent) -> dict[str, Any]:
    return {
        "timestamp": event.timestamp.isoformat(),
        "request_id": event.request_id,
        "prompt_hash": event.prompt_hash,
        "classifier_tier": event.classifier_tier,
        "routed_provider": event.routed_provider,
        "routed_model_id": event.routed_model_id,
        "routed_input_tokens": event.routed_input_tokens,
        "routed_output_tokens": event.routed_output_tokens,
        "routed_latency_ms": event.routed_latency_ms,
        "routed_cost_microusd": event.routed_cost_microusd,
        "verification_model_id": event.verification_model_id,
        "verification_quality_score": event.verification_quality_score,
        "verification_threshold": event.verification_threshold,
        "verification_passed": event.verification_passed,
        "verification_cost_microusd": event.verification_cost_microusd,
        "escalated": event.escalated,
        "rerun_model_id": event.rerun_model_id,
        "rerun_input_tokens": event.rerun_input_tokens,
        "rerun_output_tokens": event.rerun_output_tokens,
        "rerun_latency_ms": event.rerun_latency_ms,
        "rerun_cost_microusd": event.rerun_cost_microusd,
        "lifecycle_cost_microusd": event.lifecycle_cost_microusd,
        "direct_gpt4o_cost_microusd": event.direct_gpt4o_cost_microusd,
    }


def _event_from_row(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        timestamp=datetime.fromisoformat(str(row["timestamp"])),
        request_id=str(row["request_id"]),
        prompt_hash=str(row["prompt_hash"]),
        classifier_tier=str(row["classifier_tier"]),
        routed_provider=str(row["routed_provider"]),
        routed_model_id=str(row["routed_model_id"]),
        routed_input_tokens=int(row["routed_input_tokens"]),
        routed_output_tokens=int(row["routed_output_tokens"]),
        routed_latency_ms=float(row["routed_latency_ms"]),
        routed_cost_microusd=int(row["routed_cost_microusd"]),
        verification_model_id=_optional_str(row["verification_model_id"]),
        verification_quality_score=_optional_float(row["verification_quality_score"]),
        verification_threshold=_optional_float(row["verification_threshold"]),
        verification_passed=_optional_bool(row["verification_passed"]),
        verification_cost_microusd=int(row["verification_cost_microusd"]),
        escalated=bool(row["escalated"]),
        rerun_model_id=_optional_str(row["rerun_model_id"]),
        rerun_input_tokens=_optional_int(row["rerun_input_tokens"]),
        rerun_output_tokens=_optional_int(row["rerun_output_tokens"]),
        rerun_latency_ms=_optional_float(row["rerun_latency_ms"]),
        rerun_cost_microusd=int(row["rerun_cost_microusd"]),
        lifecycle_cost_microusd=int(row["lifecycle_cost_microusd"]),
        direct_gpt4o_cost_microusd=int(row["direct_gpt4o_cost_microusd"]),
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
