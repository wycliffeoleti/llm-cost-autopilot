from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from costpilot.audit import AuditEvent, SQLiteAuditStore
from costpilot.classifier import load_dataset, train_classifier
from costpilot.domain import Request
from costpilot.providers.fake import FAKE_MODELS, FakeProvider
from costpilot.routing import classify_and_route, load_routing_config
from costpilot.verification import VerificationResult, load_verification_config, verify_response

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"
ROUTING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"
VERIFICATION_CONFIG_PATH = Path(__file__).parent.parent / "config" / "verification.yaml"


def _event(
    *,
    timestamp: datetime = datetime(2026, 7, 28, 10, 30, tzinfo=UTC),
    request_id: str = "request-1",
    prompt: str = "Sensitive prompt: do not persist this text.",
    verification: bool = False,
    rerun: bool = False,
) -> AuditEvent:
    request = Request(prompt=prompt, request_id=request_id)
    provider = FakeProvider()
    routed = provider.send(request, FAKE_MODELS["claude-haiku"])
    if not verification:
        return AuditEvent.from_lifecycle(timestamp, request, "tier_1", routed)

    reference = provider.send(request, FAKE_MODELS["gpt-4o"])
    verification_response = (
        replace(reference, output_text="[gpt-4o] simulated response (digest=divergent)")
        if rerun
        else reference
    )
    result = VerificationResult(
        original_model_id=routed.model_id,
        reference_model_id=verification_response.model_id,
        quality_score=0.0 if rerun else 1.0,
        threshold=1.0,
        passed=not rerun,
        simulated=True,
        original_cost_usd=routed.cost_usd,
        reference_cost_usd=verification_response.cost_usd,
        escalation_cost_delta_usd=verification_response.cost_usd - routed.cost_usd,
    )
    rerun_response = reference if rerun else None
    return AuditEvent.from_lifecycle(
        timestamp,
        request,
        "tier_1",
        routed,
        verification=result,
        verification_response=verification_response,
        rerun_response=rerun_response,
    )


def test_event_from_lifecycle_hashes_prompt_and_accounts_for_full_lifecycle():
    event = _event(verification=True, rerun=True)

    assert event.request_id == "request-1"
    assert len(event.prompt_hash) == 64
    assert event.routed_model_id == "claude-haiku"
    assert event.verification_passed is False
    assert event.escalated is True
    assert event.lifecycle_cost_microusd == (
        event.routed_cost_microusd
        + event.verification_cost_microusd
        + event.rerun_cost_microusd
    )
    assert event.direct_gpt4o_cost_microusd > 0


def test_event_accounts_for_an_actual_phase_2_to_phase_3_simulated_flow():
    request = Request(prompt="What is the capital of Germany?", request_id="phase-2-3")
    provider = FakeProvider()
    classifier = train_classifier(load_dataset(DATASET_PATH).examples)
    routed_model = classify_and_route(request.prompt, classifier, load_routing_config(ROUTING_CONFIG_PATH))
    routed_response = provider.send(request, routed_model)
    verification_config = load_verification_config(VERIFICATION_CONFIG_PATH)
    verification = verify_response(
        request,
        routed_response,
        verification_config.reference_model,
        provider,
        verification_config.default_threshold,
    )
    verification_response = provider.send(request, verification_config.reference_model)

    event = AuditEvent.from_lifecycle(
        datetime(2026, 7, 28, tzinfo=UTC),
        request,
        "tier_1",
        routed_response,
        verification=verification,
        verification_response=verification_response,
    )

    assert event.lifecycle_cost_microusd == (
        event.routed_cost_microusd + event.verification_cost_microusd
    )
    assert event.escalated is False


def test_event_rejects_invalid_provenance_and_inconsistent_totals():
    request = Request(prompt="hello", request_id="request-1")
    response = replace(FakeProvider().send(request, FAKE_MODELS["claude-haiku"]), simulated=False)
    with pytest.raises(ValueError, match="simulated"):
        AuditEvent.from_lifecycle(datetime.now(UTC), request, "tier_1", response)

    event = _event()
    with pytest.raises(ValueError, match="lifecycle"):
        replace(event, lifecycle_cost_microusd=event.lifecycle_cost_microusd + 1)


def test_store_reads_events_in_timestamp_then_insertion_order_and_never_leaks_prompt_or_output(tmp_path):
    database = tmp_path / "audit.sqlite3"
    store = SQLiteAuditStore(database)
    later = _event(timestamp=datetime(2026, 7, 29, tzinfo=UTC), request_id="later")
    first = _event(timestamp=datetime(2026, 7, 28, tzinfo=UTC), request_id="first")
    same_time = _event(timestamp=datetime(2026, 7, 28, tzinfo=UTC), request_id="same")

    store.append(later)
    store.append(first)
    store.append(same_time)

    assert [event.request_id for event in store.read_all()] == ["first", "same", "later"]
    raw_database = database.read_bytes()
    assert b"Sensitive prompt" not in raw_database
    assert b"simulated response" not in raw_database


def test_store_schema_rejects_updates_and_deletes(tmp_path):
    database = tmp_path / "audit.sqlite3"
    store = SQLiteAuditStore(database)
    store.append(_event())

    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("UPDATE audit_events SET request_id = 'changed'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("DELETE FROM audit_events")


def test_store_rejects_duplicate_request_lifecycle_ids(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.sqlite3")
    store.append(_event())
    with pytest.raises(ValueError, match="already exists"):
        store.append(_event())
