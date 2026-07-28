# Phase 6 Portfolio Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, aggregate-only offline seeded-lifecycle portfolio artifact.

**Architecture:** Add an optional UTC timestamp seam to the offline service. A Phase 6 runner loads the fixed fixture in order, replays it six times into a temporary ignored SQLite database, then renders only aggregate data to a caller-selected HTML file.

**Tech Stack:** Python 3.11, pytest, existing FakeProvider/audit/reporting modules, Chromium for final capture.

## Global Constraints

- Reuse only `FakeProvider`; make no network, credential, HTTP-server, provider, concurrency, queue, deployment, Docker, or Phase 7 addition.
- Replay ten fixed fixture prompts six times as `phase6-0001` through `phase6-0060` with fixed UTC timestamps.
- Keep the public report aggregate-only: no prompt/output text, prompt hash, output digest, event export, or SQLite database.
- Retain the Phase 4 banner and required Phase 6 disclaimer; make no live price, quality, savings, effectiveness, reliability, or throughput claims.

---

### Task 1: Timestamp injection

**Files:**
- Modify: `tests/test_phase6.py`
- Modify: `costpilot/service.py`

**Interfaces:**
- Produces: `OfflineService.complete(prompt, request_id, verification_threshold=None, timestamp=None) -> Completion`

- [ ] Write a failing test that supplies a fixed UTC timestamp and observes it through `SQLiteAuditStore.read_all()`.
- [ ] Run `uv run pytest tests/test_phase6.py::test_complete_uses_supplied_utc_timestamp -q` and verify the missing-parameter failure.
- [ ] Add the smallest optional timestamp seam while preserving `datetime.now(UTC)` as the default.
- [ ] Re-run targeted and existing tests, then commit `feat: inject offline lifecycle timestamps`.

### Task 2: Deterministic seeded runner

**Files:**
- Modify: `tests/test_phase6.py`
- Create: `costpilot/phase6.py`

**Interfaces:**
- Produces: `run_seeded_demo(output_path: Path) -> AuditReport`

- [ ] Write failing tests for 60 ordered IDs, fixed timestamps, aggregate reconciliation, FakeProvider-only execution, and byte-identical rendering.
- [ ] Run targeted tests and verify import failure.
- [ ] Implement fixture loading, six ordered fixed-time passes, temporary ignored database, and aggregate HTML rendering.
- [ ] Re-run targeted tests, then commit `feat: add deterministic Phase 6 seeded demo`.

### Task 3: Public artifacts and case study

**Files:**
- Modify: `tests/test_phase6.py`
- Create: `docs/artifacts/phase6/README.md`
- Create: `docs/artifacts/phase6/seeded-lifecycle-report.html`
- Create: `docs/artifacts/phase6/seeded-lifecycle-report.png`
- Modify: `README.md`

- [ ] Write failing public-artifact tests for disclaimer language and prompt/output/hash exclusion.
- [ ] Regenerate HTML twice, capture one fixed-viewport local-file screenshot, and inspect it.
- [ ] Document reproduction and the evidence/privacy boundaries without claims outside the specification.
- [ ] Re-run artifact tests, then commit `docs: publish Phase 6 deterministic demo artifact`.

### Task 4: Final verification

- [ ] Regenerate HTML twice and `cmp` the outputs.
- [ ] Assert public artifacts contain no fixture prompt, fake output, 64-character SHA value, SQLite database, or export.
- [ ] Run `uv run pytest -q`, `uv run ruff check .`, `uv run mypy costpilot`, and `git diff --check`.
