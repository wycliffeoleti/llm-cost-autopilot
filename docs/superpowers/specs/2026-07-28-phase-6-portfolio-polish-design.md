# Phase 6: Truthful Portfolio Polish — Design

## Objective

Produce a reproducible **offline deterministic seeded-lifecycle demonstration**
for the portfolio. This is not a load test, benchmark, production simulation,
spend study, quality evaluation, or savings claim.

## Scope

Replay the repository's 10 fixed public fixture prompts six times, in documented
order, for 60 deterministic lifecycles. Use IDs `phase6-0001` through
`phase6-0060`, fixed UTC timestamps, existing deterministic classifier/routing,
`FakeProvider`, simulated verification, explicit reruns, local append-only
audit, and the existing static report renderer. The runner makes no network,
live-provider, credential, HTTP-server, concurrent, or deployment call.

Add a small injection point to `OfflineService.complete()` for a supplied UTC
timestamp, defaulting to the existing current-UTC behavior. Create a runner that
writes an ignored temporary SQLite audit DB and outputs a tracked static report.

Commit only public aggregate artifacts:
- `docs/artifacts/phase6/seeded-lifecycle-report.html`
- `docs/artifacts/phase6/seeded-lifecycle-report.png`
- `docs/artifacts/phase6/README.md`

Never commit SQLite DBs, raw event exports, request traces/logs, API payload
captures, raw prompts/outputs, prompt hashes, or output digests.

## Required language

> **Offline deterministic seeded demonstration.** This artifact replays 60
> request lifecycles from the repository’s fixed synthetic prompt fixture using
> `FakeProvider`, fixed constants, and fixed timestamps. It makes no network or
> live-provider calls and incurs no provider spend. Any tokens, latency,
> verification results, USD figures, or deltas shown are deterministic simulated
> values, not measurements of real spend, answer quality, routing efficacy,
> throughput, reliability, or realized savings. The routing dataset remains
> `ai_drafted_pending_human_review`; its labels and derived metrics are not
> real-world ground truth.

Retain the Phase 4 static-report banner. Use only existing labels for simulated
reduction/delta; do not headline savings or quality parity.

## Verification

Test 60 unique records, fixed ordering/timestamps, deterministic byte-identical
HTML regeneration, aggregate reconciliation, FakeProvider-only execution, and
absence of prompt/output/hash leakage from public HTML/artifacts. Capture and
visually inspect a screenshot at fixed viewport with the disclaimer visible.
Update the root README/case-study material to explain reproduction, constraints,
privacy boundary, and what evidence remains needed for real evaluation.
