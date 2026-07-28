# LLM Cost Autopilot

An offline, deterministic prototype of an LLM cost-routing pipeline (BASWE Project 2). The codebase makes no live provider calls and requires no provider credentials.

## Status

- **Phase 1 — unified model interface:** complete.
- **Phase 2 — complexity classifier and offline routing:** complete.
- **Phase 3 — offline simulated verification and explicit escalation:** complete.
- **Phase 4 — offline audit trail and static dashboard:** complete.
- **Phase 5 — offline deterministic API:** complete.
- **Phase 6 — truthful portfolio polish:** complete.

## Phase 6 — offline deterministic seeded demonstration

The tracked [Phase 6 artifact](docs/artifacts/phase6/README.md) is a
self-contained aggregate report produced by replaying the repository's ten
fixed synthetic fixture prompts six times in file order. It uses IDs
`phase6-0001` through `phase6-0060`, fixed UTC timestamps, the existing local
classifier and routing rules, simulated verification and explicit reruns, and
`FakeProvider` only. It starts no server and makes no credential, network, or
live-provider call.

Regenerate its HTML from the repository root:

```bash
uv run python -m costpilot.phase6 docs/artifacts/phase6/seeded-lifecycle-report.html
```

The public report is aggregate-only. Request/response text, request-level
records, temporary audit database, event exports, and provider/API payload
captures remain out of the repository. The full-page PNG is a local capture of
the self-contained HTML report, including every report section and its closing
disclaimer; it is not a product UI or deployment.

**Offline deterministic seeded demonstration.** The values shown are fixed
simulated values from `FakeProvider`, not measurements of real spend, answer
quality, routing efficacy, throughput, reliability, or realized savings. The
routing dataset remains `ai_drafted_pending_human_review`; its labels and
derived metrics are not real-world ground truth. A real evaluation would still
need separately authorized privacy-reviewed data, human-reviewed labels,
defined evaluation criteria, and an approved measurement protocol.

## Phase 5 — offline deterministic API

The FastAPI interface is a loopback-only interface to the existing offline
`FakeProvider` pipeline. It never uses credentials, live provider calls, or
runtime model lookup. Run it only on the local loopback interface:

```bash
uv run uvicorn costpilot.api:app --host 127.0.0.1 --port 8000
```

`POST /v1/completions` accepts a bounded nonblank `prompt`, optional bounded
`request_id`, and optional numeric `verification_threshold`. Every completion
runs routing, fake verification, and an explicit fake-reference rerun on a
failed verification before one audit lifecycle is inserted. `GET /healthz`,
`/v1/models`, `/v1/stats`, and `/v1/config` expose only sanitized metadata and
repeat the offline-deterministic provenance disclaimer. Prompt and output text
remain out of audit storage, aggregate responses, errors, and logs.

## Phase 3 — simulated verification

Phase 3 models the control flow of a quality-verification loop entirely
offline. It runs the same request through the configured high-tier
`FakeProvider` reference model, removes the fake model prefix from each output,
and records a deterministic binary agreement score. A failed agreement result
can be passed to `should_escalate`, after which the caller may explicitly call
`rerun_with_reference`.

Every `VerificationResult` has `simulated=True`. This is not a measurement of
answer quality: fake response text has no semantic content, so agreement is
only a deterministic payload comparison. Verification neither trains nor
updates the Phase 2 classifier, and Phase 3 adds no live providers,
credentials, network calls, background queues, persistence, APIs, or retries.

The reference model and threshold are validated from
`config/verification.yaml`; the reference must be one of the repository's
fixed `FAKE_MODELS`, and the threshold must be finite and between `0.0` and
`1.0`.

## Phase 4 — offline audit trail and static dashboard

Phase 4 records one explicit, append-only audit event per simulated request
lifecycle. Events include only a SHA-256 prompt hash: raw prompts and model
outputs are never persisted. Calling `FakeProvider.send()`, verification, or a
rerun never writes an event; the caller explicitly calls
`SQLiteAuditStore.append(...)` with the source lifecycle inputs; the store
constructs and validates the audit event immediately before persistence.

All simulated USD values are rounded to integer microdollars for each fake
provider invocation before lifecycle totals are calculated. This stored-unit
policy makes every total exact: `lifecycle = routed + verification + rerun`.
The SQLite schema rejects UPDATE and DELETE operations. That prevents
accidental application mutation, but it is not tamper-proof for someone who
can modify the local database file.

Create a local report from a database you populated explicitly:

```python
from pathlib import Path

from costpilot.audit import SQLiteAuditStore
from costpilot.reporting import build_report, render_html_report

store = SQLiteAuditStore(Path("audit.sqlite3"))
report = build_report(store)
Path("reports/audit.html").write_text(render_html_report(report), encoding="utf-8")
```

The database and `reports/` directory are gitignored. Reports show UTC daily
and weekly simulated costs, routing and fake-verification distributions,
simulated escalation rate, a routing-only simulated reduction against a direct
GPT-4o fake response, and an end-to-end simulated delta against that baseline.
The end-to-end delta includes simulated verification/rerun cost and can be
positive; it is not savings. Every report repeats the offline-prototype banner:
no live provider was called, and no displayed number is real spend, answer
quality, routing efficacy, or realized savings.

## Data provenance — outstanding review gate

`data/complexity_dataset.draft.json` contains 210 examples used to train and evaluate the Phase 2 classifier. It is **AI-drafted, not human-labelled**. Its status is:

```text
ai_drafted_pending_human_review
```

The held-out accuracy and confusion matrix validate the feature-extraction → classifier → evaluation pipeline only. They are **not** evidence of real-world routing quality because the labels have not received human review.

**Outstanding action:** a human reviewer must assess the 210 examples, correct low-quality or mislabeled entries, and only then treat the dataset or metrics derived from it as trustworthy ground truth.

## Verification

```bash
uv sync
uv run pytest -v
uv run ruff check .
uv run mypy costpilot
```
