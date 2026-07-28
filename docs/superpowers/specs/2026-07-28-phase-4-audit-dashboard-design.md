# Phase 4: Offline Audit Trail and Cost Dashboard — Design

## Objective

Add a local append-only audit store and generated static report for the complete
simulated lifecycle: route, fake response, simulated verification, and optional
explicit fake rerun. This is a portfolio-prototype observability feature; it
must never report real spend, real answer quality, realized savings, or live
provider activity.

## Architecture

Use stdlib `sqlite3` for local storage and a self-contained static HTML report.
SQLite provides atomic inserts and UTC day/week aggregation without adding a
server, database dependency, dashboard framework, or external asset. The
SQLite file and ordinary generated reports are gitignored.

Create:

- `costpilot/audit.py`: immutable `AuditEvent`, event construction/validation,
  `SQLiteAuditStore`, append-only schema/triggers.
- `costpilot/reporting.py`: aggregate queries and deterministic escaped HTML/text
  report generation; optional local CLI only.
- `config/audit.yaml` is intentionally excluded: Phase 4 needs no runtime
  provider, credential, or threshold configuration.

One audit event represents one request lifecycle, not each provider invocation.
Persistence is explicit (`store.append(...)` with lifecycle inputs); neither
`FakeProvider.send()` nor verification silently writes records. The store
constructs and validates the event immediately before insertion.

## Event requirements

Store: UTC ISO-8601 timestamp, request ID, SHA-256 prompt hash (never raw
prompt/output), classifier tier, routed canonical fake model/provider,
simulated response token/cost/latency fields, simulated verification outcome,
explicit escalation fields, simulated lifecycle total, and a counterfactual
single-direct-`gpt-4o` fake-response cost.

All provenance must be validated:
- routed response `simulated=True`;
- verification result `simulated=True` if present;
- canonical fake model identities only;
- values finite/non-negative; tiers valid; hash is 64 lowercase hex characters;
- totals reconcile exactly under a documented decimal/microdollar policy.

SQLite triggers reject UPDATE and DELETE. This prevents accidental application
mutation; it is not tamper-proof storage against someone who can modify the
local database file.

## Reports and claims

Reports contain a repeated visible banner:

> Offline deterministic prototype data. All responses, tokens, latency,
> verification values, and USD figures are simulated from `FakeProvider` and
> fixed constants. No live provider was called and the figures are not actual
> spend, answer quality, routing efficacy, or realized savings.

Include coverage, UTC daily/weekly simulated cost, routing distribution, fake
verification distribution/pass rate, simulated escalation rate, routing-only
simulated reduction versus one direct GPT-4o fake response, and end-to-end
simulated delta versus that baseline. The latter includes verifier/rerun cost
and may be positive; it must not be called savings.

HTML uses only local embedded CSS and escaped text. No server, CDN, JavaScript
framework, remote fonts, analytics, API, queue, live provider, credentials,
retraining, or Phase 5+ functionality is allowed.

## Acceptance criteria

Tests cover append/read/order, validation/provenance, append-only triggers,
prompt/output non-leakage, full Phase 2–3 lifecycle accounting, report totals
and UTC grouping, deterministic escaped dependency-free HTML, and full regression
checks. README documents limitations and local use. Independent review is
required before merge.
