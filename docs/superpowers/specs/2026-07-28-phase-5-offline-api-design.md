# Phase 5: Offline Deterministic API — Design

## Objective

Expose the Phase 2–4 pipeline as a **loopback-only, deterministic FastAPI
application**. It is an interface to the existing `FakeProvider` prototype,
not a production OpenAI-compatible service, live-provider proxy, or quality
system.

## Scope

Implement `costpilot/api.py` and a synchronous service boundary that loads the
fixture, routing and verification configuration once at startup; trains the
deterministic classifier once; performs route → fake response → simulated
verification → explicit rerun if failed; then persists exactly one audited
lifecycle before returning a success response.

Use FastAPI/uvicorn only. Add TestClient contract coverage and one bounded real
`uvicorn` loopback smoke test. The smoke server must bind `127.0.0.1` only,
select an ephemeral port, use no reload/workers, and terminate reliably.

Endpoints:
- `GET /healthz`
- `POST /v1/completions` (`201` only after successful audit insertion)
- `GET /v1/models`
- `GET /v1/stats`
- `GET /v1/config`

No `/audit-events`, DB/report download, streaming, batch execution, CORS,
authentication, configuration writes, provider/model selection, raw config
paths, Docker, worker, queue, retraining, or public deployment.

## Safety and provenance

- The service uses only `FakeProvider`; no provider credentials, external
  network calls, live prices, or runtime model lookup.
- Completion requests have a nonblank bounded prompt; optionally supplied
  request IDs are bounded and duplicate IDs return `409` rather than `500`.
- Only a numeric verification threshold may be overridden; clients cannot alter
  models, routes, config files, database paths, or providers.
- Keep output text only in HTTP completion response. Audit storage, stats,
  errors, and logs must not expose prompt or output text.
- Every endpoint exposing simulation-related data includes an explicit
  offline-deterministic provenance/disclaimer object. Values are simulated
  metrics/deltas, never actual spend, quality, efficacy, or savings.
- Config is read-only; startup validates effective file data and fails closed.
- Unexpected errors return generic correlation IDs rather than exception text.

## API semantics

`POST /v1/completions` returns request ID, final simulated output, routed tier
and canonical fake model, simulated token/cost values in integer microdollars,
verification/escalation metadata, and a provenance object. If verification
fails, final output is the explicit fake-reference rerun. A response must state
that classifier fixture data remains `ai_drafted_pending_human_review`.

Models/config/stats return sanitized aggregate/canonical metadata only. Stats
reuse Phase 4 reporting names such as `routing_only_simulated_delta_microusd`
and `end_to_end_simulated_delta_microusd`.

## Tests

Test startup configuration, request validation, lifecycle/audit insertion,
duplicate conflict, models/stats/config sanitization, explicit provenance,
no prompt/output leakage, generic internal errors, and a real loopback uvicorn
health + completion smoke test. Run full pytest, Ruff, MyPy, and diff check.
