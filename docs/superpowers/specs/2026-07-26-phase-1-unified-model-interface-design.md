# LLM Cost Autopilot — Phase 1: Unified Model Interface — Design

**Repository:** `/home/wolnxpc/projects/Personal/portfolio/llm-cost-autopilot`
**Source guide:** BASWE Project 2 ("LLM Cost Autopilot"), PDF pages 6–7 of
`/home/wolnxpc/projects/Personal/references/BASWE_15_AI_Engineering_Projects_Guide.pdf`
— read from the Project 2 heading through the end of Phase 1, step 3 only.
**Status:** Design-only. No Phase 1 code, dependency, credential, provider, CI
configuration, or external integration has been created. Phases 2–6 are named
below as deferred scope only; they were not read, discussed, or designed in
this session.

## 1. Scope of this document

This spec covers **Phase 1 ("Build the Unified Model Interface") only**, per
explicit project-owner scope correction during design. It intentionally
excludes any design detail for the complexity classifier, routing policy,
verification loop, audit/logging store, dashboard, or API surface — those are
BASWE Project 2 Phases 2–6 and are out of scope for this document and this
repository state.

## 2. Origin and placement decision

- BASWE Project 2 is an external architectural reference guide, not evidence
  that any of it has been built. This spec records original, proposed work
  only.
- The portfolio backlog (`/home/wolnxpc/projects/Personal/NICKY_AI_ENGINEERING_BACKLOG.md:520`,
  `:147-176`) maps BASWE Project 2 to **merge into `P1.3 LLM gateway, routing,
  caching, and reliability`** inside the existing `agent-control-plane`
  (TrustMesh) repository, together with BASWE Projects 7 (semantic caching)
  and 11 (LLM gateway).
- **Owner decision (this session):** build Cost Autopilot as a new, standalone
  repository with narrow scope — this Phase 1 unified model interface only,
  none of Projects 7/11's caching or full gateway machinery — explicitly
  positioned as a future merge candidate into P1.3 rather than built inside
  TrustMesh now. This defers the backlog's build-order plan; it does not mark
  the backlog line superseded.
- **Standing constraint for this project:** offline-first, no live provider or
  API calls. All evaluation is reproducible and deterministic using fake
  providers.

## 3. Source alignment (BASWE Project 2, PDF pp. 6–7)

**What You're Building** (guide, verbatim): "An intelligent routing layer
that sits in front of multiple LLM providers, analyzes each incoming
request's complexity, routes it to the cheapest model capable of handling it
at acceptable quality, and continuously validates that routing decisions are
correct." This describes the full system. Phase 1 builds none of the
routing/classification/validation behavior — only the foundation those
depend on. This is consistent with the guide's own phase structure, which
defers routing to Phase 2 and validation to Phase 3.

**Why This Project Lands Interviews** (guide, verbatim): "Every company
running LLMs at scale is bleeding money on over-provisioned model calls.
Building a cost optimizer signals that you understand AI engineering as a
business problem, not just a technical one." This narrative depends on an
actual cost-savings number, which only exists once routing (Phase 2) exists.
**Phase 1 alone produces no portfolio headline metric** — it is foundational
evidence (a working provider abstraction), not yet interview-story evidence.

**Tech stack reconciliation** (guide row → Phase 1 disposition):

| Guide row | Phase 1 disposition |
|---|---|
| Language: Python 3.11+ | Adopted as-is |
| LLM Providers: OpenAI, Anthropic, Ollama (real) | **Deliberate deviation** — fake, offline, deterministic equivalents, per the standing offline-first / no-live-call constraint |
| Router: FastAPI | Not present — FastAPI does not appear in any Phase 1 step; it is a Phase 5 concern |
| Classifier: scikit-learn / fine-tuned model | Not present — Phase 2 |
| Eval: custom scoring + LLM-as-judge | Not present — Phase 3 |
| Logging: SQLite + structured JSON | Not present as a persistent store in Phase 1 — Phase 1 step 3's "log outputs/costs/latencies" is satisfied by test assertions on returned `Response` objects; the SQLite audit trail is Phase 4 |
| Dashboard: Streamlit/Grafana | Not present — Phase 4/6 |
| Containerization: Docker | Not present — Phase 5/6 |

**Phase 1 requirements, reconciled step by step:**

1. **Model registry** (guide): `ModelConfig` dataclass with provider name,
   model ID, cost/input-token, cost/output-token, average latency, quality
   tier, populated with real pricing for GPT-4o, GPT-4o-mini, Claude Sonnet,
   Claude Haiku, and a local Llama model via Ollama. **Phase 1 design:**
   identical 6-field `ModelConfig`, same 5 models. Pricing is a **dated
   snapshot of publicly published list prices** (stamped with the snapshot
   date, not live-queried) — the offline-first adaptation of "real pricing,"
   already approved this session.
2. **Abstraction layer** (guide): a single `send_request(prompt,
   model_config)` function returning a standardized `Response` object with
   output text, tokens used (input + output), latency, cost, and model ID.
   **Phase 1 design:** a `Provider` protocol with `send(request, model) ->
   Response`, functionally identical fields, expressed as a protocol method
   (hexagonal ports-and-adapters, see §4) rather than a bare function.
3. **Test every provider** (guide): send the same 10 prompts to every model
   in the registry; log outputs, costs, latencies; produces baseline data and
   validates the abstraction layer. **Phase 1 design:** `tests/test_providers.py`
   exercises a fixed 10-prompt set (`evals/prompts/`) against every
   registered fake provider, asserting each returns a well-formed `Response`
   with populated output/tokens/cost/latency.

No unreconciled gaps against Phase 1 specifically.

## 4. Approved architecture

Hexagonal ports-and-adapters, consistent with the `agent-control-plane`
(TrustMesh) precedent: a framework-independent domain layer depends only on a
`Provider` protocol. Fake providers are one adapter implementing that
protocol now; a real provider adapter can implement it later without
changing domain code. This is what makes "provider-neutral request/response
contracts" a true statement rather than an aspiration.

Package name: `costpilot`. Personal GitHub identity (`@wycliffeoleti`)
applies, per this being a repo under `~/projects/`.

## 5. Phase 1 design (approved)

**Repository layout** (Phase 1 only — no routing/audit/API directories yet):

```
llm-cost-autopilot/
  costpilot/
    domain.py         # ModelConfig, Request, Response — pure dataclasses
    ports.py           # Provider Protocol (unified interface)
    providers/
      fake.py            # FakeProvider registry: 5 offline, deterministic providers
  evals/
    prompts/              # fixed 10-prompt set used for provider comparison
  tests/
    test_providers.py       # same 10 prompts x every fake provider; asserts output/tokens/cost/latency
```

**Domain contracts (`domain.py`)** — provider-neutral; no provider SDK types
leak through:

- `ModelConfig`: provider name, model id, cost/input-token, cost/output-token,
  average latency, quality tier (high/medium/low).
- `Request`: prompt text, request id.
- `Response`: output text, tokens in/out, latency, cost, model id used — the
  single shape every provider must return.

`RoutingDecision` and any classifier/routing/verification fields are
explicitly excluded from this phase's contracts; they are Phase 2/3 concepts.

**Unified provider interface (`ports.py`)**: `Provider(Protocol)` with
`send(request: Request, model: ModelConfig) -> Response`. This is the single
seam every fake (and, later, real) provider implements.

**Fake provider registry (`providers/fake.py`)** — 5 models matching the
guide's roster: `gpt-4o`, `gpt-4o-mini`, `claude-sonnet`, `claude-haiku`,
`llama-local`. Each carries a `ModelConfig` with a dated snapshot of
publicly published list prices (snapshot date stamped in the config, not
live-queried). `FakeProvider.send()` is a pure deterministic function of
`(prompt, model_config)`: token counts derive from prompt length via a fixed
per-model multiplier, cost = tokens × the model's rates, latency = the
model's fixed average, and output text is a deterministic templated string
(model id plus a short hash-derived transform of the prompt). The same
prompt against the same model always reproduces identically; different
models produce distinct but still deterministic outputs from the same
prompt.

**Provider-comparison tests (`tests/test_providers.py`)** — the guide's
Phase 1 step 3: send the same fixed 10-prompt set through every registered
fake provider and assert each returns a well-formed `Response` with
populated output/tokens/cost/latency. This is baseline evidence that the
abstraction layer works across all five providers — no routing, no
classification, no persistent logging store.

## 6. Explicitly deferred (named only — not designed in this document)

- **Phase 2:** complexity classifier, routing policy/YAML.
- **Phase 3:** verification loop.
- **Phase 4:** SQLite/JSON audit store, cost dashboard.
- **Phase 5:** FastAPI endpoints.
- **Phase 6:** portfolio load test and report.

## 7. Out of scope / non-claims for this document

- No live provider calls, API keys, or credentials of any kind.
- No routing, classification, audit persistence, dashboard, or API surface —
  these are unimplemented and undesigned pending separate approval.
- No GitHub repository, remote, or push — this repository is local-only at
  this stage.
- This document does not claim Phase 1 code exists. It is a design record
  only.
