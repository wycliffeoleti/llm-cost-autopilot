# Phase 3: Offline Quality-Verification Loop — Design

**Status:** Approved implementation boundary for the portfolio prototype

## 1. Objective

Add a deterministic, offline representation of the guide’s quality-verification
loop: verify a routed fake response against a higher-tier fake reference,
record a comparison outcome, and identify when the original routing should be
escalated. The feature must not call a live provider, use credentials, persist
an audit log, retrain the classifier, or claim that a simulated verdict is a
real quality measurement.

## 2. Constraint-driven interpretation

The project guide proposes asynchronously sending the same request to a
highest-tier provider and comparing outputs. This repository intentionally has
only `FakeProvider`, whose responses are deterministic simulations rather than
LLM answers. Therefore Phase 3 implements the control-flow and accounting
contract, not a semantic-quality claim:

- reference execution uses `FakeProvider` plus a configured high-tier model;
- comparison normalizes the fake response payload and evaluates deterministic
  agreement rules;
- every result is labelled `simulated`; it is unsuitable for production quality
  decisions or classifier training;
- no background queue service, database, dashboard, live provider, API, or
  scheduled retraining is introduced.

## 3. Public API

Create `costpilot/verification.py`.

```python
@dataclass(frozen=True)
class VerificationResult:
    original_model_id: str
    reference_model_id: str
    quality_score: float
    threshold: float
    passed: bool
    simulated: Literal[True]
    original_cost_usd: float
    reference_cost_usd: float
    escalation_cost_delta_usd: float


def verify_response(
    request: Request,
    original_response: Response,
    reference_model: ModelConfig,
    provider: Provider,
    threshold: float,
) -> VerificationResult: ...


def should_escalate(result: VerificationResult) -> bool: ...

def rerun_with_reference(request: Request, reference_model: ModelConfig, provider: Provider) -> Response: ...
```

`verify_response` invokes the supplied provider exactly once with the reference
model. It checks response/model identity and compares normalized fake output
content. Exact normalized agreement scores `1.0`; disagreement scores `0.0`.
The score is deliberately binary because fake response text has no semantics.
Thresholds must be finite and in `[0.0, 1.0]`.

`should_escalate` is pure: it returns `not result.passed`. The caller owns
latency, retry, delivery, and persistence decisions. `rerun_with_reference`
performs exactly one deterministic reference-model execution. It does not
silently alter the original response.

## 4. “Async” boundary

Use no queue, executor, daemon, or network transport in Phase 3. The verifier
functions are side-effect-limited and can be submitted to an application-owned
background worker in a later API/runtime phase. A small `verify_after_response`
helper is intentionally deferred because a synchronous return type would make
an async claim misleading in this offline library.

## 5. Model policy

Create `config/verification.yaml`:

```yaml
reference_model_id: gpt-4o
thresholds:
  default: 1.0
```

A loader validates that the reference model exists in `FAKE_MODELS` and that
all thresholds meet the documented numeric range. Phase 3 uses only `default`;
per-use-case thresholds and real judges are deferred until a defined task
schema and authorized evaluation source exist.

## 6. Test strategy

1. Configuration tests validate model ID, default threshold, and invalid values.
2. Verifier tests assert one reference call, deterministic pass/fail outcomes,
   correct cost accounting, and no mutation of the original `Response`.
3. Escalation tests assert the decision is pure and failures rerun only on
   explicit request.
4. Integration test runs Phase 2 routing, fake original execution, verification,
   and explicit reference rerun without external I/O.
5. Full pytest, Ruff, mypy, diff check, scope/security review before publish.

## 7. Explicitly deferred

- Live model calls, LLM-as-judge scoring, credentials, provider SDKs.
- Human evaluation or claims of real routing/quality efficacy.
- Background worker infrastructure, audit storage, dashboard, API, Docker.
- Adding simulated verification outcomes to the AI-drafted classifier fixture
  or automatically retraining it.

## 8. Acceptance criteria

Phase 3 is complete only when the offline verifier and explicit escalation
path are fully tested, configuration is validated against `FAKE_MODELS`, all
results state `simulated=True`, documentation states the limitation, and the
existing Phase 1–2 checks remain green.
