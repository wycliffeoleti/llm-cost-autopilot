# LLM Cost Autopilot

An offline, deterministic prototype of an LLM cost-routing pipeline (BASWE Project 2). The codebase makes no live provider calls and requires no provider credentials.

## Status

- **Phase 1 — unified model interface:** complete.
- **Phase 2 — complexity classifier and offline routing:** complete.
- **Phase 3 — offline simulated verification and explicit escalation:** complete.
- **Phases 4–6** (audit/dashboard, API, containerization): not started.

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
