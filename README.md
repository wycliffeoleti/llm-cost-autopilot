# LLM Cost Autopilot

An offline, deterministic prototype of an LLM cost-routing pipeline (BASWE Project 2). The codebase makes no live provider calls and requires no provider credentials.

## Status

- **Phase 1 — unified model interface:** complete.
- **Phase 2 — complexity classifier and offline routing:** complete.
- **Phases 3–6** (quality-verification loop, audit/dashboard, API, containerization): not started.

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
