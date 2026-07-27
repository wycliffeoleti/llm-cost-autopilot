# LLM Cost Autopilot — Phase 2: Complexity Classifier — Design

**Repository:** `/home/wolnxpc/projects/Personal/portfolio/llm-cost-autopilot`
**Source guide:** BASWE Project 2 ("LLM Cost Autopilot"), PDF pages 6–7 of
`/home/wolnxpc/projects/Personal/BASWE_15_AI_Engineering_Projects_Guide.pdf`
— read from the Project 2 heading through the end of Phase 2, step 4 only.
**Status:** Design, approved section-by-section by the project owner during
brainstorming. Phases 3–6 are named below as deferred scope only; they were
not read, discussed, or designed in this session.

## 1. Scope of this document

This spec covers **Phase 2 ("Build the Complexity Classifier") only**. It
builds directly on the approved Phase 1 spec
(`docs/superpowers/specs/2026-07-26-phase-1-unified-model-interface-design.md`)
and does not modify any Phase 1 contract (`costpilot/domain.py`, `ports.py`,
`providers/fake.py`, or their tests). It explicitly excludes the async
quality-verification loop (Phase 3), the audit/logging store and cost
dashboard (Phase 4), the FastAPI service (Phase 5), and containerization/load
testing (Phase 6).

## 2. Standing constraints (carried forward from Phase 1)

- **Offline-first, no live provider or API calls.** All classifier training,
  evaluation, and routing decisions run entirely locally against the
  hand-authored dataset and the existing fake-provider registry. No network
  access, no credentials.
- **Fully deterministic.** Every random operation (train/test split,
  classifier fitting) uses a fixed, explicit seed constant, so results
  reproduce identically across runs.

## 3. Source alignment (BASWE Project 2, Phase 2, PDF p. 7)

**Guide text (verbatim), Phase 2 steps:**

1. *Define complexity tiers.* Tier 1 (simple): reformatting, extraction,
   basic Q&A from provided context. Tier 2 (moderate): summarization,
   classification, structured analysis. Tier 3 (complex): multi-step
   reasoning, creative generation, nuanced judgment calls.
2. *Build a labeled dataset.* 200+ example prompts across all three tiers,
   labeled by hand. Extracted features: token count, presence of
   instructions like "analyze" or "compare," number of constraints, whether
   context is provided, and output format complexity.
3. *Train the classifier.* A simple scikit-learn model (logistic regression
   or random forest) on the extracted features. Not optimizing for
   classifier perfection — building the routing skeleton. Track accuracy and
   confusion matrix; >80% held-out accuracy is fine for V1.
4. *Create the routing map.* Map each complexity tier to a model (Tier 1 →
   cheapest, Tier 2 → mid-tier, Tier 3 → highest quality). Store as a
   configurable YAML so models can be swapped without code changes.

**Tech stack reconciliation** (guide row → Phase 2 disposition):

| Guide row | Phase 2 disposition |
|---|---|
| Classifier: scikit-learn / fine-tuned model | Adopted — scikit-learn `LogisticRegression`, per owner decision |
| Logging: SQLite + structured JSON | Not present — Phase 2 verification is pytest assertions only (held-out accuracy, confusion matrix shape); the persistent audit trail is Phase 4 |
| Router: FastAPI | Not present — Phase 2 produces an offline, synchronous `classify_and_route` function; the live HTTP router is Phase 5 |

Every one of the guide's four Phase 2 steps is reconciled below with no
unaddressed gap.

## 4. Dataset design

**Tiers** (fixed by the guide, not a design choice):

- **Tier 1 (simple):** reformatting, extraction, basic Q&A from provided
  context.
- **Tier 2 (moderate):** summarization, classification, structured analysis.
- **Tier 3 (complex):** multi-step reasoning, creative generation, nuanced
  judgment calls.

**Naming note:** these complexity tiers (`tier_1`/`tier_2`/`tier_3`, this
phase's classifier output) are a distinct concept from Phase 1's
`ModelConfig.quality_tier` field (`"high"|"medium"|"low"`, a static per-model
attribute). No field or value is shared between them; where both appear in
the same discussion, "complexity tier" and "quality tier" are used
explicitly to disambiguate.

**Authoring method:** 210+ prompts, individually hand-authored and
hand-labeled — each prompt is a distinct, manually composed example and each
label is a manual judgment call, not generated from a template or assigned
by construction. Target balance: ~70 examples per tier, so the held-out
confusion matrix is interpretable without majority-class artifacts.

**Provenance constraint (blocking, unresolved as of this revision):**
"hand-authored"/"hand-labeled" is a claim about who produced the data — a
human — not merely about writing style. The assistant must not generate
dataset content, individually composed or otherwise, and commit or describe
it as hand-authored human ground truth; that would misrepresent the
dataset's provenance, which matters both for the classifier's accuracy
claims and for the project's portfolio narrative. This phase's
implementation plan must resolve authorship through one of:

- **(a) AI-assisted draft, explicitly gated on human review** — the
  assistant may draft candidate examples, but they are stored/labeled as an
  unreviewed AI-assisted draft and must not be treated as golden/labeled
  ground truth (e.g. used in accuracy claims or committed as the production
  `data/complexity_dataset.json`) until the project owner has reviewed and
  explicitly approved them.
- **(b) Owner-input gate** — the assistant builds only the dataset schema,
  loader, and validation/splitting infrastructure; the actual 210+ examples
  are supplied by the project owner directly, and no classifier training
  proceeds until that data exists.

No dataset content has been committed under either path yet. This choice is
pending project-owner decision and gates the classifier/routing tasks in the
implementation plan.

**Storage** (`data/complexity_dataset.json`), matching the Phase 1
`evals/prompts/phase1_baseline.json` convention:

```json
{
  "version": "1.0",
  "examples": [
    {"id": "c001", "text": "Extract the invoice number from this text: ...", "tier": "tier_1"},
    {"id": "c002", "text": "Summarize this article in three bullet points: ...", "tier": "tier_2"},
    {"id": "c003", "text": "Design a caching strategy for a multi-region API and justify the trade-offs.", "tier": "tier_3"}
  ]
}
```

`tier` is a closed enum (`tier_1 | tier_2 | tier_3`), validated at load time;
an unrecognized value fails fast rather than silently entering training
data.

## 5. Feature extraction (`costpilot/features.py`)

Five features, matching the guide's list exactly, each a pure regex/keyword
function over prompt text (Python stdlib `re` only — no ML, no new
dependency for this module):

| Feature | Type | Heuristic |
|---|---|---|
| `token_count` | int | `len(prompt.split())` — word-count, consistent with the Phase 1 `providers/fake.py` convention |
| `instruction_verb_count` | int | count of matches against a fixed complex-instruction-verb list: `analyze, compare, evaluate, synthesize, design, critique, contrast, recommend, justify, assess` |
| `constraint_count` | int | count of matches against constraint-signaling patterns: `must`, `should`, `do not`/`don't`, `at least`, `exactly`, `no more than`, numeric ranges, numbered/bulleted requirement lines |
| `has_context` | bool | whether the prompt supplies data to work over — pattern match on cues like `"this text:"`, `"the following"`, `"given:"`, or a quoted/colon-delimited trailing block |
| `output_format_complexity` | ordinal `{0,1,2}` | 0 = free text; 1 = simple structure requested (`list`, `table`, `bullet`); 2 = strict structure requested (`json`, `schema`, explicit field names) |

`extract_features(prompt: str) -> dict[str, float]` returns all five as a
flat dict. `classifier.py` converts a list of these dicts into the numeric
matrix scikit-learn needs by projecting onto a fixed, ordered key list (no
`DictVectorizer` dependency needed — the keys are static and known).

## 6. Classifier and evaluation (`costpilot/classifier.py`)

- `load_dataset(path: Path) -> list[LabeledExample]` — parses
  `data/complexity_dataset.json`, validates every `tier` against the closed
  enum, raises on anything else.
- `train_test_split_dataset(examples, seed: int, test_size: float = 0.2)` —
  stratified split preserving each tier's proportion in both halves, using a
  fixed module-level seed constant so every run reproduces an identical
  split.
- `train_classifier(train_examples, seed: int) -> LogisticRegression` —
  builds the feature matrix via `features.extract_features`, fits
  `LogisticRegression(max_iter=1000, random_state=seed)`.
- `evaluate(model, test_examples) -> EvaluationResult` — held-out accuracy
  (`model.score`) and the 3×3 confusion matrix
  (`sklearn.metrics.confusion_matrix`, fixed label order `tier_1, tier_2,
  tier_3`).
- `predict_tier(prompt: str, model) -> str` — extracts features for one
  prompt, returns the predicted tier label.

**`tests/test_classifier.py`:** trains fresh each run (deterministic seed,
sub-second on ~210 examples × 5 features) and asserts:

- held-out accuracy ≥ 0.80 (the guide's V1 bar)
- confusion matrix shape is `(3, 3)` with the fixed label ordering

No persisted report artifact — the pytest assertions are the evidence for
this phase; no Phase 4 dashboard/audit store is introduced.

## 7. Routing map and composition (`costpilot/routing.py`)

- `config/routing.yaml` — tier → model_id mapping against the existing
  `FAKE_MODELS` registry, mirroring the guide's cheapest/mid/highest-quality
  framing:

  ```yaml
  tier_1: claude-haiku
  tier_2: gpt-4o-mini
  tier_3: gpt-4o
  ```

- `load_routing_config(path: Path) -> dict[str, str]` — parses the YAML.
- `route(tier: str, config: dict[str, str]) -> ModelConfig` — looks up the
  tier's `model_id` in `config`, then resolves it against `FAKE_MODELS`;
  raises `ValueError` immediately if the tier is missing from the config or
  the mapped `model_id` doesn't exist in the registry — fail-fast, no silent
  fallback.

**Composition —
`classify_and_route(prompt: str, model, routing_config) -> ModelConfig`:**
chains `predict_tier` then `route`, giving one pure, synchronous function
that demonstrates the full offline routing skeleton end to end. This is
still fully offline and synchronous — no API, no async, no persistence —
consistent with the guide's own framing of Phase 2 as "building the routing
skeleton," not the live router (Phase 5).

**`tests/test_routing.py`:** validates the YAML loads, every tier is
present, every mapped `model_id` resolves in `FAKE_MODELS`, and `route()`
returns the correct `ModelConfig` per tier.

**`tests/test_classify_and_route.py`:** end-to-end — a clearly
Tier-1-shaped prompt routes to the Tier-1 model, and likewise for Tiers 2
and 3.

## 8. Repository layout (Phase 2 additions)

```
llm-cost-autopilot/
  costpilot/
    domain.py, ports.py, providers/fake.py   # unchanged (Phase 1)
    features.py            # extract_features(prompt) -> feature dict
    classifier.py           # dataset loading, split, train, evaluate, predict_tier
    routing.py              # load_routing_config, route, classify_and_route
  data/
    complexity_dataset.json     # 210+ hand-authored (prompt, tier) examples
  config/
    routing.yaml                 # tier -> model_id mapping
  tests/
    test_domain.py, test_providers.py, test_fake_provider.py   # unchanged (Phase 1)
    test_features.py
    test_classifier.py
    test_routing.py
    test_classify_and_route.py
```

## 9. Dependencies

Phase 1 shipped with zero runtime dependencies. Phase 2 adds three, all
fully offline/local (no network calls, no credentials) — this deviates from
the zero-dependency stance but not from the offline-first constraint:

```toml
[project]
dependencies = ["scikit-learn>=1.4,<2", "numpy>=1.26,<2", "pyyaml>=6.0,<7"]
```

## 10. Explicitly deferred (named only — not designed in this document)

- **Phase 3:** async quality verification loop, auto-escalation, classifier
  feedback/retraining loop.
- **Phase 4:** SQLite/JSON audit store, cost dashboard.
- **Phase 5:** FastAPI endpoints.
- **Phase 6:** portfolio load test and report.

## 11. Out of scope / non-claims for this document

- No live provider calls, API keys, or credentials of any kind.
- No verification loop, audit persistence, dashboard, or API surface — these
  are unimplemented and undesigned pending separate approval.
- No GitHub push has occurred as part of writing this spec; publishing
  remains a separate, later approval gate per the project owner's commit
  policy.
- This document does not claim Phase 2 code, dataset, or dependency
  installation exists yet. It is a design record only, to be followed by an
  implementation plan.
- No dataset content has been generated or committed. Section 4's provenance
  constraint (AI-assisted draft with a human-review gate, vs. an
  owner-input gate) is unresolved and blocks the implementation plan's
  dataset task until the project owner decides between the two paths.
