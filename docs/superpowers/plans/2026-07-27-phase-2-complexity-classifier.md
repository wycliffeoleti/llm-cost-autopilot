# Phase 2 Complexity Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build BASWE Project 2's Phase 2 ("Complexity Classifier") on top of the existing Phase 1 unified model interface — complexity tiers, a feature-extraction module, a scikit-learn classifier trained/evaluated on an explicitly-flagged AI-drafted dataset, and a configurable YAML tier-to-model routing map, composed into one offline `classify_and_route` function.

**Architecture:** Three new pure, synchronous `costpilot` modules — `features.py` (regex/keyword feature extraction), `classifier.py` (dataset loading, train/test split, training, evaluation, prediction), `routing.py` (YAML-backed routing map, lookup, and `classify_and_route` composition) — layered on Phase 1's untouched `domain.py`/`ports.py`/`providers/fake.py`. No API, no async, no persistence beyond the flat dataset/config files.

**Tech Stack:** Python 3.11+, scikit-learn (`LogisticRegression`), numpy, PyYAML, pytest. Zero network access anywhere in this phase.

## Global Constraints

- Offline-first: no network calls, no credentials, no live provider/API calls anywhere in this phase.
- Fully deterministic: every random operation (train/test split, classifier fit) uses the fixed seed constant `RANDOM_SEED = 42` defined once in `costpilot/classifier.py`.
- Phase 1 contracts (`costpilot/domain.py`, `costpilot/ports.py`, `costpilot/providers/fake.py`) and their tests are never modified.
- Dataset provenance: `data/complexity_dataset.draft.json` is AI-drafted, unreviewed data. It must carry a top-level `"status": "ai_drafted_pending_human_review"` field, must never be named, described, or committed as "hand-authored"/"human-labeled" ground truth, and no accuracy or confusion-matrix figure computed against it may be described as a real-world routing-quality claim — only as pipeline/prototype validation.
- Excludes Phase 3 (verification loop), Phase 4 (audit/dashboard), Phase 5 (API), Phase 6 (load test/report) entirely.
- Commit policy: every commit is a genuine, independently meaningful unit of completed, verified work (a TDD red-green slice, a dataset batch, a validated fix). Never split one real change into multiple no-op commits to inflate the count, never use empty commits or manipulated timestamps.
- Source: BASWE guide, Phase 2 (PDF p. 7), and `docs/superpowers/specs/2026-07-27-phase-2-complexity-classifier-design.md` (the approved design spec — read it before starting if anything below is ambiguous).

---

## Task 1: Branch and Phase 2 dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `scikit-learn`, `numpy`, `pyyaml` importable as runtime dependencies; `types-PyYAML` importable as a dev dependency; mypy configured to not fail on `sklearn`'s partial type stubs.

- [ ] **Step 1: Create the feature branch**

Run: `git checkout -b feature/phase-2-complexity-classifier`
Expected: branch created and checked out, working tree clean (this repo's `main` is currently clean per `git status`).

- [ ] **Step 2: Add Phase 2 dependencies to `pyproject.toml`**

Change:
```toml
[project]
dependencies = []
```
to:
```toml
[project]
dependencies = ["scikit-learn>=1.4,<2", "numpy>=1.26,<2", "pyyaml>=6.0,<7"]
```

Change:
```toml
[dependency-groups]
dev = ["pytest>=8,<9", "ruff>=0.6,<1", "mypy>=1.11,<2"]
```
to:
```toml
[dependency-groups]
dev = ["pytest>=8,<9", "ruff>=0.6,<1", "mypy>=1.11,<2", "types-PyYAML>=6.0,<7"]
```

Change:
```toml
[tool.mypy]
python_version = "3.11"
strict = true
plugins = []
exclude = ["tests"]
```
to:
```toml
[tool.mypy]
python_version = "3.11"
strict = true
plugins = []
exclude = ["tests"]

[[tool.mypy.overrides]]
module = ["sklearn.*"]
ignore_missing_imports = true
```

- [ ] **Step 3: Install the new dependencies**

Run: `uv sync` (or `pip install -e ".[dev]"` if this project isn't using `uv` — check which lockfile/tool is present in the repo root before choosing).
Expected: `scikit-learn`, `numpy`, `pyyaml`, `types-PyYAML` install without error, no network calls beyond the local package index already configured on this machine.

- [ ] **Step 4: Verify the toolchain still runs cleanly**

Run: `pytest && ruff check . && mypy costpilot`
Expected: all pass (no Phase 2 code exists yet, so this is just confirming the dependency addition didn't break Phase 1).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add scikit-learn, numpy, pyyaml for Phase 2 classifier"
```

---

## Task 2: Feature extraction — `token_count`

**Files:**
- Create: `costpilot/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `token_count(prompt: str) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features.py`:
```python
from costpilot.features import token_count


def test_token_count_counts_whitespace_separated_words():
    assert token_count("one two three") == 3


def test_token_count_handles_single_word():
    assert token_count("hello") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'costpilot.features'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

Create `costpilot/features.py`:
```python
def token_count(prompt: str) -> int:
    return len(prompt.split())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/features.py tests/test_features.py
git commit -m "feat: add token_count feature extractor"
```

---

## Task 3: Feature extraction — `instruction_verb_count`

**Files:**
- Modify: `costpilot/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `instruction_verb_count(prompt: str) -> int`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:
```python
from costpilot.features import instruction_verb_count


def test_instruction_verb_count_counts_known_verbs():
    assert instruction_verb_count("Analyze and compare these two options.") == 2


def test_instruction_verb_count_is_case_insensitive():
    assert instruction_verb_count("ANALYZE this data.") == 1


def test_instruction_verb_count_returns_zero_for_plain_question():
    assert instruction_verb_count("What is the capital of France?") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'instruction_verb_count'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/features.py`:
```python
import re

_INSTRUCTION_VERBS = (
    "analyze", "compare", "evaluate", "synthesize", "design",
    "critique", "contrast", "recommend", "justify", "assess",
)
_INSTRUCTION_VERB_PATTERN = re.compile(
    r"\b(?:" + "|".join(_INSTRUCTION_VERBS) + r")\b", re.IGNORECASE
)


def instruction_verb_count(prompt: str) -> int:
    return len(_INSTRUCTION_VERB_PATTERN.findall(prompt))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/features.py tests/test_features.py
git commit -m "feat: add instruction_verb_count feature extractor"
```

---

## Task 4: Feature extraction — `constraint_count`

**Files:**
- Modify: `costpilot/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `constraint_count(prompt: str) -> int`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:
```python
from costpilot.features import constraint_count


def test_constraint_count_counts_constraint_keywords():
    assert constraint_count("You must include at least three examples.") == 2


def test_constraint_count_counts_bulleted_lines():
    assert constraint_count("- one\n- two\n- three") == 3


def test_constraint_count_returns_zero_for_plain_question():
    assert constraint_count("What is the capital of France?") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'constraint_count'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/features.py`:
```python
_CONSTRAINT_PATTERNS = (
    r"\bmust\b",
    r"\bshould\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bat least\b",
    r"\bexactly\b",
    r"\bno more than\b",
    r"\bbetween\s+\d+\s+and\s+\d+\b",
    r"^\s*[-*]\s+\S.*$",
    r"^\s*\d+[.)]\s+\S.*$",
)
_CONSTRAINT_REGEXES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in _CONSTRAINT_PATTERNS
]


def constraint_count(prompt: str) -> int:
    return sum(len(regex.findall(prompt)) for regex in _CONSTRAINT_REGEXES)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/features.py tests/test_features.py
git commit -m "feat: add constraint_count feature extractor"
```

---

## Task 5: Feature extraction — `has_context`

**Files:**
- Modify: `costpilot/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `has_context(prompt: str) -> bool`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:
```python
from costpilot.features import has_context


def test_has_context_true_for_colon_delimited_data_block():
    prompt = "Extract the total from this invoice: Subtotal $10, Tax $1, Total $11."
    assert has_context(prompt) is True


def test_has_context_true_for_the_following_cue():
    prompt = "Summarize the following report and highlight the key risks it describes in detail."
    assert has_context(prompt) is True


def test_has_context_false_for_plain_question():
    assert has_context("What is the capital of France?") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'has_context'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/features.py`:
```python
_CONTEXT_CUE_PATTERN = re.compile(
    r"\bthe following\b|\bgiven this\b|\bbased on this\b|\baccording to this\b",
    re.IGNORECASE,
)


def has_context(prompt: str) -> bool:
    if _CONTEXT_CUE_PATTERN.search(prompt):
        return True
    if ":" in prompt:
        trailing = prompt.split(":", 1)[1].strip()
        return len(trailing.split()) >= 5
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/features.py tests/test_features.py
git commit -m "feat: add has_context feature extractor"
```

---

## Task 6: Feature extraction — `output_format_complexity`

**Files:**
- Modify: `costpilot/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `output_format_complexity(prompt: str) -> int`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:
```python
from costpilot.features import output_format_complexity


def test_output_format_complexity_strict_for_json_request():
    assert output_format_complexity("Return the result as JSON with fields name and age.") == 2


def test_output_format_complexity_simple_for_list_request():
    assert output_format_complexity("Reformat this into a bulleted list.") == 1


def test_output_format_complexity_zero_for_free_text():
    assert output_format_complexity("What is the capital of France?") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'output_format_complexity'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/features.py`:
```python
_STRICT_FORMAT_PATTERN = re.compile(r"\bjson\b|\bschema\b|\bfields?:", re.IGNORECASE)
_SIMPLE_FORMAT_PATTERN = re.compile(r"\blist\b|\btable\b|\bbulleted?\b", re.IGNORECASE)


def output_format_complexity(prompt: str) -> int:
    if _STRICT_FORMAT_PATTERN.search(prompt):
        return 2
    if _SIMPLE_FORMAT_PATTERN.search(prompt):
        return 1
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS (14 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/features.py tests/test_features.py
git commit -m "feat: add output_format_complexity feature extractor"
```

---

## Task 7: Feature extraction — `extract_features` aggregator

**Files:**
- Modify: `costpilot/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: `token_count`, `instruction_verb_count`, `constraint_count`, `has_context`, `output_format_complexity` (all defined above, same module)
- Produces: `FEATURE_KEYS: tuple[str, ...]`, `extract_features(prompt: str) -> dict[str, float]` — this is what `costpilot/classifier.py` (Task 24) will import and call.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:
```python
from costpilot.features import FEATURE_KEYS, extract_features


def test_extract_features_returns_all_five_keys():
    features = extract_features("Analyze and compare this data.")
    assert set(features.keys()) == set(FEATURE_KEYS)


def test_extract_features_values_are_floats():
    features = extract_features("Analyze and compare this data.")
    assert all(isinstance(value, float) for value in features.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'FEATURE_KEYS'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/features.py`:
```python
FEATURE_KEYS = (
    "token_count",
    "instruction_verb_count",
    "constraint_count",
    "has_context",
    "output_format_complexity",
)


def extract_features(prompt: str) -> dict[str, float]:
    return {
        "token_count": float(token_count(prompt)),
        "instruction_verb_count": float(instruction_verb_count(prompt)),
        "constraint_count": float(constraint_count(prompt)),
        "has_context": float(has_context(prompt)),
        "output_format_complexity": float(output_format_complexity(prompt)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v && mypy costpilot/features.py`
Expected: PASS (16 passed), mypy clean.

- [ ] **Step 5: Commit**

```bash
git add costpilot/features.py tests/test_features.py
git commit -m "feat: add extract_features aggregator"
```

---

## Task 8: Dataset schema validation + draft batch 1 (Tier 1, 14 examples)

**Files:**
- Create: `data/complexity_dataset.draft.json`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `data/complexity_dataset.draft.json` with top-level `version`, `status`, `examples` — the file `costpilot/classifier.py`'s `load_dataset` (Task 24) will read.

**Provenance reminder:** every example below is AI-drafted by the assistant, explicitly pending project-owner review — never described as hand-authored/human-labeled ground truth.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset.py`:
```python
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"
VALID_TIERS = {"tier_1", "tier_2", "tier_3"}


def _load_raw():
    return json.loads(DATASET_PATH.read_text())


def test_dataset_file_has_required_top_level_fields():
    raw = _load_raw()
    assert raw["status"] == "ai_drafted_pending_human_review"
    assert isinstance(raw["examples"], list)
    assert len(raw["examples"]) > 0


def test_dataset_examples_have_valid_tiers_and_unique_ids():
    raw = _load_raw()
    ids = [example["id"] for example in raw["examples"]]
    assert len(ids) == len(set(ids)), "duplicate example id found"
    for example in raw["examples"]:
        assert example["tier"] in VALID_TIERS, f"invalid tier for {example['id']}"
        assert example["text"].strip(), f"empty text for {example['id']}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `FileNotFoundError` (the dataset file doesn't exist yet).

- [ ] **Step 3: Create the dataset file with the first batch (Tier 1, examples c001-c014)**

Create `data/complexity_dataset.draft.json`:
```json
{
  "version": "1.0",
  "status": "ai_drafted_pending_human_review",
  "examples": [
    {"id": "c001", "tier": "tier_1", "text": "Convert this list of names into alphabetical order: Zara, Michael, Amina, Devon, Priya."},
    {"id": "c002", "tier": "tier_1", "text": "Rewrite the following sentence in passive voice: The technician repaired the printer this morning."},
    {"id": "c003", "tier": "tier_1", "text": "Convert these temperatures from Fahrenheit to Celsius: 98.6, 32, 212."},
    {"id": "c004", "tier": "tier_1", "text": "Reformat this run-on sentence into two shorter sentences: The meeting started late because the projector wasn't working and nobody could find the remote so we had to reschedule the demo for the following day."},
    {"id": "c005", "tier": "tier_1", "text": "Turn this paragraph into a numbered list of the three steps it describes: First you preheat the oven, then you mix the dry ingredients, and finally you fold in the wet ingredients before baking."},
    {"id": "c006", "tier": "tier_1", "text": "Convert this date from US format to European format: 03/14/2026."},
    {"id": "c007", "tier": "tier_1", "text": "Rewrite this sentence to remove all contractions: I don't think we'll make it, but we can't be sure yet."},
    {"id": "c008", "tier": "tier_1", "text": "Reformat this comma-separated list into a bulleted list: apples, oranges, bananas, grapes."},
    {"id": "c009", "tier": "tier_1", "text": "Convert this measurement from miles to kilometers: 26.2 miles."},
    {"id": "c010", "tier": "tier_1", "text": "Rewrite this title in title case: the quick fox and the lazy dog."},
    {"id": "c011", "tier": "tier_1", "text": "Reformat this phone number into the standard XXX-XXX-XXXX format: 5551234567."},
    {"id": "c012", "tier": "tier_1", "text": "Convert this currency amount into a plain number without the dollar sign: $1,250.00."},
    {"id": "c013", "tier": "tier_1", "text": "Rewrite this sentence in the past tense: She walks to the store every morning."},
    {"id": "c014", "tier": "tier_1", "text": "Reformat this key-value text into a simple list: name is Alex, age is 29."}
  ]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 14 examples.

- [ ] **Step 5: Commit**

```bash
git add data/complexity_dataset.draft.json tests/test_dataset.py
git commit -m "data: add dataset schema validation + AI-drafted tier_1 batch 1/5 (c001-c014)"
```

---

## Task 9: Dataset draft batch 2 (Tier 1, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing `data/complexity_dataset.draft.json` structure from Task 8
- Produces: running total 28 examples

- [ ] **Step 1: Append 14 more examples to the `examples` array**

Add these entries (after `c014`, before the closing `]`):
```json
    {"id": "c015", "tier": "tier_1", "text": "Convert this 24-hour time to 12-hour format: 14:30."},
    {"id": "c016", "tier": "tier_1", "text": "Rewrite this list in reverse order: one, two, three, four, five."},
    {"id": "c017", "tier": "tier_1", "text": "Reformat this address onto a single line: 123 Maple Street, Springfield, IL, 62704."},
    {"id": "c018", "tier": "tier_1", "text": "Convert this fraction to a decimal: 3/8."},
    {"id": "c019", "tier": "tier_1", "text": "Rewrite this sentence so it starts with Although: The flight was delayed, we still arrived on time."},
    {"id": "c020", "tier": "tier_1", "text": "Reformat this list of numbers in ascending order: 42, 7, 19, 3, 88."},
    {"id": "c021", "tier": "tier_1", "text": "Convert this Celsius temperature to Fahrenheit: 100."},
    {"id": "c022", "tier": "tier_1", "text": "Rewrite this heading in sentence case: HOW TO BAKE BREAD AT HOME."},
    {"id": "c023", "tier": "tier_1", "text": "Reformat these bullet points into a single paragraph: wake up early, exercise, eat breakfast."},
    {"id": "c024", "tier": "tier_1", "text": "Convert this weight from pounds to kilograms: 150 lbs."},
    {"id": "c025", "tier": "tier_1", "text": "Extract the shipping address from this text: John Smith, 42 Baker Street, London, NW1 6XE, United Kingdom."},
    {"id": "c026", "tier": "tier_1", "text": "Pull out all the dates mentioned in this paragraph: The project kicked off on March 3rd, had a checkpoint on April 15th, and is due to wrap up by June 1st."},
    {"id": "c027", "tier": "tier_1", "text": "Extract the total amount due from this invoice text: Subtotal $340.00, Tax $27.20, Total $367.20."},
    {"id": "c028", "tier": "tier_1", "text": "Find the email address in this text: For questions, please reach out to support@examplecorp.com during business hours."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 28 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_1 batch 2/5 (c015-c028)"
```

---

## Task 10: Dataset draft batch 3 (Tier 1, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 9
- Produces: running total 42 examples

- [ ] **Step 1: Append 14 more examples**

```json
    {"id": "c029", "tier": "tier_1", "text": "Extract the flight number from this confirmation text: Your booking is confirmed for flight AA1234 departing at 6:45 AM."},
    {"id": "c030", "tier": "tier_1", "text": "Pull the author's name from this book excerpt: Silent Tides, a novel by Elena Marchetti, explores grief and memory."},
    {"id": "c031", "tier": "tier_1", "text": "Extract the phone number from this text: You can reach the front desk at 555-867-5309 anytime."},
    {"id": "c032", "tier": "tier_1", "text": "Find the product SKU in this description: Wireless Mouse, Model WM-2200, SKU RT-88213."},
    {"id": "c033", "tier": "tier_1", "text": "Extract the meeting time from this email snippet: Let's sync up on Thursday at 3:00 PM in the main conference room."},
    {"id": "c034", "tier": "tier_1", "text": "Pull out the zip code from this address: 1600 Pennsylvania Avenue NW, Washington, DC 20500."},
    {"id": "c035", "tier": "tier_1", "text": "Extract the order number from this receipt text: Thank you for your purchase, order A29384 has shipped."},
    {"id": "c036", "tier": "tier_1", "text": "Find the temperature reading in this sensor log line: timestamp 2026-03-01T08:00:00 sensor_id 12 reading 21.4C status ok."},
    {"id": "c037", "tier": "tier_1", "text": "Extract the CEO's name from this press release: Acme Robotics announced today that CEO Priya Nair will present at the upcoming tech summit."},
    {"id": "c038", "tier": "tier_1", "text": "Pull the ISBN from this book listing: The Silent Wave, Hardcover, ISBN 978-1-234567-89-0, Published 2024."},
    {"id": "c039", "tier": "tier_1", "text": "Extract the discount percentage from this promo text: Use code SAVE20 for 20% off your next order through Friday."},
    {"id": "c040", "tier": "tier_1", "text": "Find the error code in this log line: ERROR 503 Service Unavailable, upstream connection refused."},
    {"id": "c041", "tier": "tier_1", "text": "Extract the checkout time from this hotel confirmation: Check-in 3:00 PM, Check-out 11:00 AM, Room type Deluxe King."},
    {"id": "c042", "tier": "tier_1", "text": "Pull the license plate number from this text: The reported vehicle had license plate number 7XKT204."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 42 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_1 batch 3/5 (c029-c042)"
```

---

## Task 11: Dataset draft batch 4 (Tier 1, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 10
- Produces: running total 56 examples

- [ ] **Step 1: Append 14 more examples**

```json
    {"id": "c043", "tier": "tier_1", "text": "Extract the interest rate from this loan summary: Loan amount $15,000, APR 6.25%, Term 60 months."},
    {"id": "c044", "tier": "tier_1", "text": "Find the tracking number in this shipping notification: Your package has shipped via UPS, tracking number 1Z999AA10123456784."},
    {"id": "c045", "tier": "tier_1", "text": "Extract the job title from this profile snippet: Maria Chen, Senior Data Engineer at Northwind Analytics."},
    {"id": "c046", "tier": "tier_1", "text": "Pull the currency code from this transaction record: Amount 250.00 EUR, Merchant Cafe Lumiere, Date 2026-02-14."},
    {"id": "c047", "tier": "tier_1", "text": "Extract the room number from this hotel note: Guest has been assigned to Room 412 on the fourth floor."},
    {"id": "c048", "tier": "tier_1", "text": "What is the capital of Japan?"},
    {"id": "c049", "tier": "tier_1", "text": "Based on this paragraph, what year did the company relocate its headquarters? Founded in 1998, the company grew steadily and relocated its headquarters to Austin in 2015 to be closer to its engineering talent pool."},
    {"id": "c050", "tier": "tier_1", "text": "According to this text, who is the CEO of the company? Northwind Robotics, led by CEO David Okafor since 2020, specializes in warehouse automation."},
    {"id": "c051", "tier": "tier_1", "text": "How many legs does a spider have?"},
    {"id": "c052", "tier": "tier_1", "text": "Based on this weather report, will it rain tomorrow? Tomorrow's forecast shows a 70 percent chance of rain with highs near 60F and gusty winds in the afternoon."},
    {"id": "c053", "tier": "tier_1", "text": "What is the boiling point of water at sea level in Celsius?"},
    {"id": "c054", "tier": "tier_1", "text": "According to this excerpt, what caused the delay in the shipment? The shipment was delayed due to a customs inspection triggered by incomplete paperwork on the exporter's side."},
    {"id": "c055", "tier": "tier_1", "text": "Who wrote the novel Pride and Prejudice?"},
    {"id": "c056", "tier": "tier_1", "text": "Based on this passage, how many employees does the company have? With just under 500 employees spread across three offices, the company has doubled in size since its last funding round."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 56 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_1 batch 4/5 (c043-c056)"
```

---

## Task 12: Dataset draft batch 5 (Tier 1, 14 more examples — completes Tier 1)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 11
- Produces: running total 70 examples (Tier 1 complete)

- [ ] **Step 1: Append 14 more examples**

```json
    {"id": "c057", "tier": "tier_1", "text": "What is the chemical symbol for gold?"},
    {"id": "c058", "tier": "tier_1", "text": "According to this bio, where did the author grow up? Born in rural Kenya, the author later moved to London to pursue a career in journalism."},
    {"id": "c059", "tier": "tier_1", "text": "What is the largest planet in our solar system?"},
    {"id": "c060", "tier": "tier_1", "text": "Based on this paragraph, what triggered the recall? The recall was issued after several units were found to have a faulty battery connector that posed a fire risk."},
    {"id": "c061", "tier": "tier_1", "text": "Who is credited with inventing the telephone?"},
    {"id": "c062", "tier": "tier_1", "text": "According to this schedule, what time does the last train depart? Trains run every 20 minutes until the last departure at 11:40 PM."},
    {"id": "c063", "tier": "tier_1", "text": "What is the freezing point of water in Fahrenheit?"},
    {"id": "c064", "tier": "tier_1", "text": "Based on this text, what was the main reason for the merger? The two companies merged primarily to combine their complementary technology stacks and expand into new markets together."},
    {"id": "c065", "tier": "tier_1", "text": "How many continents are there on Earth?"},
    {"id": "c066", "tier": "tier_1", "text": "According to this memo, who approved the budget increase? The revised budget was approved by the finance director following a review of Q3 spending."},
    {"id": "c067", "tier": "tier_1", "text": "What year did World War II end?"},
    {"id": "c068", "tier": "tier_1", "text": "Based on this passage, what is the warranty period for the product? The product comes with a standard one-year warranty covering manufacturing defects, extendable to three years for an additional fee."},
    {"id": "c069", "tier": "tier_1", "text": "What is the currency used in Japan?"},
    {"id": "c070", "tier": "tier_1", "text": "According to this note, why was the flight rescheduled? The flight was rescheduled due to a mechanical issue discovered during the pre-flight inspection."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 70 examples, Tier 1 complete.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_1 batch 5/5 (c057-c070) - tier_1 complete"
```

---

## Task 13: Dataset draft batch 6 (Tier 2, 14 examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 12
- Produces: running total 84 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c071", "tier": "tier_2", "text": "Summarize this article in three sentences: Renewable energy adoption has accelerated globally over the past decade, driven by falling solar and wind costs, supportive government policy, and growing corporate demand for clean power. Grid operators are increasingly investing in battery storage to manage the intermittency of these sources. Analysts expect renewables to account for the majority of new electricity generation capacity added worldwide through the end of the decade."},
    {"id": "c072", "tier": "tier_2", "text": "Provide a one-paragraph summary of this meeting transcript: The team discussed Q2 roadmap priorities, agreeing to delay the mobile redesign in favor of shipping the new billing system first. Engineering raised concerns about staffing, and the group agreed to revisit headcount needs in the next planning cycle. Marketing requested earlier visibility into feature timelines going forward."},
    {"id": "c073", "tier": "tier_2", "text": "Summarize the key findings of this research abstract in two sentences: This study examined sleep patterns in 300 shift workers over six months, finding that irregular shift schedules were associated with a 40 percent higher rate of reported insomnia compared to fixed schedules. Interventions involving controlled light exposure modestly improved sleep quality in the irregular-shift group."},
    {"id": "c074", "tier": "tier_2", "text": "Condense this customer support ticket thread into a short summary for the account manager: Customer reported the app crashing on login. Support asked for device info and logs. Customer provided an iOS 17 device and crash logs. Engineering identified a token refresh bug and shipped a fix in the next release."},
    {"id": "c075", "tier": "tier_2", "text": "Summarize this product changelog entry for a non-technical audience: Version 3.4.0 introduces lazy-loading for the dashboard's chart widgets, reducing initial page load time by roughly 35 percent, along with a fix for a memory leak in the real-time notification service and improved error messages across the settings pages."},
    {"id": "c076", "tier": "tier_2", "text": "Write a brief summary of this legal notice for a general audience: This notice informs affected users that a data processing vendor experienced an unauthorized access incident between January 3 and January 9, potentially exposing names and email addresses, though no financial information was involved."},
    {"id": "c077", "tier": "tier_2", "text": "Summarize the plot of this short story synopsis in one paragraph: A retired lighthouse keeper discovers a message in a bottle that leads her back to a childhood friend she believed had died at sea decades earlier, forcing her to confront long-buried guilt over their final conversation."},
    {"id": "c078", "tier": "tier_2", "text": "Provide a two-sentence summary of this quarterly earnings statement: Revenue grew 12 percent year over year to $340 million, driven primarily by strength in the subscription segment, while operating margin narrowed slightly due to increased investment in customer support headcount."},
    {"id": "c079", "tier": "tier_2", "text": "Summarize this employee handbook section on remote work: Employees may work remotely up to three days per week with manager approval, must remain reachable during core hours of 10 AM to 3 PM local time, and are responsible for maintaining a secure home network setup."},
    {"id": "c080", "tier": "tier_2", "text": "Condense this scientific paper's methodology section into plain language: Participants were randomly assigned to either a control group receiving standard care or a treatment group receiving a twice-weekly telehealth check-in, with outcomes measured via a validated symptom questionnaire administered at baseline, six weeks, and twelve weeks."},
    {"id": "c081", "tier": "tier_2", "text": "Summarize the main complaint in this customer review: I ordered the blue version but received the black one, and when I called support I was on hold for over 40 minutes before anyone answered, and even then they couldn't tell me when a replacement would ship."},
    {"id": "c082", "tier": "tier_2", "text": "Provide a short summary of this incident report for leadership: At 2:14 AM, the payments service began returning elevated error rates due to a database connection pool exhaustion issue. On-call engineers restored service by 2:41 AM after scaling the connection pool and restarting the affected pods."},
    {"id": "c083", "tier": "tier_2", "text": "Summarize this travel itinerary in a few sentences: Day one, arrive in Lisbon, check into hotel, walk the Alfama district. Day two, day trip to Sintra to see the palaces. Day three, free day for shopping before an evening flight home."},
    {"id": "c084", "tier": "tier_2", "text": "Condense this policy document excerpt about expense reimbursement: Employees must submit itemized receipts for any expense over $25 within 30 days of the purchase, and reimbursements for travel booked outside the approved corporate travel portal require prior manager sign-off."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 84 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_2 batch 1/5 (c071-c084)"
```

---

## Task 14: Dataset draft batch 7 (Tier 2, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 13
- Produces: running total 98 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c085", "tier": "tier_2", "text": "Summarize the central argument of this opinion column: The author argues that remote work has not reduced overall productivity but has instead shifted where and when work gets done, and that companies mandating a full return to office are conflating visibility with output."},
    {"id": "c086", "tier": "tier_2", "text": "Provide a brief summary of this software release notes excerpt for the support team: This release patches a security vulnerability in the authentication module, adds two-factor authentication support for enterprise accounts, and deprecates the legacy API endpoints scheduled for removal next quarter."},
    {"id": "c087", "tier": "tier_2", "text": "Summarize this survey result paragraph in one sentence: Of the 1,200 respondents, 68 percent said they would recommend the product to a colleague, though satisfaction scores were notably lower among customers on the free tier compared to paid subscribers."},
    {"id": "c088", "tier": "tier_2", "text": "Condense this history lecture excerpt into a short summary: The excerpt describes how the printing press, invented in the mid-15th century, dramatically lowered the cost of producing books and is widely credited with accelerating literacy rates and the spread of new ideas across Europe."},
    {"id": "c089", "tier": "tier_2", "text": "Summarize this restaurant review in two sentences: The tasting menu was inventive and beautifully plated, though the pacing between courses was inconsistent, with a nearly 45-minute wait before dessert that left several tables visibly frustrated."},
    {"id": "c090", "tier": "tier_2", "text": "Provide a plain-language summary of this insurance claim status update: Your claim has been reviewed and approved for partial coverage of the water damage repairs, with the remaining balance due to a pre-existing roof condition that falls outside the policy's coverage terms."},
    {"id": "c091", "tier": "tier_2", "text": "Summarize this product recall announcement: The manufacturer is recalling approximately 40,000 units of the model X200 space heater after receiving reports of overheating during extended use, and is offering free replacement units to affected customers."},
    {"id": "c092", "tier": "tier_2", "text": "Condense this internal memo about the office relocation: Starting next quarter, the downtown office will consolidate into the new building on Fifth Avenue, with parking passes reissued automatically and a phased moving schedule by department over three weeks."},
    {"id": "c093", "tier": "tier_2", "text": "Summarize the key takeaway from this podcast episode description: In this episode, the hosts interview a former air traffic controller about decision-making under pressure, drawing parallels to how software teams handle high-stakes production incidents."},
    {"id": "c094", "tier": "tier_2", "text": "Provide a short summary of this weather advisory: A winter storm warning is in effect from 6 PM tonight through noon tomorrow, with 8 to 14 inches of snow expected and localized power outages possible due to high winds."},
    {"id": "c095", "tier": "tier_2", "text": "Classify this email as spam or not spam: Congratulations, you've been selected to receive a free gift card. Click here now to claim your prize before it expires."},
    {"id": "c096", "tier": "tier_2", "text": "Categorize this customer review as positive, negative, or neutral: The product works exactly as described and arrived a day early. Nothing special, but it does the job."},
    {"id": "c097", "tier": "tier_2", "text": "Determine whether this support ticket is a bug report, feature request, or general question: It would be great if the app let me export my data as a CSV file."},
    {"id": "c098", "tier": "tier_2", "text": "Classify this news headline by topic, choosing politics, sports, technology, or entertainment: Local council approves new zoning rules for downtown development."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 98 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_2 batch 2/5 (c085-c098)"
```

---

## Task 15: Dataset draft batch 8 (Tier 2, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 14
- Produces: running total 112 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c099", "tier": "tier_2", "text": "Categorize this tweet's sentiment as positive, negative, or neutral: Just tried the new update and honestly it's fine, nothing has really changed for me either way."},
    {"id": "c100", "tier": "tier_2", "text": "Determine whether this transaction should be flagged as potentially fraudulent: A $2,400 purchase was made at 3 AM from a device location 900 miles from the cardholder's registered address, immediately followed by three smaller test transactions."},
    {"id": "c101", "tier": "tier_2", "text": "Classify this job posting by seniority level, choosing junior, mid, senior, or staff: We're looking for someone with 8 plus years of experience to define our platform architecture and mentor a team of six engineers."},
    {"id": "c102", "tier": "tier_2", "text": "Categorize this product return reason, choosing defective, wrong item, changed mind, or other: The item arrived with a cracked screen right out of the box."},
    {"id": "c103", "tier": "tier_2", "text": "Determine the priority level of this incident report, choosing low, medium, high, or critical: Users are unable to log in to the production application, the issue is affecting all customers."},
    {"id": "c104", "tier": "tier_2", "text": "Classify this restaurant review as a complaint about food, service, or ambiance: The waiter never came back to refill our drinks and it took 20 minutes to get the check."},
    {"id": "c105", "tier": "tier_2", "text": "Categorize this loan application as approved, denied, or needs manual review: Applicant has a 610 credit score, stable income, but a recent 30-day late payment on an existing account."},
    {"id": "c106", "tier": "tier_2", "text": "Determine whether this social media post violates the community guideline against harassment: Great points in the article, though I disagree with the third paragraph's conclusion."},
    {"id": "c107", "tier": "tier_2", "text": "Classify this software bug report by severity, choosing cosmetic, minor, major, or blocker: The submit button is slightly misaligned on mobile screens smaller than 375 pixels wide."},
    {"id": "c108", "tier": "tier_2", "text": "Categorize this email as urgent or non-urgent: Just a reminder that the team lunch has been moved to Friday at noon."},
    {"id": "c109", "tier": "tier_2", "text": "Determine whether this expense report line item is reimbursable per typical corporate travel policy: Dinner for one at a mid-range restaurant during a business trip, $42 total."},
    {"id": "c110", "tier": "tier_2", "text": "Classify this customer inquiry as a billing question, technical issue, or account request: I was charged twice for my subscription this month and need one of the charges reversed."},
    {"id": "c111", "tier": "tier_2", "text": "Categorize this research paper's methodology as qualitative, quantitative, or mixed methods: The study surveyed 500 participants using a Likert-scale questionnaire and also conducted in-depth interviews with a subset of 20 respondents."},
    {"id": "c112", "tier": "tier_2", "text": "Determine whether this online review appears to be genuine or likely fake: Best product ever, five stars, buy now, changed my life completely."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 112 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_2 batch 3/5 (c099-c112)"
```

---

## Task 16: Dataset draft batch 9 (Tier 2, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 15
- Produces: running total 126 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c113", "tier": "tier_2", "text": "Classify this maintenance request, choosing plumbing, electrical, HVAC, or general: The upstairs bathroom faucet has been dripping constantly for the past two days."},
    {"id": "c114", "tier": "tier_2", "text": "Categorize this job candidate's interview feedback, choosing strong hire, hire, no hire, or strong no hire: Solid technical skills and clear communication, but struggled with the system design portion and seemed underprepared for that section."},
    {"id": "c115", "tier": "tier_2", "text": "Determine the sentiment trend of this stock analyst note, choosing bullish, bearish, or neutral: While near-term headwinds from supply chain costs persist, we see the company's long-term positioning in the market as increasingly favorable."},
    {"id": "c116", "tier": "tier_2", "text": "Classify this insurance claim type: The claimant reports a rear-end collision at a stoplight resulting in bumper damage and no injuries."},
    {"id": "c117", "tier": "tier_2", "text": "Categorize this forum post's tone, choosing seeking advice, venting, or sharing an update: Just wanted to share that after months of sleep training, our toddler is finally sleeping through the night."},
    {"id": "c118", "tier": "tier_2", "text": "Analyze this sales data and identify the top three trends: Q1 sales, Electronics $120K, Apparel $85K, Home Goods $60K. Q2 sales, Electronics $95K, Apparel $110K, Home Goods $72K. Q3 sales, Electronics $130K, Apparel $102K, Home Goods $58K."},
    {"id": "c119", "tier": "tier_2", "text": "Break down the pros and cons of remote work based on this survey data: 62 percent of respondents reported higher job satisfaction working remotely, while 45 percent reported feeling less connected to their team, and 30 percent cited difficulty separating work from personal life."},
    {"id": "c120", "tier": "tier_2", "text": "Analyze this website traffic report and identify which channel is underperforming relative to its cost: Paid search, 40 percent of spend, 25 percent of conversions. Organic search, 10 percent of spend, 35 percent of conversions. Social ads, 50 percent of spend, 40 percent of conversions."},
    {"id": "c121", "tier": "tier_2", "text": "Break down this budget spreadsheet into categories and flag any line item that exceeds 20 percent of total spend: Marketing $45,000. Engineering $180,000. Sales $60,000. Operations $35,000. Total $320,000."},
    {"id": "c122", "tier": "tier_2", "text": "Analyze this customer churn data and identify the strongest predictor of cancellation: Customers who contacted support more than twice churned at a 38 percent rate, versus 12 percent for customers with zero or one support contact. Tenure and plan tier showed weaker correlation with churn."},
    {"id": "c123", "tier": "tier_2", "text": "Break down the risk factors described in this project status report: The project is currently two weeks behind schedule due to a key vendor delay, has used 70 percent of its budget with 40 percent of scope remaining, and has one open dependency on a team outside the project."},
    {"id": "c124", "tier": "tier_2", "text": "Analyze this employee engagement survey and summarize the three lowest-scoring categories: Compensation 3.2 out of 5. Career growth 2.8 out of 5. Management support 3.9 out of 5. Work-life balance 3.1 out of 5. Recognition 2.6 out of 5. Team collaboration 4.1 out of 5."},
    {"id": "c125", "tier": "tier_2", "text": "Break down this A/B test result and determine which variant performed better and by how much: Variant A, 2,000 visitors, 180 signups. Variant B, 2,050 visitors, 215 signups."},
    {"id": "c126", "tier": "tier_2", "text": "Analyze this server performance log summary and identify the likely bottleneck: CPU utilization averaged 35 percent, memory utilization averaged 42 percent, and database query latency spiked to 1,800 milliseconds during peak traffic while application response time remained under 200 milliseconds outside of database calls."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 126 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_2 batch 4/5 (c113-c126)"
```

---

## Task 17: Dataset draft batch 10 (Tier 2, 14 more examples — completes Tier 2)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 16
- Produces: running total 140 examples (Tier 2 complete)

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c127", "tier": "tier_2", "text": "Break down this competitive pricing table and identify where the company is positioned relative to competitors: Our product $49 per month. Competitor A $39 per month with fewer features. Competitor B $59 per month with a similar feature set. Competitor C $45 per month with a usage cap."},
    {"id": "c128", "tier": "tier_2", "text": "Analyze this hiring funnel data and identify the stage with the largest drop-off: Applications 500. Phone screens 120. Technical interviews 60. Onsite interviews 25. Offers 10. Accepted 8."},
    {"id": "c129", "tier": "tier_2", "text": "Break down this expense report by category and calculate the percentage spent on travel: Airfare $850. Hotel $600. Meals $220. Ground transportation $130. Total $1,800."},
    {"id": "c130", "tier": "tier_2", "text": "Analyze this feature usage data and identify which features are candidates for deprecation due to low adoption: Dashboard, 92 percent of users active monthly. Reports export, 8 percent of users active monthly. Custom alerts, 65 percent of users active monthly. Legacy import tool, 3 percent of users active monthly."},
    {"id": "c131", "tier": "tier_2", "text": "Break down this warehouse inventory report and flag SKUs at risk of stockout within two weeks based on current sell-through rate: SKU A, 40 units on hand, selling 5 per day. SKU B, 200 units on hand, selling 3 per day. SKU C, 15 units on hand, selling 4 per day."},
    {"id": "c132", "tier": "tier_2", "text": "Analyze this marketing campaign performance summary across channels and identify the best-performing one by cost per acquisition: Email, $8,000 spend, 400 conversions. Social, $15,000 spend, 500 conversions. Search, $20,000 spend, 800 conversions."},
    {"id": "c133", "tier": "tier_2", "text": "Break down this patient satisfaction survey by department and identify the department needing the most improvement: Cardiology 4.5 out of 5. Emergency 3.1 out of 5. Pediatrics 4.7 out of 5. Radiology 4.0 out of 5."},
    {"id": "c134", "tier": "tier_2", "text": "Analyze this energy usage report and identify the time period with the highest consumption relative to typical baseline: Weekday daytime, 120 percent of baseline. Weekday evening, 140 percent of baseline. Weekend daytime, 90 percent of baseline. Weekend evening, 160 percent of baseline."},
    {"id": "c135", "tier": "tier_2", "text": "Break down this supply chain delay report and identify the single largest contributing factor: Delays attributed to customs processing 45 percent, supplier production capacity 30 percent, freight carrier availability 15 percent, internal order processing 10 percent."},
    {"id": "c136", "tier": "tier_2", "text": "Analyze this classroom test score distribution and identify whether the results suggest the test was too difficult: Out of 30 students, 4 scored above 90 percent, 8 scored 70 to 89 percent, 12 scored 50 to 69 percent, and 6 scored below 50 percent."},
    {"id": "c137", "tier": "tier_2", "text": "Break down this app store review data by star rating and summarize the most common complaint among 1-star reviews: 1-star reviews, 45 total, 30 mention crashes on startup, 10 mention poor customer support, 5 mention pricing."},
    {"id": "c138", "tier": "tier_2", "text": "Analyze this quarterly headcount report and identify which department grew fastest in percentage terms: Engineering 40 to 52. Sales 20 to 24. Support 15 to 25. Marketing 10 to 11."},
    {"id": "c139", "tier": "tier_2", "text": "Break down this donor contribution report for the nonprofit and identify what percentage of total funds came from the top 10 donors: Total raised $500,000. Top 10 donors contributed a combined $310,000, with the remaining amount coming from 1,200 smaller individual donations."},
    {"id": "c140", "tier": "tier_2", "text": "Analyze this website A/B test on page load time and conversion rate, and determine whether the data supports a causal link: Pages loading under 2 seconds converted at 4.8 percent. Pages loading 2 to 4 seconds converted at 3.9 percent. Pages loading over 4 seconds converted at 2.1 percent."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 140 examples, Tier 2 complete.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_2 batch 5/5 (c127-c140) - tier_2 complete"
```

---

## Task 18: Dataset draft batch 11 (Tier 3, 14 examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 17
- Produces: running total 154 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c141", "tier": "tier_3", "text": "Given these three job offers with different salary, equity, and location terms, walk through the trade-offs step by step and recommend one: Offer A, $150K salary, 0.1 percent equity, fully remote. Offer B, $135K salary, 0.4 percent equity, hybrid in a high cost-of-living city. Offer C, $160K salary, no equity, requires relocation to a lower cost-of-living city."},
    {"id": "c142", "tier": "tier_3", "text": "Diagnose the likely root cause of this multi-service outage given these log excerpts, explaining your chain of reasoning: Service A logs show a spike in timeouts calling Service B at 14:02. Service B logs show its database connection pool exhausted starting at 14:00. A deployment to Service B's config went out at 13:58 reducing its max connections."},
    {"id": "c143", "tier": "tier_3", "text": "Walk through the trade-offs of these two database migration strategies and recommend one given the constraints: The team has a 4-hour maintenance window, the dataset is 2TB, and the application cannot tolerate more than 10 minutes of write downtime. One option is a blue-green cutover, the other is an online schema migration tool."},
    {"id": "c144", "tier": "tier_3", "text": "Given this set of interdependent project tasks and their durations, determine the critical path and explain your reasoning: Task A, 3 days, must finish before Task B, 5 days, and Task C, 2 days, can start. Task B and Task C must both finish before Task D, 1 day, can start."},
    {"id": "c145", "tier": "tier_3", "text": "Reason through this pricing change scenario and predict the likely effect on both revenue and customer churn: The company plans to raise its base subscription price by 15 percent while adding a lower-tier plan at the old price point with reduced features."},
    {"id": "c146", "tier": "tier_3", "text": "Given these conflicting sensor readings from a manufacturing line, determine which sensor is most likely malfunctioning and explain your reasoning: Sensor 1 reports temperature steady at 72F for six hours. Sensor 2 shows temperature fluctuating between 68F and 90F over the same period. The physical process being measured is known to run at a stable temperature."},
    {"id": "c147", "tier": "tier_3", "text": "Walk through how you would allocate a limited $50,000 marketing budget across three channels given this performance data, and justify your allocation: Channel A returns $3 per $1 spent up to a $20,000 cap before diminishing returns. Channel B returns $2.50 per $1 spent with no known cap. Channel C is untested but has strong anecdotal signal."},
    {"id": "c148", "tier": "tier_3", "text": "Given this set of symptoms and lab results, reason through the most likely explanation while noting what additional information would help confirm it: Patient reports fatigue and cold intolerance over three months. Lab results show elevated TSH and low free T4."},
    {"id": "c149", "tier": "tier_3", "text": "Reason step by step through whether this proposed office layout change will likely increase or decrease cross-team collaboration, given this description: The plan replaces individual desks with a hybrid of hot-desking pods and dedicated team neighborhoods, removing assigned seating for individual contributors but keeping fixed seating for team leads."},
    {"id": "c150", "tier": "tier_3", "text": "Given these two competing vendor proposals for a software contract, walk through a structured comparison and recommend one: Vendor X offers a lower upfront cost but a 3-year lock-in and limited customization. Vendor Y costs 20 percent more upfront but offers month-to-month terms and an open API for integrations."},
    {"id": "c151", "tier": "tier_3", "text": "Diagnose why this A/B test's results might be misleading despite showing a statistically significant winner, given this description: The test ran for only 3 days, included a site-wide holiday sale during that window, and the winning variant was shown disproportionately to returning customers due to a caching bug."},
    {"id": "c152", "tier": "tier_3", "text": "Walk through the second-order effects of this policy change on customer behavior: The company is removing free returns and replacing them with a flat $6.99 return shipping fee for all online orders."},
    {"id": "c153", "tier": "tier_3", "text": "Given this incident timeline, determine at which step the escalation process broke down and explain why: Alert fired at 1:00 AM but was routed to a deprecated on-call rotation. A customer complaint was filed at 1:45 AM but not linked to the outage until 3:15 AM, when an engineer noticed the pattern manually."},
    {"id": "c154", "tier": "tier_3", "text": "Reason through the likely long-term consequences of this staffing decision: The team is disbanding its dedicated QA function and distributing testing responsibilities across the existing engineers without adding headcount."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 154 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_3 batch 1/5 (c141-c154)"
```

---

## Task 19: Dataset draft batch 12 (Tier 3, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 18
- Produces: running total 168 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c155", "tier": "tier_3", "text": "Given this contract dispute scenario, walk through the reasoning both parties might use to support their position: A freelance designer delivered work two weeks late due to the client changing requirements midway through the project, and the client is now withholding the final payment citing the missed deadline."},
    {"id": "c156", "tier": "tier_3", "text": "Determine the most likely bottleneck in this manufacturing process given this description of throughput at each stage, and explain your reasoning: Raw material intake, 1,000 units per day capacity. Assembly, 600 units per day capacity. Quality inspection, 900 units per day capacity. Packaging, 1,200 units per day capacity."},
    {"id": "c157", "tier": "tier_3", "text": "Walk through how currency exchange rate volatility would affect this company's quarterly earnings given this exposure: 60 percent of revenue is collected in euros while 90 percent of costs are paid in US dollars, and the euro has weakened 8 percent against the dollar this quarter."},
    {"id": "c158", "tier": "tier_3", "text": "Given these two conflicting eyewitness accounts of the same incident, reason through what might explain the discrepancy: Witness A says the car was traveling fast and ran the red light. Witness B, standing on the opposite corner, says the light was yellow when the car entered the intersection."},
    {"id": "c159", "tier": "tier_3", "text": "Reason through the implications of this proposed change to a recommendation algorithm: The team wants to weight recent user engagement twice as heavily as historical engagement when generating personalized recommendations."},
    {"id": "c160", "tier": "tier_3", "text": "Walk through the trade-offs of centralizing versus keeping decentralized this company's customer support function, given this context: The company operates in five regions with different languages and time zones, and currently has a support team embedded in each regional office."},
    {"id": "c161", "tier": "tier_3", "text": "Given this experiment design, identify a potential confounding variable and explain how it could bias the results: Researchers compared productivity between employees who opted into a new four-day workweek pilot and those who did not, measuring output over the following quarter."},
    {"id": "c162", "tier": "tier_3", "text": "Determine the most cost-effective way to reduce this data center's cooling costs given these options and constraints: Option 1, upgrade to more efficient cooling units, $200K upfront, 15 percent reduction in cooling energy use. Option 2, relocate to a cooler climate region, $2M upfront, 40 percent reduction. Option 3, adjust airflow containment, $30K upfront, 8 percent reduction."},
    {"id": "c163", "tier": "tier_3", "text": "Walk through the reasoning for whether this startup should pursue an enterprise sales motion or a self-serve motion given this product and customer profile: The product requires significant configuration to fit a customer's existing workflows and the average deal size under early trials has been around $40,000 annually."},
    {"id": "c164", "tier": "tier_3", "text": "Given this described network topology and the reported symptom, trace through where the fault is most likely occurring: Users in Office A can reach the internal file server but not the internet. Users in Office B can reach both. Both offices share the same core router but have separate internet gateways."},
    {"id": "c165", "tier": "tier_3", "text": "Write a short story about a robot who discovers music, in the style of a fairy tale."},
    {"id": "c166", "tier": "tier_3", "text": "Compose a persuasive product launch email for a new productivity app targeting busy freelancers."},
    {"id": "c167", "tier": "tier_3", "text": "Write a two-stanza poem about the feeling of returning home after a long trip."},
    {"id": "c168", "tier": "tier_3", "text": "Draft a toast for a colleague's retirement after 30 years at the same company, striking a warm and slightly humorous tone."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 168 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_3 batch 2/5 (c155-c168)"
```

---

## Task 20: Dataset draft batch 13 (Tier 3, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 19
- Produces: running total 182 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c169", "tier": "tier_3", "text": "Write a short dialogue between two rival chefs meeting unexpectedly at a farmers market."},
    {"id": "c170", "tier": "tier_3", "text": "Compose a brief tribute for a beloved family dog, focusing on the joy it brought the household."},
    {"id": "c171", "tier": "tier_3", "text": "Write a scene where a detective realizes the key witness has been lying the entire time."},
    {"id": "c172", "tier": "tier_3", "text": "Draft an imaginative opening paragraph for a science fiction novel set on a generation ship that has lost contact with Earth."},
    {"id": "c173", "tier": "tier_3", "text": "Write a persuasive op-ed arguing that cities should redesign downtown areas around pedestrians rather than cars."},
    {"id": "c174", "tier": "tier_3", "text": "Compose a short fable, in the tradition of Aesop, about a fox who learns the value of patience."},
    {"id": "c175", "tier": "tier_3", "text": "Write a monologue for a character who has just decided to leave their hometown for good."},
    {"id": "c176", "tier": "tier_3", "text": "Draft a whimsical bedtime story about a cloud who is afraid of rain."},
    {"id": "c177", "tier": "tier_3", "text": "Write a suspenseful opening scene for a heist story set inside a museum after closing hours."},
    {"id": "c178", "tier": "tier_3", "text": "Compose a heartfelt wedding speech from a sibling of the bride, including one embarrassing childhood memory."},
    {"id": "c179", "tier": "tier_3", "text": "Write a short piece of flash fiction, no more than 150 words, about a lighthouse keeper's last night on the job before automation replaces the role."},
    {"id": "c180", "tier": "tier_3", "text": "Draft a motivational speech for a youth soccer team before a championship match they are expected to lose."},
    {"id": "c181", "tier": "tier_3", "text": "Write a comedic scene where two coworkers are stuck in an elevator during a fire drill."},
    {"id": "c182", "tier": "tier_3", "text": "Compose a reflective personal essay opening about the first time you failed at something that mattered to you."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 182 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_3 batch 3/5 (c169-c182)"
```

---

## Task 21: Dataset draft batch 14 (Tier 3, 14 more examples)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 20
- Produces: running total 196 examples

- [ ] **Step 1: Append 14 examples**

```json
    {"id": "c183", "tier": "tier_3", "text": "Write a short story from the point of view of a house that has just been sold after 40 years with the same family."},
    {"id": "c184", "tier": "tier_3", "text": "Draft an imaginative product description for a fictional gadget that translates a pet's body language into plain English."},
    {"id": "c185", "tier": "tier_3", "text": "Write a tense negotiation scene between two business partners on the verge of dissolving their long-running partnership."},
    {"id": "c186", "tier": "tier_3", "text": "Compose a short letter from a time traveler to their past self, warning about one specific mistake without giving too much away."},
    {"id": "c187", "tier": "tier_3", "text": "Write an evocative travel-blog-style paragraph describing the first morning of a solo trip to a remote mountain village."},
    {"id": "c188", "tier": "tier_3", "text": "An employee has been late to work five times this month due to a documented medical condition. Should this count against their performance review? Justify your reasoning considering both fairness and policy consistency."},
    {"id": "c189", "tier": "tier_3", "text": "Evaluate whether this marketing claim is misleading, weighing the literal wording against the likely consumer interpretation: Our supplement is clinically proven to support energy levels, based on a single unpublished internal study with 12 participants."},
    {"id": "c190", "tier": "tier_3", "text": "A manager wants to promote a highly skilled engineer who is difficult to work with over a less technically strong but highly collaborative peer. Weigh the considerations and offer a recommendation."},
    {"id": "c191", "tier": "tier_3", "text": "Assess whether it is ethical for a company to use a customer's past purchase history to show them a higher price than a new customer would see for the identical product, and explain the reasoning on both sides."},
    {"id": "c192", "tier": "tier_3", "text": "A whistleblower has revealed that a mid-level manager falsified a minor compliance report to meet a deadline, with no material harm resulting. Weigh how the company should respond, balancing accountability with proportionality."},
    {"id": "c193", "tier": "tier_3", "text": "Evaluate the fairness of a school's policy that bases valedictorian selection solely on GPA, without accounting for differences in course difficulty between students, and suggest whether adjustment is warranted."},
    {"id": "c194", "tier": "tier_3", "text": "A freelance contractor delivered excellent work but missed the agreed deadline by three days due to a family emergency they disclosed partway through. Weigh whether the client is justified in withholding the final 20 percent of payment per the contract's late-delivery clause."},
    {"id": "c195", "tier": "tier_3", "text": "Assess whether a company should terminate a long-tenured, well-liked employee whose performance has quietly declined over the past year due to what appears to be untreated burnout, and outline the considerations involved."},
    {"id": "c196", "tier": "tier_3", "text": "Evaluate whether it is reasonable for a landlord to raise rent by 25 percent at lease renewal in a rapidly appreciating rental market, weighing the tenant's hardship against the landlord's financial rationale."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 196 examples.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_3 batch 4/5 (c183-c196)"
```

---

## Task 22: Dataset draft batch 15 (Tier 3, 14 more examples — completes the dataset)

**Files:**
- Modify: `data/complexity_dataset.draft.json`

**Interfaces:**
- Consumes: existing dataset file from Task 21
- Produces: running total 210 examples (all three tiers complete)

- [ ] **Step 1: Append the final 14 examples**

```json
    {"id": "c197", "tier": "tier_3", "text": "A researcher discovers a flaw in their own already-published study's methodology that modestly weakens but does not invalidate the main conclusion. Weigh the researcher's obligations here and what the right course of action is."},
    {"id": "c198", "tier": "tier_3", "text": "Assess whether a social media platform should remove a post that is factually accurate but framed in a way likely to cause public alarm, and explain the trade-offs involved."},
    {"id": "c199", "tier": "tier_3", "text": "A manager must decide whether to lay off a strong performer or a weaker performer with significantly more seniority and tenure protections during a required team reduction. Reason through the fairness considerations."},
    {"id": "c200", "tier": "tier_3", "text": "Evaluate whether a doctor should honor a competent adult patient's explicit refusal of a low-risk, clearly beneficial treatment, and explain the ethical tension involved."},
    {"id": "c201", "tier": "tier_3", "text": "Assess whether it's appropriate for a nonprofit to accept a large donation from a company whose broader business practices conflict with the nonprofit's mission, and lay out the considerations on each side."},
    {"id": "c202", "tier": "tier_3", "text": "A student is caught using an AI tool to help draft an essay in a way that falls into a gray area of the course's academic integrity policy. Weigh how strictly the policy should be enforced given the ambiguity."},
    {"id": "c203", "tier": "tier_3", "text": "Evaluate whether a company should disclose a minor, already-fixed security vulnerability to customers even though disclosure carries reputational risk and no customer data was actually accessed."},
    {"id": "c204", "tier": "tier_3", "text": "Assess the fairness of a hiring policy that gives internal referrals a fast-tracked interview process, potentially disadvantaging equally qualified external candidates, and explain the trade-offs."},
    {"id": "c205", "tier": "tier_3", "text": "A team lead wants to give a high-visibility project to a newer employee to help their growth, over a tenured employee who has quietly wanted the opportunity for years. Weigh the considerations involved."},
    {"id": "c206", "tier": "tier_3", "text": "Evaluate whether an insurance company is justified in denying a claim based on a technicality in the policy wording, even though the denial contradicts the clear intent both parties understood when the policy was purchased."},
    {"id": "c207", "tier": "tier_3", "text": "Assess whether a journalist should publish leaked internal documents that reveal wrongdoing, given that the leak itself may have violated a confidentiality agreement the source signed."},
    {"id": "c208", "tier": "tier_3", "text": "A company discovers that a popular product feature, beloved by a small vocal group of power users, is used by less than 1 percent of the overall customer base and is expensive to maintain. Weigh whether it should be deprecated."},
    {"id": "c209", "tier": "tier_3", "text": "Evaluate whether a manager should intervene in a personal conflict between two team members that is beginning to affect team morale but has not yet crossed into policy-violating behavior."},
    {"id": "c210", "tier": "tier_3", "text": "Assess whether a city should approve a new development project that would ease a housing shortage but requires demolishing a historically significant, though not officially landmarked, building."}
```

- [ ] **Step 2: Run test to verify it still passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (2 passed). Running total: 210 examples, all three tiers complete.

- [ ] **Step 3: Commit**

```bash
git add data/complexity_dataset.draft.json
git commit -m "data: add AI-drafted tier_3 batch 5/5 (c197-c210) - dataset complete (210 examples)"
```

---

## Task 23: Dataset size and balance validation

**Files:**
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: the complete `data/complexity_dataset.draft.json` from Task 22
- Produces: a durable regression check that the dataset never silently shrinks or becomes unbalanced

- [ ] **Step 1: Write the new test**

This test is written last (not first) because it asserts the *final* size — appending it before Task 22 completed would fail correctly-in-progress batches. Append to `tests/test_dataset.py`:
```python
from collections import Counter


def test_dataset_has_at_least_210_examples_balanced_across_tiers():
    raw = _load_raw()
    assert len(raw["examples"]) >= 210
    counts = Counter(example["tier"] for example in raw["examples"])
    for tier in VALID_TIERS:
        assert 60 <= counts[tier] <= 80, f"{tier} has {counts[tier]} examples, expected ~70"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (3 passed) — 210 total, 70/70/70 per tier, well inside the 60-80 tolerance band.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dataset.py
git commit -m "test: add dataset size and tier-balance regression check"
```

---

## Task 24: Classifier — `Dataset`/`LabeledExample` and `load_dataset`

**Files:**
- Create: `costpilot/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `data/complexity_dataset.draft.json` (Task 22)
- Produces: `TIER_LABELS: tuple[str, ...]`, `RANDOM_SEED: int`, `LabeledExample` (dataclass: `id: str`, `text: str`, `tier: str`), `Dataset` (dataclass: `status: str`, `examples: list[LabeledExample]`), `load_dataset(path: Path) -> Dataset`

- [ ] **Step 1: Write the failing test**

Create `tests/test_classifier.py`:
```python
from pathlib import Path

import pytest

from costpilot.classifier import load_dataset

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"


def test_load_dataset_returns_status_and_examples():
    dataset = load_dataset(DATASET_PATH)
    assert dataset.status == "ai_drafted_pending_human_review"
    assert len(dataset.examples) >= 210


def test_load_dataset_rejects_unknown_tier(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(
        '{"version": "1.0", "status": "ai_drafted_pending_human_review", '
        '"examples": [{"id": "x1", "text": "hello", "tier": "tier_9"}]}'
    )
    with pytest.raises(ValueError):
        load_dataset(bad_file)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    bad_file = tmp_path / "dup.json"
    bad_file.write_text(
        '{"version": "1.0", "status": "ai_drafted_pending_human_review", '
        '"examples": ['
        '{"id": "x1", "text": "hello", "tier": "tier_1"}, '
        '{"id": "x1", "text": "world", "tier": "tier_2"}'
        ']}'
    )
    with pytest.raises(ValueError):
        load_dataset(bad_file)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'costpilot.classifier'`.

- [ ] **Step 3: Write minimal implementation**

Create `costpilot/classifier.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TIER_LABELS = ("tier_1", "tier_2", "tier_3")
RANDOM_SEED = 42


@dataclass(frozen=True)
class LabeledExample:
    id: str
    text: str
    tier: str


@dataclass(frozen=True)
class Dataset:
    status: str
    examples: list[LabeledExample]


def load_dataset(path: Path) -> Dataset:
    raw = json.loads(Path(path).read_text())
    status = raw["status"]
    examples: list[LabeledExample] = []
    seen_ids: set[str] = set()
    for entry in raw["examples"]:
        if entry["tier"] not in TIER_LABELS:
            raise ValueError(f"Unknown tier {entry['tier']!r} for example {entry['id']!r}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate example id {entry['id']!r}")
        seen_ids.add(entry["id"])
        examples.append(LabeledExample(id=entry["id"], text=entry["text"], tier=entry["tier"]))
    return Dataset(status=status, examples=examples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/classifier.py tests/test_classifier.py
git commit -m "feat: add Dataset/LabeledExample types and load_dataset"
```

---

## Task 25: Classifier — `train_test_split_dataset`

**Files:**
- Modify: `costpilot/classifier.py`
- Modify: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `Dataset`, `LabeledExample`, `RANDOM_SEED` (Task 24)
- Produces: `train_test_split_dataset(dataset: Dataset, seed: int = RANDOM_SEED, test_size: float = 0.2) -> tuple[list[LabeledExample], list[LabeledExample]]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classifier.py`:
```python
from costpilot.classifier import train_test_split_dataset


def test_train_test_split_covers_every_example_exactly_once():
    dataset = load_dataset(DATASET_PATH)
    train, test = train_test_split_dataset(dataset)
    assert len(train) + len(test) == len(dataset.examples)
    assert {e.id for e in train} | {e.id for e in test} == {e.id for e in dataset.examples}
    assert {e.id for e in train} & {e.id for e in test} == set()


def test_train_test_split_is_deterministic_for_a_fixed_seed():
    dataset = load_dataset(DATASET_PATH)
    train_a, test_a = train_test_split_dataset(dataset, seed=42)
    train_b, test_b = train_test_split_dataset(dataset, seed=42)
    assert [e.id for e in train_a] == [e.id for e in train_b]
    assert [e.id for e in test_a] == [e.id for e in test_b]


def test_train_test_split_is_stratified_across_tiers():
    dataset = load_dataset(DATASET_PATH)
    _, test = train_test_split_dataset(dataset)
    tiers_present = {e.tier for e in test}
    assert tiers_present == {"tier_1", "tier_2", "tier_3"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'train_test_split_dataset'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/classifier.py`:
```python
from sklearn.model_selection import train_test_split


def train_test_split_dataset(
    dataset: Dataset, seed: int = RANDOM_SEED, test_size: float = 0.2
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    labels = [example.tier for example in dataset.examples]
    train, test = train_test_split(
        dataset.examples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/classifier.py tests/test_classifier.py
git commit -m "feat: add stratified, deterministic train_test_split_dataset"
```

---

## Task 26: Classifier — `train_classifier`

**Files:**
- Modify: `costpilot/classifier.py`
- Modify: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `costpilot.features.FEATURE_KEYS`, `costpilot.features.extract_features` (Task 7); `LabeledExample`, `RANDOM_SEED` (Task 24)
- Produces: `train_classifier(train_examples: list[LabeledExample], seed: int = RANDOM_SEED) -> LogisticRegression`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classifier.py`:
```python
from sklearn.linear_model import LogisticRegression

from costpilot.classifier import train_classifier


def test_train_classifier_returns_a_fitted_logistic_regression():
    dataset = load_dataset(DATASET_PATH)
    train, _ = train_test_split_dataset(dataset)
    model = train_classifier(train)
    assert isinstance(model, LogisticRegression)
    assert set(model.classes_) == {"tier_1", "tier_2", "tier_3"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'train_classifier'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/classifier.py`:
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from costpilot.features import FEATURE_KEYS, extract_features


def _feature_matrix(texts: list[str]) -> np.ndarray:
    rows = [[extract_features(text)[key] for key in FEATURE_KEYS] for text in texts]
    return np.array(rows, dtype=float)


def train_classifier(train_examples: list[LabeledExample], seed: int = RANDOM_SEED) -> LogisticRegression:
    X = _feature_matrix([example.text for example in train_examples])
    y = [example.tier for example in train_examples]
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X, y)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/classifier.py tests/test_classifier.py
git commit -m "feat: add train_classifier (LogisticRegression on extracted features)"
```

---

## Task 27: Classifier — `evaluate` and the prototype held-out accuracy test

**Files:**
- Modify: `costpilot/classifier.py`
- Modify: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `_feature_matrix`, `LabeledExample`, `TIER_LABELS` (Tasks 24, 26)
- Produces: `EvaluationResult` (dataclass: `accuracy: float`, `confusion: list[list[int]]`), `evaluate(model, test_examples: list[LabeledExample]) -> EvaluationResult`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classifier.py`:
```python
from costpilot.classifier import evaluate


def test_prototype_held_out_accuracy_on_draft_dataset():
    """Pipeline sanity check only: this measures whether the mechanical
    feature-extraction -> classifier -> evaluation pipeline works, NOT
    real-world routing quality. The dataset is AI-drafted and unreviewed."""
    dataset = load_dataset(DATASET_PATH)
    assert dataset.status == "ai_drafted_pending_human_review"
    train, test = train_test_split_dataset(dataset)
    model = train_classifier(train)
    result = evaluate(model, test)
    assert result.accuracy >= 0.80
    assert len(result.confusion) == 3
    assert all(len(row) == 3 for row in result.confusion)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/classifier.py` (the module already imports `dataclass` from Task 24 — add this new import alongside the existing ones at the top of the file, and the class/function anywhere below):
```python
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    confusion: list[list[int]]


def evaluate(model: LogisticRegression, test_examples: list[LabeledExample]) -> EvaluationResult:
    X = _feature_matrix([example.text for example in test_examples])
    y_true = [example.tier for example in test_examples]
    y_pred = model.predict(X)
    accuracy = float(model.score(X, y_true))
    cm = confusion_matrix(y_true, y_pred, labels=list(TIER_LABELS))
    return EvaluationResult(accuracy=accuracy, confusion=cm.tolist())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS (8 passed). If `result.accuracy < 0.80`, do not weaken the assertion — instead revisit the feature heuristics (Tasks 2-7) or flag the shortfall for human review; a failing pipeline sanity check is real signal, not noise.

- [ ] **Step 5: Commit**

```bash
git add costpilot/classifier.py tests/test_classifier.py
git commit -m "feat: add evaluate (held-out accuracy + confusion matrix)

Prototype pipeline validation on AI-drafted, unreviewed data only -
not a real-world routing-quality claim."
```

---

## Task 28: Classifier — `predict_tier`

**Files:**
- Modify: `costpilot/classifier.py`
- Modify: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `_feature_matrix` (Task 26, now `list[str] -> np.ndarray`)
- Produces: `predict_tier(prompt: str, model: LogisticRegression) -> str` — this is what `costpilot/routing.py` (Task 31) imports.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classifier.py`:
```python
from costpilot.classifier import predict_tier


def test_predict_tier_classifies_a_simple_novel_prompt_as_tier_1():
    dataset = load_dataset(DATASET_PATH)
    model = train_classifier(dataset.examples)
    assert predict_tier("What is the capital of Germany?", model) == "tier_1"


def test_predict_tier_classifies_a_complex_novel_prompt_as_tier_3():
    dataset = load_dataset(DATASET_PATH)
    model = train_classifier(dataset.examples)
    prompt = (
        "Analyze and compare these two hiring strategies, considering at least "
        "three trade-offs, and recommend one with justification."
    )
    assert predict_tier(prompt, model) == "tier_3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'predict_tier'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/classifier.py`:
```python
def predict_tier(prompt: str, model: LogisticRegression) -> str:
    X = _feature_matrix([prompt])
    return str(model.predict(X)[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py -v && mypy costpilot/classifier.py`
Expected: PASS (10 passed), mypy clean (the `sklearn.*` override from Task 1 suppresses missing-stub errors).

If either new test fails to predict the expected tier: this is a real signal about feature/data quality, not a flaky test — do not loosen the assertion. Instead, check that the novel prompt's extracted features (`extract_features(prompt)`) actually resemble the training examples of the expected tier; adjust the test prompt to be more clearly representative of that tier if needed, since the goal is a meaningful sanity check, not a guaranteed-pass tautology.

- [ ] **Step 5: Commit**

```bash
git add costpilot/classifier.py tests/test_classifier.py
git commit -m "feat: add predict_tier for single-prompt classification"
```

---

## Task 29: Routing — `routing.yaml` and `load_routing_config`

**Files:**
- Create: `config/routing.yaml`
- Create: `costpilot/routing.py`
- Test: `tests/test_routing.py`

**Interfaces:**
- Consumes: `costpilot.providers.fake.FAKE_MODELS` (Phase 1, unchanged)
- Produces: `config/routing.yaml`, `load_routing_config(path: Path) -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_routing.py`:
```python
from pathlib import Path

from costpilot.providers.fake import FAKE_MODELS
from costpilot.routing import load_routing_config

ROUTING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"


def test_load_routing_config_has_all_three_tiers():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    assert set(config.keys()) == {"tier_1", "tier_2", "tier_3"}


def test_load_routing_config_maps_to_known_models():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    for model_id in config.values():
        assert model_id in FAKE_MODELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_routing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'costpilot.routing'`.

- [ ] **Step 3: Write minimal implementation**

Create `config/routing.yaml`:
```yaml
tier_1: claude-haiku
tier_2: gpt-4o-mini
tier_3: gpt-4o
```

Create `costpilot/routing.py`:
```python
from __future__ import annotations

from pathlib import Path

import yaml


def load_routing_config(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text())
    return dict(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_routing.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add config/routing.yaml costpilot/routing.py tests/test_routing.py
git commit -m "feat: add routing.yaml tier-to-model map and load_routing_config"
```

---

## Task 30: Routing — `route`

**Files:**
- Modify: `costpilot/routing.py`
- Modify: `tests/test_routing.py`

**Interfaces:**
- Consumes: `FAKE_MODELS` (Phase 1), `load_routing_config` (Task 29)
- Produces: `route(tier: str, config: dict[str, str]) -> ModelConfig`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_routing.py`:
```python
import pytest

from costpilot.routing import route


def test_route_returns_the_correct_model_config_per_tier():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    assert route("tier_1", config) == FAKE_MODELS["claude-haiku"]
    assert route("tier_2", config) == FAKE_MODELS["gpt-4o-mini"]
    assert route("tier_3", config) == FAKE_MODELS["gpt-4o"]


def test_route_raises_for_unknown_tier():
    config = load_routing_config(ROUTING_CONFIG_PATH)
    with pytest.raises(ValueError):
        route("tier_9", config)


def test_route_raises_for_unknown_model_id_in_config():
    with pytest.raises(ValueError):
        route("tier_1", {"tier_1": "not-a-real-model"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_routing.py -v`
Expected: FAIL with `ImportError: cannot import name 'route'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/routing.py`:
```python
from costpilot.domain import ModelConfig
from costpilot.providers.fake import FAKE_MODELS


def route(tier: str, config: dict[str, str]) -> ModelConfig:
    if tier not in config:
        raise ValueError(f"No routing entry for tier {tier!r}")
    model_id = config[tier]
    if model_id not in FAKE_MODELS:
        raise ValueError(f"Routing config maps {tier!r} to unknown model {model_id!r}")
    return FAKE_MODELS[model_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_routing.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add costpilot/routing.py tests/test_routing.py
git commit -m "feat: add route lookup with fail-fast validation"
```

---

## Task 31: Routing — `classify_and_route` composition (end-to-end)

**Files:**
- Modify: `costpilot/routing.py`
- Create: `tests/test_classify_and_route.py`

**Interfaces:**
- Consumes: `costpilot.classifier.predict_tier` (Task 28), `route` (Task 30)
- Produces: `classify_and_route(prompt: str, model, routing_config: dict[str, str]) -> ModelConfig`

- [ ] **Step 1: Write the failing test**

Create `tests/test_classify_and_route.py`:
```python
from pathlib import Path

from costpilot.classifier import load_dataset, train_classifier
from costpilot.providers.fake import FAKE_MODELS
from costpilot.routing import classify_and_route, load_routing_config

DATASET_PATH = Path(__file__).parent.parent / "data" / "complexity_dataset.draft.json"
ROUTING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"


def test_classify_and_route_end_to_end_per_tier():
    dataset = load_dataset(DATASET_PATH)
    model = train_classifier(dataset.examples)
    config = load_routing_config(ROUTING_CONFIG_PATH)

    simple_prompt = "What is the capital of Germany?"
    assert classify_and_route(simple_prompt, model, config) == FAKE_MODELS["claude-haiku"]

    complex_prompt = (
        "Analyze and compare these two hiring strategies, considering at least "
        "three trade-offs, and recommend one with justification."
    )
    assert classify_and_route(complex_prompt, model, config) == FAKE_MODELS["gpt-4o"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classify_and_route.py -v`
Expected: FAIL with `ImportError: cannot import name 'classify_and_route'`.

- [ ] **Step 3: Write minimal implementation**

Add to `costpilot/routing.py`:
```python
from costpilot.classifier import predict_tier


def classify_and_route(prompt: str, model, routing_config: dict[str, str]) -> ModelConfig:
    tier = predict_tier(prompt, model)
    return route(tier, routing_config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classify_and_route.py -v`
Expected: PASS (1 passed). If either assertion fails, treat it the same as Task 28's note — check whether the prompt's features actually resemble that tier's training data before adjusting anything.

- [ ] **Step 5: Commit**

```bash
git add costpilot/routing.py tests/test_classify_and_route.py
git commit -m "feat: add classify_and_route end-to-end offline routing skeleton"
```

---

## Task 32: README, review-gate notice, and full verification

**Files:**
- Create: `README.md`

**Interfaces:**
- Produces: a top-level, human-visible statement of the outstanding data-review gate — the deliverable the project owner asked to have "prominently documented."

- [ ] **Step 1: Create `README.md`**

```markdown
# LLM Cost Autopilot

An offline, deterministic prototype of an LLM cost-routing pipeline (BASWE Project 2), built in phases with no live provider calls, credentials, or network access anywhere in the codebase.

## Status

- **Phase 1 (unified model interface):** complete. See `docs/superpowers/specs/2026-07-26-phase-1-unified-model-interface-design.md`.
- **Phase 2 (complexity classifier):** complete. See `docs/superpowers/specs/2026-07-27-phase-2-complexity-classifier-design.md`.
- **Phases 3-6** (verification loop, audit/dashboard, API, containerization): not started.

## Data provenance — outstanding review gate

`data/complexity_dataset.draft.json` (210 examples used to train and evaluate the Phase 2 classifier) is **AI-drafted, not human-labeled**. Its `status` field is `"ai_drafted_pending_human_review"`. The held-out accuracy and confusion matrix reported by `tests/test_classifier.py` validate that the feature-extraction -> classifier -> evaluation *pipeline* works end to end — they are **not** a claim about real-world routing quality, because the ground truth itself has not been reviewed by a human.

**Outstanding action for the project owner:** review the 210 examples in `data/complexity_dataset.draft.json`, correct any mislabeled or low-quality examples, and only then consider the dataset (and any accuracy figures derived from it) trustworthy ground truth. Until that review happens, all Phase 2 results should be read as prototype/pipeline validation only.

## Running the tests

```bash
uv sync
pytest
ruff check .
mypy costpilot
```
```

- [ ] **Step 2: Run the full verification suite**

Run: `pytest -v && ruff check . && mypy costpilot`
Expected: every Phase 1 and Phase 2 test passes (Phase 1's `test_domain.py`, `test_providers.py`, `test_fake_provider.py` are untouched and still green; Phase 2's `test_features.py`, `test_dataset.py`, `test_classifier.py`, `test_routing.py`, `test_classify_and_route.py` all pass); `ruff check .` reports no issues; `mypy costpilot` reports no issues (the `sklearn.*` override from Task 1 handles the one third-party stub gap).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README documenting Phase 2 status and the dataset review gate"
```

---

## After all tasks: review, merge, push

This plan's tasks end with the feature branch fully green and the review gate documented in `README.md`. The remaining steps — **not** numbered as plan tasks because they aren't TDD slices — are:

1. Run `superpowers:requesting-code-review` against the full `main...feature/phase-2-complexity-classifier` diff. Apply any fixes it surfaces as additional small, genuine commits (each its own commit — this is where "docs/review fixes" commits in the owner's target count come from).
2. Run `superpowers:finishing-a-development-branch` to decide and execute the merge into `main` (following the same precedent as Phase 1: `git log` shows `d247d93`..`5c6088c` were merged from `feature/phase-1-unified-model-interface`).
3. Push `main` to `origin` — this was explicitly authorized by the project owner for this phase, but confirm the push target and command with the owner immediately before running it, consistent with this project's git safety norms.
