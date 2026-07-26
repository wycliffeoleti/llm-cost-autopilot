# LLM Cost Autopilot — Phase 1 Unified Model Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the provider-neutral model registry, unified `Provider` interface, and deterministic offline fake providers that BASWE Project 2 Phase 1 requires, with tests proving every registered fake provider handles the same fixed 10-prompt set.

**Architecture:** Hexagonal ports-and-adapters. `costpilot/domain.py` defines framework-independent data contracts (`ModelConfig`, `Request`, `Response`). `costpilot/ports.py` defines the `Provider` protocol every adapter implements. `costpilot/providers/fake.py` is the one adapter this phase ships: a deterministic, offline simulator keyed off a 5-model registry with dated real pricing. No routing, classification, verification, persistent audit store, API, or container layer exists yet — those are Phases 2–6 and are out of scope.

**Tech Stack:** Python 3.11+, `uv` for environment/dependency management, `pytest` for tests, `ruff` + `mypy --strict` for lint/type checks. Zero runtime dependencies (no FastAPI, no HTTP client, no SDKs) — Phase 1 needs none of them.

## Global Constraints

- Python `>=3.11` (per approved spec §3, guide tech stack).
- Offline-first: no network calls, no real provider SDKs, no API keys/credentials anywhere in Phase 1 code (per spec §2, "Standing constraint").
- No dependencies beyond `pytest`, `ruff`, `mypy` (dev-only) — zero runtime dependencies (per spec §5, Phase 1 layout has no FastAPI/HTTP/SDK imports).
- No files or scaffolding for Phases 2–6: no classifier, routing YAML, verification loop, SQLite/JSON audit store, dashboard, FastAPI app, or Docker config (per spec §6).
- No GitHub repository, remote, or `git push` (per spec §7). All commits stay local on `main`.
- Package name `costpilot`; repository root `/home/wolnxpc/projects/Personal/portfolio/llm-cost-autopilot` (per spec §4).
- Model pricing must be a **dated snapshot** of real, publicly published list prices, not live-queried at runtime (per spec §5, "Fake provider registry"). Snapshot captured 2026-07-27 via web search against OpenAI's and Anthropic's public pricing pages.
- Fake provider responses must be a **pure deterministic function** of `(prompt, model)` — same inputs always reproduce an identical `Response` (per spec §5).

---

## File Structure

```
llm-cost-autopilot/
  pyproject.toml               # project metadata, uv dependency groups, ruff/mypy/pytest config
  .gitignore                    # standard Python + uv ignores
  costpilot/
    __init__.py
    domain.py                    # ModelConfig, Request, Response dataclasses (Task 1)
    ports.py                      # Provider protocol (Task 2)
    providers/
      __init__.py
      fake.py                       # FAKE_MODELS registry + FakeProvider (Task 2)
  evals/
    prompts/
      phase1_baseline.json           # fixed 10-prompt set (Task 3)
  tests/
    test_domain.py                    # dataclass contract tests (Task 1)
    test_fake_provider.py              # FakeProvider unit/determinism tests (Task 2)
    test_providers.py                   # guide's step 3: 10 prompts x 5 providers (Task 3)
```

Three test files, one per implementation concern, rather than one large file — `test_domain.py` and `test_fake_provider.py` are component-level unit tests; `test_providers.py` is the guide-mandated full-registry baseline sweep the spec names explicitly. This keeps each file focused and independently reviewable.

---

### Task 1: Project scaffolding and domain contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `costpilot/__init__.py`
- Create: `costpilot/providers/__init__.py`
- Create: `costpilot/domain.py`
- Test: `tests/test_domain.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `costpilot.domain.ModelConfig(provider: str, model_id: str, cost_per_input_token: float, cost_per_output_token: float, avg_latency_ms: float, quality_tier: Literal["high", "medium", "low"])` — frozen dataclass.
  - `costpilot.domain.Request(prompt: str, request_id: str)` — frozen dataclass.
  - `costpilot.domain.Response(output_text: str, input_tokens: int, output_tokens: int, latency_ms: float, cost_usd: float, model_id: str)` — frozen dataclass.

- [ ] **Step 1: Create project scaffolding**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-cost-autopilot"
version = "0.1.0"
description = "BASWE Project 2 - LLM Cost Autopilot (Phase 1: unified model interface)"
requires-python = ">=3.11"
dependencies = []

[dependency-groups]
dev = ["pytest>=8,<9", "ruff>=0.6,<1", "mypy>=1.11,<2"]

[tool.setuptools]
packages = ["costpilot", "costpilot.providers"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
plugins = []
exclude = ["tests"]
```

Create `.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
*.egg-info/
```

Create `costpilot/__init__.py` (empty file).

Create `costpilot/providers/__init__.py` (empty file).

- [ ] **Step 2: Sync the environment**

Run: `cd /home/wolnxpc/projects/Personal/portfolio/llm-cost-autopilot && uv sync --group dev`
Expected: creates `.venv/` and `uv.lock`; exits 0.

- [ ] **Step 3: Write the failing test**

Create `tests/test_domain.py`:

```python
from costpilot.domain import ModelConfig, Request, Response


def test_model_config_holds_all_required_fields():
    config = ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        cost_per_input_token=0.0000025,
        cost_per_output_token=0.00001,
        avg_latency_ms=1100.0,
        quality_tier="high",
    )
    assert config.provider == "openai"
    assert config.model_id == "gpt-4o"
    assert config.cost_per_input_token == 0.0000025
    assert config.cost_per_output_token == 0.00001
    assert config.avg_latency_ms == 1100.0
    assert config.quality_tier == "high"


def test_request_holds_prompt_and_id():
    request = Request(prompt="Hello", request_id="req-1")
    assert request.prompt == "Hello"
    assert request.request_id == "req-1"


def test_response_holds_all_required_fields():
    response = Response(
        output_text="hi",
        input_tokens=3,
        output_tokens=5,
        latency_ms=1100.0,
        cost_usd=0.00012,
        model_id="gpt-4o",
    )
    assert response.output_text == "hi"
    assert response.input_tokens == 3
    assert response.output_tokens == 5
    assert response.latency_ms == 1100.0
    assert response.cost_usd == 0.00012
    assert response.model_id == "gpt-4o"
```

- [ ] **Step 4: Run the test and verify it fails**

Run: `uv run pytest tests/test_domain.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'costpilot.domain'`

- [ ] **Step 5: Implement the domain contracts**

Create `costpilot/domain.py`:

```python
from dataclasses import dataclass
from typing import Literal

QualityTier = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_ms: float
    quality_tier: QualityTier


@dataclass(frozen=True)
class Request:
    prompt: str
    request_id: str


@dataclass(frozen=True)
class Response:
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_id: str
```

- [ ] **Step 6: Run the test and verify it passes**

Run: `uv run pytest tests/test_domain.py -v`
Expected: 3 passed

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check . && uv run mypy costpilot`
Expected: both exit 0. Fix any reported issues before continuing.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore costpilot/__init__.py costpilot/providers/__init__.py costpilot/domain.py tests/test_domain.py uv.lock
git commit -m "feat: add project scaffolding and domain contracts"
```

---

### Task 2: Unified provider interface and fake provider registry

**Files:**
- Create: `costpilot/ports.py`
- Create: `costpilot/providers/fake.py`
- Test: `tests/test_fake_provider.py`

**Interfaces:**
- Consumes: `costpilot.domain.ModelConfig`, `costpilot.domain.Request`, `costpilot.domain.Response` (Task 1).
- Produces:
  - `costpilot.ports.Provider` — `@runtime_checkable` `Protocol` with `send(self, request: Request, model: ModelConfig) -> Response`.
  - `costpilot.providers.fake.FAKE_MODELS: dict[str, ModelConfig]` with exactly the keys `"gpt-4o"`, `"gpt-4o-mini"`, `"claude-sonnet"`, `"claude-haiku"`, `"llama-local"`.
  - `costpilot.providers.fake.FakeProvider` — class implementing `Provider`, with `send(self, request: Request, model: ModelConfig) -> Response`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fake_provider.py`:

```python
from costpilot.domain import Request
from costpilot.ports import Provider
from costpilot.providers.fake import FAKE_MODELS, FakeProvider


def test_fake_models_registry_has_five_expected_models():
    assert set(FAKE_MODELS.keys()) == {
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet",
        "claude-haiku",
        "llama-local",
    }


def test_fake_provider_satisfies_provider_protocol():
    assert isinstance(FakeProvider(), Provider)


def test_send_returns_well_formed_response():
    provider = FakeProvider()
    request = Request(prompt="Summarize this in one sentence.", request_id="req-1")
    model = FAKE_MODELS["gpt-4o-mini"]

    response = provider.send(request, model)

    assert response.model_id == "gpt-4o-mini"
    assert response.output_text
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms == model.avg_latency_ms
    assert response.cost_usd >= 0


def test_send_is_deterministic_for_same_request_and_model():
    provider = FakeProvider()
    request = Request(prompt="What is the capital of France?", request_id="req-2")
    model = FAKE_MODELS["claude-sonnet"]

    first = provider.send(request, model)
    second = provider.send(request, model)

    assert first == second


def test_send_produces_different_costs_across_models_for_same_prompt():
    provider = FakeProvider()
    request = Request(
        prompt="Explain quantum entanglement in two sentences.", request_id="req-3"
    )

    costs = {
        model_id: provider.send(request, model).cost_usd
        for model_id, model in FAKE_MODELS.items()
    }

    assert len(set(costs.values())) > 1
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `uv run pytest tests/test_fake_provider.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'costpilot.ports'`

- [ ] **Step 3: Implement the Provider protocol**

Create `costpilot/ports.py`:

```python
from typing import Protocol, runtime_checkable

from costpilot.domain import ModelConfig, Request, Response


@runtime_checkable
class Provider(Protocol):
    def send(self, request: Request, model: ModelConfig) -> Response: ...
```

- [ ] **Step 4: Implement the fake provider registry**

Create `costpilot/providers/fake.py`:

```python
import hashlib

from costpilot.domain import ModelConfig, Request, Response

# Pricing snapshot captured 2026-07-27 from OpenAI's and Anthropic's public
# API pricing pages (per-million-token list prices, converted to per-token
# here). Not live-queried at runtime -- these are fixed constants.
#   gpt-4o:        $2.50 / $10.00 per 1M input/output tokens
#   gpt-4o-mini:   $0.15 / $0.60  per 1M input/output tokens
#   claude-sonnet: $3.00 / $15.00 per 1M input/output tokens (priced as Sonnet 4.6)
#   claude-haiku:  $1.00 / $5.00  per 1M input/output tokens (priced as Haiku 4.5)
#   llama-local:   $0.00 / $0.00 -- local inference via Ollama, no per-token API cost
FAKE_MODELS: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        cost_per_input_token=2.50 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        avg_latency_ms=1100.0,
        quality_tier="high",
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        avg_latency_ms=450.0,
        quality_tier="medium",
    ),
    "claude-sonnet": ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet",
        cost_per_input_token=3.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        avg_latency_ms=1200.0,
        quality_tier="high",
    ),
    "claude-haiku": ModelConfig(
        provider="anthropic",
        model_id="claude-haiku",
        cost_per_input_token=1.00 / 1_000_000,
        cost_per_output_token=5.00 / 1_000_000,
        avg_latency_ms=400.0,
        quality_tier="medium",
    ),
    "llama-local": ModelConfig(
        provider="ollama",
        model_id="llama-local",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=1800.0,
        quality_tier="low",
    ),
}

# Per-model deterministic simulation profile. tokens_per_word approximates
# each provider's real tokenizer (illustrative, not measured from a real
# tokenizer library -- Phase 1 has zero runtime dependencies). verbosity
# scales simulated output length relative to input length.
_SIM_PROFILE: dict[str, dict[str, float]] = {
    "gpt-4o": {"tokens_per_word": 1.30, "verbosity": 1.50},
    "gpt-4o-mini": {"tokens_per_word": 1.30, "verbosity": 0.80},
    "claude-sonnet": {"tokens_per_word": 1.25, "verbosity": 1.40},
    "claude-haiku": {"tokens_per_word": 1.25, "verbosity": 0.70},
    "llama-local": {"tokens_per_word": 1.40, "verbosity": 1.00},
}


class FakeProvider:
    """Deterministic, offline Provider adapter.

    The same (prompt, model) pair always reproduces an identical Response.
    No network access; no real tokenizer or model call of any kind.
    """

    def send(self, request: Request, model: ModelConfig) -> Response:
        profile = _SIM_PROFILE[model.model_id]
        word_count = max(1, len(request.prompt.split()))
        input_tokens = max(1, round(word_count * profile["tokens_per_word"]))
        output_tokens = max(1, round(input_tokens * profile["verbosity"]))
        cost = (
            input_tokens * model.cost_per_input_token
            + output_tokens * model.cost_per_output_token
        )
        digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:8]
        output_text = (
            f"[{model.model_id}] simulated response "
            f"(digest={digest}, input_tokens={input_tokens})"
        )
        return Response(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=model.avg_latency_ms,
            cost_usd=cost,
            model_id=model.model_id,
        )
```

Note: `hashlib.sha256` is used deliberately instead of Python's builtin `hash()` — `hash()` on strings is randomized per-process (`PYTHONHASHSEED`) and would break the "same prompt always reproduces identically" requirement across separate test runs or processes.

- [ ] **Step 5: Run the test and verify it passes**

Run: `uv run pytest tests/test_fake_provider.py -v`
Expected: 5 passed

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check . && uv run mypy costpilot`
Expected: both exit 0. Fix any reported issues before continuing.

- [ ] **Step 7: Commit**

```bash
git add costpilot/ports.py costpilot/providers/fake.py tests/test_fake_provider.py
git commit -m "feat: add unified provider interface and fake provider registry"
```

---

### Task 3: Fixed baseline prompt set and full-registry provider comparison test

**Files:**
- Create: `evals/prompts/phase1_baseline.json`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `costpilot.domain.Request` (Task 1); `costpilot.providers.fake.FAKE_MODELS`, `costpilot.providers.fake.FakeProvider` (Task 2).
- Produces: `evals/prompts/phase1_baseline.json` — `{"version": "1.0", "prompts": [{"id": str, "text": str}, ...]}` with exactly 10 entries. No further Phase 1 tasks consume this; it is the guide's Phase 1 step 3 deliverable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_providers.py`:

```python
import json
from pathlib import Path

import pytest

from costpilot.domain import Request
from costpilot.providers.fake import FAKE_MODELS, FakeProvider

PROMPTS_PATH = Path(__file__).parent.parent / "evals" / "prompts" / "phase1_baseline.json"


def _load_prompts() -> list[dict[str, str]]:
    data = json.loads(PROMPTS_PATH.read_text())
    return data["prompts"]


def test_baseline_prompt_set_has_exactly_ten_unique_prompts():
    prompts = _load_prompts()
    assert len(prompts) == 10
    assert len({p["id"] for p in prompts}) == 10
    assert all(p["text"].strip() for p in prompts)


@pytest.mark.parametrize("model_id", sorted(FAKE_MODELS.keys()))
def test_every_provider_handles_every_baseline_prompt(model_id):
    provider = FakeProvider()
    model = FAKE_MODELS[model_id]
    prompts = _load_prompts()

    for entry in prompts:
        request = Request(prompt=entry["text"], request_id=entry["id"])
        response = provider.send(request, model)

        assert response.model_id == model_id
        assert response.output_text
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms == model.avg_latency_ms
        assert response.cost_usd >= 0
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL — `FileNotFoundError` (or similar) reading `evals/prompts/phase1_baseline.json`, since the fixture does not exist yet.

- [ ] **Step 3: Create the fixed baseline prompt set**

Create `evals/prompts/phase1_baseline.json`:

```json
{
  "version": "1.0",
  "prompts": [
    {
      "id": "p01",
      "text": "Summarize the following paragraph in one sentence: The quarterly report showed a 12% increase in revenue driven primarily by the new subscription tier, while operating costs remained flat year over year."
    },
    {
      "id": "p02",
      "text": "Extract the shipping address from this text: John Smith, 42 Baker Street, London, NW1 6XE, United Kingdom."
    },
    {
      "id": "p03",
      "text": "What is the capital of France?"
    },
    {
      "id": "p04",
      "text": "Classify the sentiment of this review as positive, negative, or neutral: The product arrived on time but the packaging was damaged."
    },
    {
      "id": "p05",
      "text": "Reformat this list into alphabetical order: banana, apple, cherry, date."
    },
    {
      "id": "p06",
      "text": "Compare the tradeoffs between REST and GraphQL APIs for a mobile application with limited bandwidth."
    },
    {
      "id": "p07",
      "text": "Analyze the following error log and identify the most likely root cause: ConnectionResetError, Errno 104, Connection reset by peer, occurring every 30 seconds under load."
    },
    {
      "id": "p08",
      "text": "Write a short, professional email declining a meeting invitation due to a scheduling conflict."
    },
    {
      "id": "p09",
      "text": "Given the constraints that the output must be valid JSON, include exactly three fields (name, age, city), and the age must be an integer, generate a sample record for a person named Priya in Berlin."
    },
    {
      "id": "p10",
      "text": "Critique the following product pitch for logical inconsistencies and suggest one concrete improvement: Our app is the fastest on the market, though we haven't benchmarked it against competitors, and users love it based on the two reviews we've received."
    }
  ]
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `uv run pytest tests/test_providers.py -v`
Expected: 6 passed (1 fixture-shape test + 5 parametrized provider tests)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: 14 passed (3 from `test_domain.py`, 5 from `test_fake_provider.py`, 6 from `test_providers.py`), 0 failed.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check . && uv run mypy costpilot`
Expected: both exit 0. Fix any reported issues before continuing.

- [ ] **Step 7: Commit**

```bash
git add evals/prompts/phase1_baseline.json tests/test_providers.py
git commit -m "feat: add fixed baseline prompt set and full-registry provider test"
```

---

## Definition of done for Phase 1

- `uv run pytest -v` passes with 14 tests, 0 failures.
- `uv run ruff check .` and `uv run mypy costpilot` both exit 0.
- Three new local commits exist on `main`, on top of the existing spec commit (`fc4cdeb`), each independently reviewable (scaffolding+contracts, provider interface+fake registry, baseline prompt set+full sweep test).
- No file under `costpilot/`, `evals/`, or `tests/` references routing, classification, verification, an audit/logging store, an API framework, or Docker.
- No remote has been added; no push has occurred.
