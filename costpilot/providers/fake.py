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
