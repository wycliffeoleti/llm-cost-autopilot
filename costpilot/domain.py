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
