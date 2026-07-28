"""Loopback-oriented HTTP interface for the offline deterministic prototype."""

from __future__ import annotations

import math
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from costpilot.audit import cost_usd_to_microusd
from costpilot.service import Completion, OfflineService

MAX_PROMPT_CHARS = 12_000
MAX_REQUEST_ID_CHARS = 128
ROOT = Path(__file__).resolve().parent.parent


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: Annotated[str, Field(min_length=1, max_length=MAX_PROMPT_CHARS)]
    request_id: Annotated[str | None, Field(default=None, max_length=MAX_REQUEST_ID_CHARS)] = None
    verification_threshold: float | None = None

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be blank")
        return value

    @field_validator("request_id")
    @classmethod
    def request_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Request ID must not be blank")
        return value

    @field_validator("verification_threshold")
    @classmethod
    def threshold_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError("Verification threshold must be between 0.0 and 1.0")
        return value


def _completion_payload(completion: Completion, provenance: dict[str, object]) -> dict[str, object]:
    routed_cost = cost_usd_to_microusd(completion.routed_response.cost_usd)
    verification_cost = cost_usd_to_microusd(completion.verification_response.cost_usd)
    rerun_cost = (
        0
        if completion.rerun_response is None
        else cost_usd_to_microusd(completion.rerun_response.cost_usd)
    )
    return {
        "request_id": completion.request_id,
        "output_text": completion.output_text,
        "classifier_tier": completion.classifier_tier,
        "routed_model_id": completion.routed_response.model_id,
        "routed_input_tokens": completion.routed_response.input_tokens,
        "routed_output_tokens": completion.routed_response.output_tokens,
        "routed_cost_microusd": routed_cost,
        "verification_model_id": completion.verification_response.model_id,
        "verification_score": completion.verification_score,
        "verification_threshold": completion.verification_threshold,
        "verification_passed": completion.verification_passed,
        "verification_cost_microusd": verification_cost,
        "escalated": completion.escalated,
        "rerun_model_id": None if completion.rerun_response is None else completion.rerun_response.model_id,
        "rerun_cost_microusd": rerun_cost,
        "lifecycle_cost_microusd": routed_cost + verification_cost + rerun_cost,
        "provenance": provenance,
    }


def create_app(
    *,
    audit_database_path: Path | None = None,
    dataset_path: Path = ROOT / "data" / "complexity_dataset.draft.json",
    routing_config_path: Path = ROOT / "config" / "routing.yaml",
    verification_config_path: Path = ROOT / "config" / "verification.yaml",
) -> FastAPI:
    database_path = audit_database_path or ROOT / "audit.sqlite3"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = OfflineService(
            dataset_path=dataset_path,
            routing_config_path=routing_config_path,
            verification_config_path=verification_config_path,
            audit_database_path=database_path,
        )
        yield

    app = FastAPI(title="LLM Cost Autopilot", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Invalid completion request"})

    @app.exception_handler(Exception)
    async def internal_error(_: Request, __: Exception) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "correlation_id": correlation_id},
        )

    def service(request: Request) -> OfflineService:
        return request.app.state.service  # type: ignore[no-any-return]

    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, object]:
        return {"status": "ok", "provenance": service(request).provenance()}

    @app.post("/v1/completions", status_code=201)
    def completions(payload: CompletionRequest, request: Request) -> dict[str, object]:
        request_id = payload.request_id or str(uuid.uuid4())
        try:
            completion = service(request).complete(
                payload.prompt, request_id, payload.verification_threshold
            )
        except ValueError as error:
            if "request ID" in str(error):
                raise HTTPException(status_code=409, detail="Duplicate request ID") from error
            raise
        return _completion_payload(completion, service(request).provenance())

    @app.get("/v1/models")
    def models(request: Request) -> dict[str, object]:
        return {"models": service(request).models(), "provenance": service(request).provenance()}

    @app.get("/v1/stats")
    def stats(request: Request) -> dict[str, object]:
        return {"stats": service(request).stats(), "provenance": service(request).provenance()}

    @app.get("/v1/config")
    def config(request: Request) -> dict[str, object]:
        return {"config": service(request).config(), "provenance": service(request).provenance()}

    return app


app = create_app()
