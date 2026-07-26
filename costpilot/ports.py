from typing import Protocol, runtime_checkable

from costpilot.domain import ModelConfig, Request, Response


@runtime_checkable
class Provider(Protocol):
    def send(self, request: Request, model: ModelConfig) -> Response: ...
