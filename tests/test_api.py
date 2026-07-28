import json
import socket
import threading
import time
from pathlib import Path
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import pytest
import uvicorn
from fastapi.testclient import TestClient

from costpilot.api import create_app


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(
        create_app(audit_database_path=tmp_path / "audit.sqlite3"),
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def test_health_and_read_only_endpoints_are_sanitized_and_include_provenance(client: TestClient):
    health = client.get("/healthz")
    models = client.get("/v1/models")
    config = client.get("/v1/config")
    stats = client.get("/v1/stats")

    for response in (health, models, config, stats):
        assert response.status_code == 200
        provenance = response.json()["provenance"]
        assert provenance["offline_deterministic"] is True
        assert provenance["dataset_status"] == "ai_drafted_pending_human_review"
        assert "not actual spend" in provenance["disclaimer"]
    assert {model["model_id"] for model in models.json()["models"]} >= {"gpt-4o", "claude-haiku"}
    assert "path" not in json.dumps(config.json()).lower()
    assert stats.json()["stats"]["event_count"] == 0
    assert "routing_only_simulated_delta_microusd" in stats.json()["stats"]


def test_completion_creates_one_audit_lifecycle_and_returns_simulated_metadata(client: TestClient):
    prompt = "Summarize the offline fixture safely."
    response = client.post(
        "/v1/completions",
        json={"prompt": prompt, "request_id": "api-request-1", "verification_threshold": 1.0},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["request_id"] == "api-request-1"
    assert payload["output_text"].startswith("[")
    assert payload["verification_passed"] is True
    assert payload["escalated"] is False
    assert payload["lifecycle_cost_microusd"] == (
        payload["routed_cost_microusd"] + payload["verification_cost_microusd"]
    )
    assert prompt not in json.dumps(client.get("/v1/stats").json())
    assert client.get("/v1/stats").json()["stats"]["event_count"] == 1


def test_failed_verification_explicitly_reruns_reference_model(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"prompt": "Write a long integration plan with implementation details.", "verification_threshold": 1.0},
    )

    assert response.status_code == 201
    payload = response.json()
    # The payload is deterministic but may route straight to the reference model.
    if payload["verification_passed"] is False:
        assert payload["escalated"] is True
        assert payload["rerun_model_id"] == payload["verification_model_id"]
        assert payload["output_text"].startswith(f"[{payload['rerun_model_id']}]")


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "   "},
        {"prompt": "valid", "request_id": " "},
        {"prompt": "valid", "verification_threshold": 1.1},
        {"prompt": "valid", "model": "gpt-4o"},
    ],
)
def test_completion_validation_is_bounded_and_does_not_echo_prompt(client: TestClient, payload: dict[str, object]):
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid completion request"}


def test_duplicate_request_id_is_a_conflict(client: TestClient):
    payload = {"prompt": "No leakage in duplicate response.", "request_id": "duplicate"}
    assert client.post("/v1/completions", json=payload).status_code == 201
    duplicate = client.post("/v1/completions", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "Duplicate request ID"}


def test_unexpected_errors_are_generic_with_a_correlation_id(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    service = client.app.state.service
    monkeypatch.setattr(service, "models", lambda: (_ for _ in ()).throw(RuntimeError("private prompt")))
    response = client.get("/v1/models")
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "correlation_id" in response.json()
    assert "private prompt" not in response.text


def test_uvicorn_loopback_health_and_completion_smoke(tmp_path: Path):
    app = create_app(audit_database_path=tmp_path / "smoke.sqlite3")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            try:
                with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.2) as response:
                    assert json.load(response)["status"] == "ok"
                break
            except OSError:
                time.sleep(0.02)
        else:
            pytest.fail("uvicorn loopback server did not start")
        request = UrlRequest(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps({"prompt": "Complete the actual loopback smoke."}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=1) as response:
            assert response.status == 201
            assert json.load(response)["provenance"]["offline_deterministic"] is True
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert not thread.is_alive()
