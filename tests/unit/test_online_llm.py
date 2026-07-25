from __future__ import annotations

import json

import pytest
import requests
from pydantic import BaseModel

from clinical_nlp.config import ModelConfig
from clinical_nlp.llm.base import LLMTask
from clinical_nlp.llm.openai_compatible import OpenAICompatibleBackend


class SmokeResponse(BaseModel):
    status: str


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _body(content: str, reasoning: str | None = None) -> dict:
    message = {"content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "model": "Qwen3.5-9B",
        "choices": [
            {
                "finish_reason": "stop",
                "message": message,
            }
        ],
        "usage": {"total_tokens": 10},
    }


def _config() -> ModelConfig:
    return ModelConfig(
        backend="openai_compatible",
        model_id="Qwen/Qwen3.5-9B:fastest",
        endpoint="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        max_retries=2,
    )


def test_huggingface_payload_omits_unsupported_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    session = FakeSession(
        [FakeResponse(200, _body('{"status":"ok"}', reasoning="hidden"))]
    )
    backend = OpenAICompatibleBackend(_config(), session=session)

    response = backend.generate_json(
        LLMTask.ENTITY_RECOVERY,
        [{"role": "user", "content": "test"}],
        SmokeResponse,
        max_new_tokens=128,
    )

    assert response.status == "ok"
    assert "extra_body" not in session.calls[0]["json"]
    assert session.calls[0]["json"]["reasoning_effort"] == "high"
    assert session.calls[0]["json"]["max_tokens"] == 128
    assert backend.last_response_metadata["reasoning_present"] is True
    assert "hidden" not in json.dumps(backend.last_response_metadata)


def test_huggingface_retries_rate_limit_and_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "clinical_nlp.llm.openai_compatible.time.sleep",
        lambda _: None,
    )
    session = FakeSession(
        [
            FakeResponse(429, {}, {"Retry-After": "0"}),
            FakeResponse(200, _body("not-json")),
            FakeResponse(200, _body('{"status":"ok"}')),
        ]
    )
    backend = OpenAICompatibleBackend(_config(), session=session)

    response = backend.generate_json(
        LLMTask.ENTITY_RECOVERY,
        [{"role": "user", "content": "test"}],
        SmokeResponse,
    )

    assert response.status == "ok"
    assert len(session.calls) == 3


def test_huggingface_requires_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        OpenAICompatibleBackend(_config(), session=FakeSession([]))


@pytest.mark.parametrize("status_code", [400, 401, 402, 403])
def test_huggingface_does_not_retry_non_retryable_4xx(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    session = FakeSession([FakeResponse(status_code, {})])
    backend = OpenAICompatibleBackend(_config(), session=session)

    with pytest.raises(RuntimeError, match=f"status {status_code}"):
        backend.generate_json(
            LLMTask.ENTITY_RECOVERY,
            [{"role": "user", "content": "test"}],
            SmokeResponse,
        )

    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "first_response",
    [
        FakeResponse(500, {}),
        requests.Timeout("timed out"),
    ],
)
def test_huggingface_retries_server_error_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    first_response: FakeResponse | Exception,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "clinical_nlp.llm.openai_compatible.time.sleep",
        lambda _: None,
    )
    session = FakeSession(
        [
            first_response,
            FakeResponse(200, _body('{"status":"ok"}')),
        ]
    )
    backend = OpenAICompatibleBackend(_config(), session=session)

    response = backend.generate_json(
        LLMTask.ENTITY_RECOVERY,
        [{"role": "user", "content": "test"}],
        SmokeResponse,
    )

    assert response.status == "ok"
    assert len(session.calls) == 2


def test_reasoning_effort_payload_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    config = _config().model_copy(update={"send_reasoning_effort": False})
    session = FakeSession([FakeResponse(200, _body('{"status":"ok"}'))])
    backend = OpenAICompatibleBackend(config, session=session)

    backend.generate_json(
        LLMTask.ENTITY_RECOVERY,
        [{"role": "user", "content": "test"}],
        SmokeResponse,
    )

    assert "reasoning_effort" not in session.calls[0]["json"]
