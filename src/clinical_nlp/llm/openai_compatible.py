from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from pydantic import BaseModel, ValidationError

from clinical_nlp.config import ModelConfig
from clinical_nlp.llm.base import LLMTask
from clinical_nlp.llm.parsing import parse_final_json


class OpenAICompatibleBackend:
    name = "openai_compatible"

    def __init__(
        self,
        config: ModelConfig,
        session: requests.Session | None = None,
    ) -> None:
        if not config.endpoint:
            raise ValueError("OpenAI-compatible backend requires an endpoint")
        self.config = config
        self.session = session or requests.Session()
        self.api_key_env = config.api_key_env or (
            "HF_TOKEN"
            if "router.huggingface.co" in config.endpoint
            else "OPENAI_API_KEY"
        )
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"missing API credential in environment variable {self.api_key_env}"
            )
        self.last_response_metadata: dict[str, Any] = {}

    def generate_json(
        self,
        task: LLMTask,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> BaseModel:
        endpoint = self.config.endpoint.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "reasoning_effort": self.config.reasoning_effort,
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.config.request_timeout_seconds,
                )
                if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    raise _RetryableHTTPError(response)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                row = choice["message"]
                content = row.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("model response has no final content")
                parsed = parse_final_json(content, response_schema)
                self.last_response_metadata = {
                    "response_model": body.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "reasoning_present": bool(
                        row.get("reasoning") or row.get("reasoning_content")
                    ),
                    "usage": body.get("usage", {}),
                }
                # Deliberately never retain reasoning/reasoning_content or raw output.
                return parsed
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                raise RuntimeError(
                    f"{task.value} failed with non-retryable HTTP status {status}"
                ) from exc
            except (
                requests.RequestException,
                _RetryableHTTPError,
                json.JSONDecodeError,
                ValidationError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                delay = _retry_delay(exc, attempt)
                time.sleep(delay)
        raise RuntimeError(
            f"{task.value} failed after {self.config.max_retries + 1} attempts"
        ) from last_error


class _RetryableHTTPError(Exception):
    def __init__(self, response: requests.Response) -> None:
        super().__init__(f"retryable HTTP status {response.status_code}")
        self.response = response


def _retry_delay(exc: Exception, attempt: int) -> float:
    if isinstance(exc, _RetryableHTTPError):
        value = exc.response.headers.get("Retry-After")
        if value:
            try:
                return min(float(value), 30.0)
            except ValueError:
                pass
    return min(2.0**attempt, 10.0)
