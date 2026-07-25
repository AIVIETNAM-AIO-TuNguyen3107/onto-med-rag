from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ValidationError

from clinical_nlp.atomic import atomic_write_json
from clinical_nlp.config import ModelConfig
from clinical_nlp.llm.base import LLMTask
from clinical_nlp.llm.cache import LLMResponseCache
from clinical_nlp.llm.parsing import parse_final_json


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
SAFE_CALL_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class OpenAICompatibleBackend:
    name = "openai_compatible"

    def __init__(
        self,
        config: ModelConfig,
        session: requests.Session | None = None,
        *,
        cache_path: Path | None = None,
    ) -> None:
        if not config.endpoint:
            raise ValueError("OpenAI-compatible backend requires an endpoint")
        self.config = config
        self.session = session
        self.cache = LLMResponseCache(cache_path) if cache_path is not None else None
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
        self._state_lock = threading.Lock()
        self._call_audits: list[dict[str, Any]] = []

    def generate_json(
        self,
        task: LLMTask,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
        max_new_tokens: int | None = None,
        reasoning_enabled: bool | None = None,
        call_id: str | None = None,
        checkpoint_dir: Path | None = None,
        cache_enabled: bool = True,
    ) -> BaseModel:
        started = time.monotonic()
        completion_tokens = max_new_tokens or self.config.max_new_tokens
        requested_reasoning = (
            reasoning_enabled
            if reasoning_enabled is not None
            else self.config.reasoning_enabled
        )
        cache_key = self._cache_key(
            task=task,
            messages=messages,
            response_schema=response_schema,
            completion_tokens=completion_tokens,
            reasoning_enabled=requested_reasoning,
        )
        resolved_call_id = call_id or f"{task.value}-{cache_key[:12]}"
        checkpoint_path = self._checkpoint_path(
            checkpoint_dir,
            resolved_call_id,
            cache_key,
        )

        checkpoint = (
            self._load_checkpoint(
                checkpoint_path,
                cache_key,
                response_schema,
            )
            if cache_enabled
            else None
        )
        if checkpoint is not None:
            parsed, metadata = checkpoint
            self._record_success_metadata(metadata)
            self._record_audit(
                self._reuse_audit(
                    task=task,
                    call_id=resolved_call_id,
                    cache_key=cache_key,
                    source="checkpoint",
                    requested_reasoning=requested_reasoning,
                    metadata=metadata,
                    started=started,
                ),
                checkpoint_dir,
            )
            return parsed

        cached = (
            self.cache.get(cache_key)
            if cache_enabled and self.cache is not None
            else None
        )
        if cached is not None:
            payload, metadata = cached
            parsed = response_schema.model_validate(payload)
            self._record_success_metadata(metadata)
            self._write_checkpoint(
                checkpoint_path,
                cache_key,
                parsed,
                metadata,
            )
            self._record_audit(
                self._reuse_audit(
                    task=task,
                    call_id=resolved_call_id,
                    cache_key=cache_key,
                    source="cache",
                    requested_reasoning=requested_reasoning,
                    metadata=metadata,
                    started=started,
                ),
                checkpoint_dir,
            )
            return parsed

        phases = [
            {
                "reasoning_enabled": requested_reasoning,
                "max_tokens": completion_tokens,
                "fallback": False,
            }
        ]
        if self.config.decision_retries:
            phases.append(
                {
                    "reasoning_enabled": False,
                    "max_tokens": 2048,
                    "fallback": True,
                }
            )

        attempts = 0
        aggregate_usage = _zero_usage()
        last_error: Exception | None = None
        last_metadata: dict[str, Any] = {}
        fatal_error: Exception | None = None
        for phase in phases:
            try:
                parsed, metadata, phase_attempts, phase_usage = self._request_phase(
                    task=task,
                    messages=messages,
                    response_schema=response_schema,
                    max_tokens=int(phase["max_tokens"]),
                    reasoning_enabled=phase["reasoning_enabled"],
                )
            except _DecisionResponseError as exc:
                attempts += exc.attempts
                aggregate_usage = _merge_usage(aggregate_usage, exc.usage)
                last_metadata = exc.metadata
                last_error = exc
                continue
            except Exception as exc:
                attempts += int(getattr(exc, "attempts", 1))
                last_error = exc
                fatal_error = exc
                break
            attempts += phase_attempts
            aggregate_usage = _merge_usage(aggregate_usage, phase_usage)
            metadata["usage"] = aggregate_usage
            self._record_success_metadata(metadata)
            if cache_enabled and self.cache is not None:
                self.cache.put(
                    cache_key,
                    parsed.model_dump(mode="json"),
                    metadata,
                )
            if cache_enabled:
                self._write_checkpoint(
                    checkpoint_path,
                    cache_key,
                    parsed,
                    metadata,
                )
            self._record_audit(
                {
                    "task": task.value,
                    "call_id": resolved_call_id,
                    "cache_key": cache_key,
                    "status": "ok",
                    "source": "api",
                    "attempts": attempts,
                    "reasoning_requested": bool(requested_reasoning),
                    "reasoning_fallback": bool(phase["fallback"]),
                    "elapsed_seconds": time.monotonic() - started,
                    "response_model": metadata.get("response_model"),
                    "finish_reason": metadata.get("finish_reason"),
                    "usage": aggregate_usage,
                },
                checkpoint_dir,
            )
            return parsed

        audit = {
            "task": task.value,
            "call_id": resolved_call_id,
            "cache_key": cache_key,
            "status": "failed",
            "source": "api",
            "attempts": attempts,
            "reasoning_requested": bool(requested_reasoning),
            "reasoning_fallback": len(phases) > 1,
            "elapsed_seconds": time.monotonic() - started,
            "response_model": last_metadata.get("response_model"),
            "finish_reason": last_metadata.get("finish_reason"),
            "usage": aggregate_usage,
            "error_type": type(last_error).__name__ if last_error else "unknown",
        }
        self._record_audit(audit, checkpoint_dir)
        if fatal_error is not None:
            raise fatal_error
        raise RuntimeError(
            f"{task.value} failed after {attempts} HTTP attempt(s)"
        ) from last_error

    def call_audits(self) -> list[dict[str, Any]]:
        with self._state_lock:
            return sorted(
                (dict(row) for row in self._call_audits),
                key=lambda row: (row["call_id"], row["cache_key"]),
            )

    def _request_phase(
        self,
        *,
        task: LLMTask,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
        max_tokens: int,
        reasoning_enabled: bool | None,
    ) -> tuple[BaseModel, dict[str, Any], int, dict[str, Any]]:
        endpoint = self.config.endpoint.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = self._request_payload(
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            reasoning_enabled=reasoning_enabled,
        )
        attempts = 0
        last_error: Exception | None = None
        for transient_attempt in range(self.config.max_retries + 1):
            attempts += 1
            try:
                client = self.session if self.session is not None else requests
                response = client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.config.request_timeout_seconds,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    if transient_attempt >= self.config.max_retries:
                        raise RuntimeError(
                            f"{task.value} exhausted transient HTTP retries "
                            f"with status {response.status_code}"
                        )
                    time.sleep(_retry_delay(response, transient_attempt))
                    continue
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                metadata = _safe_metadata(body, choice, message)
                usage = _usage_from_metadata(metadata)
                if choice.get("finish_reason") == "length":
                    raise _DecisionResponseError(
                        "model exhausted max_tokens before final JSON",
                        attempts=attempts,
                        metadata=metadata,
                        usage=usage,
                    )
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise _DecisionResponseError(
                        "model response has no final content",
                        attempts=attempts,
                        metadata=metadata,
                        usage=usage,
                    )
                try:
                    parsed = parse_final_json(content, response_schema)
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise _DecisionResponseError(
                        "model response failed strict JSON validation",
                        attempts=attempts,
                        metadata=metadata,
                        usage=usage,
                    ) from exc
                return parsed, metadata, attempts, usage
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                raise RuntimeError(
                    f"{task.value} failed with non-retryable HTTP status {status}"
                ) from exc
            except requests.RequestException as exc:
                last_error = exc
                if transient_attempt >= self.config.max_retries:
                    raise RuntimeError(
                        f"{task.value} exhausted transient request retries"
                    ) from exc
                time.sleep(min(2.0**transient_attempt, 10.0))
        raise RuntimeError(f"{task.value} request phase failed") from last_error

    def _request_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
        max_tokens: int,
        reasoning_enabled: bool | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.config.send_reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if reasoning_enabled is not None:
            reasoning: dict[str, Any] = {"exclude": self.config.reasoning_exclude}
            if not reasoning_enabled:
                reasoning["enabled"] = False
            elif self.config.reasoning_max_tokens is not None:
                reasoning["max_tokens"] = min(
                    self.config.reasoning_max_tokens,
                    max(1, max_tokens // 2),
                )
            else:
                reasoning["effort"] = self.config.reasoning_effort
            payload["reasoning"] = reasoning
        if self.config.structured_outputs:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                },
            }
        return payload

    def _cache_key(
        self,
        *,
        task: LLMTask,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
        completion_tokens: int,
        reasoning_enabled: bool | None,
    ) -> str:
        payload = {
            "endpoint": self.config.endpoint.rstrip("/"),
            "model": self.config.model_id,
            "task": task.value,
            "messages": messages,
            "schema": response_schema.model_json_schema(),
            "max_tokens": completion_tokens,
            "reasoning_enabled": reasoning_enabled,
            "reasoning_effort": self.config.reasoning_effort,
            "reasoning_max_tokens": self.config.reasoning_max_tokens,
            "reasoning_exclude": self.config.reasoning_exclude,
            "structured_outputs": self.config.structured_outputs,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _record_success_metadata(self, metadata: dict[str, Any]) -> None:
        with self._state_lock:
            self.last_response_metadata = dict(metadata)

    def _record_audit(
        self,
        audit: dict[str, Any],
        checkpoint_dir: Path | None,
    ) -> None:
        with self._state_lock:
            self._call_audits.append(dict(audit))
        if checkpoint_dir is None:
            return
        audit_dir = checkpoint_dir / "audits"
        safe_id = SAFE_CALL_ID_RE.sub("_", str(audit["call_id"]))
        atomic_write_json(
            audit_dir / f"{safe_id}-{audit['cache_key'][:12]}.json",
            audit,
        )

    @staticmethod
    def _reuse_audit(
        *,
        task: LLMTask,
        call_id: str,
        cache_key: str,
        source: str,
        requested_reasoning: bool | None,
        metadata: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        return {
            "task": task.value,
            "call_id": call_id,
            "cache_key": cache_key,
            "status": "ok",
            "source": source,
            "attempts": 0,
            "reasoning_requested": bool(requested_reasoning),
            "reasoning_fallback": False,
            "elapsed_seconds": time.monotonic() - started,
            "response_model": metadata.get("response_model"),
            "finish_reason": metadata.get("finish_reason"),
            "usage": _zero_usage(),
        }

    @staticmethod
    def _checkpoint_path(
        checkpoint_dir: Path | None,
        call_id: str,
        cache_key: str,
    ) -> Path | None:
        if checkpoint_dir is None:
            return None
        safe_id = SAFE_CALL_ID_RE.sub("_", call_id)
        return checkpoint_dir / "responses" / f"{safe_id}-{cache_key[:12]}.json"

    @staticmethod
    def _load_checkpoint(
        path: Path | None,
        cache_key: str,
        response_schema: type[BaseModel],
    ) -> tuple[BaseModel, dict[str, Any]] | None:
        if path is None or not path.exists():
            return None
        payload = json.loads(path.read_text("utf-8"))
        if payload.get("cache_key") != cache_key:
            return None
        parsed = response_schema.model_validate(payload["response"])
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid checkpoint metadata: {path}")
        return parsed, metadata

    @staticmethod
    def _write_checkpoint(
        path: Path | None,
        cache_key: str,
        response: BaseModel,
        metadata: dict[str, Any],
    ) -> None:
        if path is None:
            return
        atomic_write_json(
            path,
            {
                "cache_key": cache_key,
                "response": response.model_dump(mode="json"),
                "metadata": metadata,
            },
        )


class _DecisionResponseError(Exception):
    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        metadata: dict[str, Any],
        usage: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.metadata = metadata
        self.usage = usage


def _safe_metadata(
    body: dict[str, Any],
    choice: dict[str, Any],
    message: dict[str, Any],
) -> dict[str, Any]:
    return {
        "response_model": body.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_present": bool(
            message.get("reasoning")
            or message.get("reasoning_content")
            or message.get("reasoning_details")
        ),
        "usage": body.get("usage", {}),
    }


def _zero_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


def _usage_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    usage = metadata.get("usage", {})
    completion_details = usage.get("completion_tokens_details", {})
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cost": float(usage.get("cost") or 0.0),
    }


def _merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in _zero_usage()}


def _retry_delay(response: requests.Response, attempt: int) -> float:
    value = response.headers.get("Retry-After")
    if value:
        try:
            return min(float(value), 30.0)
        except ValueError:
            pass
    return min(2.0**attempt, 10.0)
