from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from clinical_nlp.config import ModelConfig


class LLMTask(StrEnum):
    ENTITY_RECOVERY = "entity_recovery"
    TYPE_ADJUDICATION = "type_adjudication"
    ASSERTION_ADJUDICATION = "assertion_adjudication"
    ICD_RERANK = "icd_rerank"
    RXNORM_RERANK = "rxnorm_rerank"


class LLMDecisionError(RuntimeError):
    """The provider responded, but no attempt matched the required schema."""


class LLMBackend(Protocol):
    name: str

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
    ) -> BaseModel: ...


class NoopLLMBackend:
    name = "noop"

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
        del reasoning_enabled, call_id, checkpoint_dir, cache_enabled
        raise RuntimeError("LLM backend is unavailable")


def create_llm_backend(
    config: ModelConfig,
    *,
    cache_path: Path | None = None,
) -> LLMBackend:
    if config.backend == "noop":
        return NoopLLMBackend()
    if config.backend == "qwen_transformers":
        from .qwen_transformers import QwenTransformersBackend

        return QwenTransformersBackend(config)
    if config.backend == "openai_compatible":
        from .openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(config, cache_path=cache_path)
    raise ValueError(f"unsupported LLM backend: {config.backend}")
