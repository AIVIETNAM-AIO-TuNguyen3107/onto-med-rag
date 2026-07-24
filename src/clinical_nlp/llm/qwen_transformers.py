from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from clinical_nlp.config import ModelConfig
from clinical_nlp.llm.base import LLMTask
from clinical_nlp.llm.parsing import parse_final_json


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Qwen3.5 multimodal processors require content blocks when tokenizing.
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({**message, "content": content})
    return normalized


def _model_load_kwargs(config: ModelConfig) -> dict[str, Any]:
    """Build from_pretrained kwargs; keep vision/text path on one code path."""
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "local_files_only": config.local_files_only,
    }
    if not config.quantization:
        kwargs["torch_dtype"] = "auto"
        return kwargs
    if config.quantization == "bnb_4bit":
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "bnb_4bit quantization requires bitsandbytes and transformers"
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return kwargs
    raise ValueError(f"unsupported llm.quantization: {config.quantization!r}")


class QwenTransformersBackend:
    name = "qwen_transformers"

    def __init__(self, config: ModelConfig) -> None:
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5 requires a Transformers version exposing "
                "AutoModelForMultimodalLM"
            ) from exc
        self.config = config
        self.last_response_metadata: dict[str, Any] = {}
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            local_files_only=config.local_files_only,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            config.model_id,
            **_model_load_kwargs(config),
        )

    def generate_json(
        self,
        task: LLMTask,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> BaseModel:
        inputs = self.processor.apply_chat_template(
            _normalize_messages(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.config.thinking,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        last_error: json.JSONDecodeError | ValidationError | None = None
        for attempt in range(self.config.max_retries + 1):
            started = time.monotonic()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
            generated = outputs[0][prompt_tokens:]
            raw = self.processor.decode(generated, skip_special_tokens=True)
            elapsed = round(time.monotonic() - started, 2)
            try:
                parsed = parse_final_json(raw, response_schema)
                # Leaf of model_id (path or hub id) satisfies online_preflight match.
                self.last_response_metadata = {
                    "response_model": Path(self.config.model_id).name,
                    "finish_reason": "stop",
                    "reasoning_present": self.config.thinking,
                    "new_tokens": int(generated.shape[-1]),
                    "elapsed_seconds": elapsed,
                }
                return parsed
            except (json.JSONDecodeError, ValidationError) as exc:
                # Raw output can contain hidden reasoning, so do not log or store it.
                last_error = exc
        raise RuntimeError(
            "Qwen did not return valid final JSON after "
            f"{self.config.max_retries + 1} attempts"
        ) from last_error
