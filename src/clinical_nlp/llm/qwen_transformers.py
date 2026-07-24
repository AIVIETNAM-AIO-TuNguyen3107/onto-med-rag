from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from clinical_nlp.config import ModelConfig
from clinical_nlp.llm.base import LLMTask
from clinical_nlp.llm.parsing import parse_final_json


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
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            local_files_only=config.local_files_only,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            config.model_id,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=config.local_files_only,
        )

    def generate_json(
        self,
        task: LLMTask,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> BaseModel:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.config.thinking,
        ).to(self.model.device)
        last_error: json.JSONDecodeError | ValidationError | None = None
        for _ in range(self.config.max_retries + 1):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            raw = self.processor.decode(generated, skip_special_tokens=True)
            try:
                return parse_final_json(raw, response_schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                # Raw output can contain hidden reasoning, so do not log or store it.
                last_error = exc
        raise RuntimeError(
            "Qwen did not return valid final JSON after "
            f"{self.config.max_retries + 1} attempts"
        ) from last_error
