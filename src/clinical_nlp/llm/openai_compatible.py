from __future__ import annotations

import os
from typing import Any

import requests
from pydantic import BaseModel

from clinical_nlp.config import ModelConfig
from clinical_nlp.llm.base import LLMTask
from clinical_nlp.llm.parsing import parse_final_json


class OpenAICompatibleBackend:
    name = "openai_compatible"

    def __init__(self, config: ModelConfig) -> None:
        if not config.endpoint:
            raise ValueError("OpenAI-compatible backend requires an endpoint")
        self.config = config

    def generate_json(
        self,
        task: LLMTask,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
    ) -> BaseModel:
        endpoint = self.config.endpoint.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": self.config.thinking,
                }
            },
        }
        response = requests.post(endpoint, json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        row = response.json()["choices"][0]["message"]
        # Deliberately ignore/discard row.get("reasoning_content").
        return parse_final_json(row["content"], response_schema)

