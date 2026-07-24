from __future__ import annotations

import json
import re

from pydantic import BaseModel


THINK_END = "</think>"
FENCED_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def final_content_only(raw: str) -> str:
    if THINK_END in raw:
        raw = raw.rsplit(THINK_END, 1)[1]
    fenced = FENCED_RE.search(raw)
    if fenced:
        raw = fenced.group(1)
    raw = raw.strip()
    first_object = min(
        [index for index in (raw.find("{"), raw.find("[")) if index >= 0] or [0]
    )
    raw = raw[first_object:]
    return raw.strip()


def parse_final_json(raw: str, schema: type[BaseModel]) -> BaseModel:
    final = final_content_only(raw)
    payload = json.loads(final)
    return schema.model_validate(payload)

