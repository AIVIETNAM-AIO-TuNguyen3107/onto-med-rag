from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError


THINK_START = "<think>"
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


def _iter_json_values(text: str) -> list[object]:
    decoder = json.JSONDecoder()
    values: list[object] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index] not in "{[":
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        values.append(value)
        index = end
    return values


def parse_final_json(raw: str, schema: type[BaseModel]) -> BaseModel:
    # Unfinished thinking often embeds draft JSON; force a retry instead of
    # validating the wrong object.
    if THINK_START in raw and THINK_END not in raw:
        raise json.JSONDecodeError("unfinished thinking block", raw, 0)
    final = final_content_only(raw)
    values = _iter_json_values(final)
    if not values:
        raise json.JSONDecodeError("No JSON object found", final, 0)
    last_error: ValidationError | None = None
    # Prefer the last schema-valid value (final answer after drafts / Extra data).
    for value in reversed(values):
        try:
            return schema.model_validate(value)
        except ValidationError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error
