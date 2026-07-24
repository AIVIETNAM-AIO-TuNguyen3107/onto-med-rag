from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from clinical_nlp.llm.parsing import final_content_only, parse_final_json


class Smoke(BaseModel):
    status: str
    sum: int | None = None


class Entities(BaseModel):
    entities: list[dict]


def test_parse_ignores_trailing_text_after_json() -> None:
    raw = '<think>plan</think>\n{"status":"ok","sum":4}\nDone.'
    assert parse_final_json(raw, Smoke).sum == 4


def test_parse_prefers_last_json_object_when_extra_data() -> None:
    # Matches production failure mode: first complete object then more JSON.
    raw = (
        '<think>draft {"status":"bad"}</think>\n'
        '{"status":"draft","sum":1}{"status":"ok","sum":4}'
    )
    assert parse_final_json(raw, Smoke).sum == 4


def test_parse_rejects_unfinished_thinking_block() -> None:
    raw = '<think>still thinking {"status":"ok","sum":4}'
    with pytest.raises(json.JSONDecodeError, match="unfinished thinking"):
        parse_final_json(raw, Smoke)


def test_final_content_only_strips_think_and_fence() -> None:
    raw = '<think>x</think>\n```json\n{"status":"ok"}\n```'
    assert final_content_only(raw) == '{"status":"ok"}'


def test_parse_raises_when_no_schema_match() -> None:
    raw = '{"entities":[]}'
    with pytest.raises(ValidationError):
        parse_final_json(raw, Smoke)
