import json
from pathlib import Path

from src.eval.metrics import score_sample
from src.extract.rules import extract_entities
from src.schemas.validate import validate_entities


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INPUT = (ROOT / "data/examples/sample_input.txt").read_text(encoding="utf-8")
SAMPLE_OUTPUT = json.loads(
    (ROOT / "data/examples/sample_output.json").read_text(encoding="utf-8")
)


def test_validate_accepts_sample_output():
    assert validate_entities(SAMPLE_OUTPUT, SAMPLE_INPUT) == []


def test_validate_rejects_bad_span():
    bad = [
        {
            "text": "wrong",
            "position": [0, 5],
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "candidates": [],
        }
    ]
    errors = validate_entities(bad, SAMPLE_INPUT)
    assert any("does not match" in e for e in errors)


def test_extract_finds_symptoms_and_diagnosis():
    entities = extract_entities(SAMPLE_INPUT)
    types = {e["type"] for e in entities}
    assert "TRIỆU_CHỨNG" in types
    assert "CHẨN_ĐOÁN" in types
    assert "THUỐC" in types


def test_extract_positions_match_text():
    entities = extract_entities(SAMPLE_INPUT)
    for ent in entities:
        s, e = ent["position"]
        assert ent["text"] == SAMPLE_INPUT[s:e]


def test_score_sample_perfect_match():
    s = score_sample(SAMPLE_OUTPUT, SAMPLE_OUTPUT)
    assert s.final == 1.0
