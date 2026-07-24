from pathlib import Path

import pytest

from clinical_nlp.schemas import (
    Assertion,
    Document,
    Entity,
    EntityType,
)
from clinical_nlp.text import chunk_document, find_occurrence
from clinical_nlp.validation import validate_entities, validate_output_directory


def test_unicode_offsets_are_original_python_offsets() -> None:
    text = "Chào em, không ho nhưng sốt."
    start, end = find_occurrence(text, "ho")
    entity = Entity(
        text="ho",
        type=EntityType.SYMPTOM,
        assertions=[Assertion.NEGATED],
        position=(start, end),
    )
    validate_entities(Document(id="x", text=text), [entity])
    assert text[start:end] == "ho"


def test_chunks_are_exact_original_views() -> None:
    document = Document(id="x", text="Một câu.\n\nMột câu khác.\n" * 50)
    chunks = chunk_document(document, max_chars=80, overlap_chars=10)
    assert len(chunks) > 1
    assert all(document.text[row.start : row.end] == row.text for row in chunks)


def test_repeated_occurrence() -> None:
    text = "ho, không ho, vẫn ho"
    assert find_occurrence(text, "ho", 2) == (10, 12)


def test_overlap_is_rejected() -> None:
    document = Document(id="x", text="đau đầu")
    entities = [
        Entity(
            text="đau đầu",
            type=EntityType.SYMPTOM,
            position=(0, 7),
        ),
        Entity(
            text="đầu",
            type=EntityType.SYMPTOM,
            position=(4, 7),
        ),
    ]
    try:
        validate_entities(document, entities)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlap was accepted")


def test_output_directory_validation_checks_original_text(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "1.txt").write_text("Mẹ ho.", encoding="utf-8")
    (output_dir / "1.json").write_text(
        '[{"text":"sai","type":"TRIỆU_CHỨNG","assertions":[],"position":[3,6]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="substring mismatch"):
        validate_output_directory(output_dir, input_dir)
