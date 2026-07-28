from pathlib import Path

import pytest

from clinical_nlp.schemas import (
    Assertion,
    Document,
    Entity,
    EntityType,
)
from clinical_nlp.text import (
    chunk_document,
    find_occurrence,
    find_occurrence_relaxed,
    is_masked_span,
)
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


def test_canonically_equivalent_occurrence_maps_to_original_offsets() -> None:
    text = "có cục máu đông"

    start, end = find_occurrence(text, "có cục máu đông")

    assert text[start:end] == text
    assert (start, end) == (0, len(text))


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


def test_output_directory_can_validate_selected_subset(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "1.txt").write_text("ho", encoding="utf-8")
    (input_dir / "2.txt").write_text("sốt", encoding="utf-8")
    (output_dir / "1.json").write_text(
        '[{"text":"ho","type":"TRIỆU_CHỨNG","assertions":[],"position":[0,2]}]',
        encoding="utf-8",
    )

    validate_output_directory(output_dir, input_dir, expected_stems={"1"})


def test_non_linkable_output_includes_empty_candidates_and_round_trips() -> None:
    entity = Entity(
        text="ho",
        type=EntityType.SYMPTOM,
        position=(0, 2),
    )

    payload = entity.output_dict()

    assert payload["candidates"] == []
    assert set(payload) == {
        "text",
        "type",
        "candidates",
        "assertions",
        "position",
    }
    assert Entity.model_validate(payload) == entity


def test_non_linkable_output_can_omit_candidates_without_changing_default() -> None:
    entity = Entity(
        text="ho",
        type=EntityType.SYMPTOM,
        position=(0, 2),
    )

    strict_payload = entity.output_dict(omit_nonlinkable_candidates=True)

    assert "candidates" not in strict_payload
    assert strict_payload["assertions"] == []
    assert entity.output_dict()["candidates"] == []


def test_lab_entities_reject_non_empty_assertions() -> None:
    with pytest.raises(
        ValueError,
        match="assertions are only allowed",
    ):
        Entity(
            text="WBC",
            type=EntityType.TEST_NAME,
            assertions=[Assertion.HISTORICAL],
            position=(0, 3),
        )


def test_medication_candidates_are_numeric_rxnorm_identifier_strings() -> None:
    entity = Entity(
        text="amlodipine 10 mg po daily",
        type=EntityType.MEDICATION,
        candidates=["308135"],
        assertions=[Assertion.HISTORICAL],
        position=(58, 83),
    )

    assert entity.output_dict() == {
        "text": "amlodipine 10 mg po daily",
        "type": "THUỐC",
        "candidates": ["308135"],
        "assertions": ["isHistorical"],
        "position": [58, 83],
    }

    with pytest.raises(
        ValueError,
        match="medication candidates must be numeric RxNorm identifiers",
    ):
        Entity(
            text="amlodipine",
            type=EntityType.MEDICATION,
            candidates=["RX-308135"],
            position=(0, 10),
        )


def test_masked_placeholder_spans_are_detected() -> None:
    assert is_masked_span("************")
    assert is_masked_span("*******")
    assert is_masked_span("** **")
    # Real text alongside an asterisk is a genuine span.
    assert not is_masked_span("aspirin*")
    assert not is_masked_span("aspirin")
    assert not is_masked_span("")


def test_masked_placeholder_entity_is_rejected_by_validation() -> None:
    text = "Bác sĩ kê đơn thuốc ************ mỗi tối."
    document = Document(id="1", text=text)
    start = text.index("*")
    entity = Entity(
        text=text[start : start + 12],
        type=EntityType.MEDICATION,
        candidates=[],
        position=(start, start + 12),
    )
    with pytest.raises(ValueError, match="masked placeholder"):
        validate_entities(document, [entity])


def test_relaxed_occurrence_tolerates_whitespace_differences() -> None:
    text = "Bệnh nhân bị  đau   bụng dữ dội\nvà sốt cao."
    # The model reproduces the mention with single spaces; the source has runs.
    start, end = find_occurrence_relaxed(text, "đau bụng dữ dội")
    assert text[start:end] == "đau   bụng dữ dội"
    # Offsets still index the original text exactly.
    with pytest.raises(ValueError):
        find_occurrence(text, "đau bụng dữ dội")


def test_relaxed_occurrence_respects_occurrence_index() -> None:
    text = "sốt  cao rồi sốt cao lần nữa"
    first = find_occurrence_relaxed(text, "sốt cao", 1)
    second = find_occurrence_relaxed(text, "sốt cao", 2)
    assert text[first[0] : first[1]] == "sốt  cao"
    assert text[second[0] : second[1]] == "sốt cao"
    assert second[0] > first[0]
