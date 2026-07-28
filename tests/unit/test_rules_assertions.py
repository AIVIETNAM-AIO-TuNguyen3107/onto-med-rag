from clinical_nlp.assertion_detection import AssertionDetector
from clinical_nlp.entity_finding import RuleEntityFinder, merge_proposals
from clinical_nlp.schemas import Assertion, EntityType, SpanProposal


def _proposal(text: str, target: str, entity_type: EntityType) -> SpanProposal:
    start = text.index(target)
    return SpanProposal(
        start=start,
        end=start + len(target),
        text=target,
        type=entity_type,
        source="test",
        score=1,
    )


def test_medication_expansion_stops_at_indication() -> None:
    text = "guaifenesin ml po q6h:prn điều trị ho"
    proposals = RuleEntityFinder().find(text)
    medication = next(row for row in proposals if row.type == EntityType.MEDICATION)
    assert medication.text == "guaifenesin ml po q6h:prn"


def test_lab_value_includes_contiguous_unit() -> None:
    text = "Glucose: 7,2 mmol/L (3,9-6,4) H"
    proposals = RuleEntityFinder().find(text)
    result = next(row for row in proposals if row.type == EntityType.TEST_RESULT)
    assert result.text == "7,2 mmol/L"


def test_negation_scope_stops_at_contrast() -> None:
    text = "Bệnh nhân không ho nhưng đau ngực."
    detector = AssertionDetector()
    ho = detector.detect(text, _proposal(text, "ho", EntityType.SYMPTOM))
    pain = detector.detect(text, _proposal(text, "đau ngực", EntityType.SYMPTOM))
    assert Assertion.NEGATED in ho
    assert Assertion.NEGATED not in pain


def test_family_context_and_false_positives() -> None:
    detector = AssertionDetector()
    positive = "Mẹ em bị bệnh động mạch vành."
    labels = detector.detect(
        positive,
        _proposal(positive, "bệnh động mạch vành", EntityType.DIAGNOSIS),
    )
    assert Assertion.FAMILY in labels

    current_patient = "Em bị tăng huyết áp và hồi hộp."
    labels = detector.detect(
        current_patient,
        _proposal(current_patient, "tăng huyết áp", EntityType.DIAGNOSIS),
    )
    assert Assertion.FAMILY not in labels

    animal = "Em chơi với một con chó con có nguy cơ bệnh dại."
    labels = detector.detect(
        animal,
        _proposal(animal, "bệnh dại", EntityType.DIAGNOSIS),
    )
    assert Assertion.FAMILY not in labels


def test_merge_is_flat() -> None:
    proposals = [
        SpanProposal(
            start=0,
            end=7,
            text="đau đầu",
            type=EntityType.SYMPTOM,
            source="rule_dictionary",
            score=0.8,
        ),
        SpanProposal(
            start=4,
            end=7,
            text="đầu",
            type=EntityType.SYMPTOM,
            source="gliner",
            score=0.9,
        ),
    ]
    assert [row.text for row in merge_proposals(proposals)] == ["đau đầu"]


def _span(start: int, end: int, source: str, score: float) -> SpanProposal:
    return SpanProposal(
        start=start,
        end=end,
        text="x" * (end - start),
        type=EntityType.SYMPTOM,
        source=source,
        score=score,
    )


def test_merge_prefers_two_granular_spans_over_one_enclosing_span() -> None:
    # Gold is granular: the organizer splits "lo âu mất ngủ" into two symptoms.
    selected = merge_proposals(
        [
            _span(0, 20, "gliner", 0.8),
            _span(0, 9, "gliner", 0.7),
            _span(10, 20, "gliner", 0.7),
        ]
    )
    assert [(row.start, row.end) for row in selected] == [(0, 9), (10, 20)]


def test_merge_keeps_structured_span_over_weak_fragments() -> None:
    # Counting entities must not shred a high-confidence rule match.
    selected = merge_proposals(
        [
            _span(0, 20, "structured_lab_rule", 0.95),
            _span(0, 9, "fuzzy_dictionary", 0.30),
            _span(10, 20, "fuzzy_dictionary", 0.30),
        ]
    )
    assert [(row.start, row.end) for row in selected] == [(0, 20)]
    assert selected[0].source == "structured_lab_rule"


def test_merge_output_is_sorted_and_non_overlapping() -> None:
    selected = merge_proposals(
        [
            _span(30, 40, "gliner", 0.6),
            _span(0, 20, "gliner", 0.8),
            _span(0, 9, "gliner", 0.7),
            _span(10, 20, "gliner", 0.7),
            _span(15, 35, "fuzzy_dictionary", 0.4),
        ]
    )
    positions = [(row.start, row.end) for row in selected]
    assert positions == sorted(positions)
    assert all(
        left[1] <= right[0] for left, right in zip(positions, positions[1:])
    )


def test_merge_is_order_independent() -> None:
    proposals = [
        _span(0, 20, "gliner", 0.8),
        _span(0, 9, "gliner", 0.7),
        _span(10, 20, "gliner", 0.7),
        _span(25, 30, "gliner", 0.5),
    ]
    expected = [(row.start, row.end) for row in merge_proposals(proposals)]
    for rotation in range(len(proposals)):
        shuffled = proposals[rotation:] + proposals[:rotation]
        assert [
            (row.start, row.end) for row in merge_proposals(shuffled)
        ] == expected



def test_numbered_section_heading_is_recognised() -> None:
    # The corpus writes headings as outline items, not bare lines.
    detector = AssertionDetector()
    text = "1.  Tiền sử bệnh\n- Bệnh nhân bị hen suyễn nhiều năm.\n"
    start = text.index("hen suyễn")
    proposal = SpanProposal(
        start=start,
        end=start + len("hen suyễn"),
        text="hen suyễn",
        type=EntityType.DIAGNOSIS,
        source="test",
    )

    assert detector.section_at(text, start) == "past_history"
    assert Assertion.HISTORICAL in detector.detect(text, proposal)


def test_prose_diagnosis_mention_does_not_mask_the_real_heading() -> None:
    # "được chẩn đoán ..." is prose; returning "diagnosis" here would end the
    # backward scan before reaching the Tiền sử heading above it.
    detector = AssertionDetector()
    text = (
        "2. Tiền sử bệnh nội khoa\n"
        "Bệnh nhân được chẩn đoán đái tháo đường năm 2019.\n"
        "Hiện tại còn tăng huyết áp.\n"
    )
    start = text.index("tăng huyết áp")

    assert detector.section_at(text, start) == "past_history"


def test_heading_shaped_diagnosis_line_still_matches() -> None:
    detector = AssertionDetector()
    text = "Chẩn đoán hình ảnh\n- chụp ct sọ não: âm tính\n"
    start = text.index("chụp ct")

    assert detector.section_at(text, start) == "diagnosis"


def test_negation_bridges_a_detection_verb() -> None:
    detector = AssertionDetector()
    text = "Trẻ không được phát hiện thiếu men G6PD khi sàng lọc."
    start = text.index("thiếu men G6PD")
    proposal = SpanProposal(
        start=start,
        end=start + len("thiếu men G6PD"),
        text="thiếu men G6PD",
        type=EntityType.DIAGNOSIS,
        source="test",
    )

    assert Assertion.NEGATED in detector.detect(text, proposal)


def test_negation_does_not_leak_across_unrelated_text() -> None:
    detector = AssertionDetector()
    text = "Bệnh không lây từ trẻ này sang trẻ khác nhưng gây thiếu máu."
    start = text.index("thiếu máu")
    proposal = SpanProposal(
        start=start,
        end=start + len("thiếu máu"),
        text="thiếu máu",
        type=EntityType.SYMPTOM,
        source="test",
    )

    assert Assertion.NEGATED not in detector.detect(text, proposal)
