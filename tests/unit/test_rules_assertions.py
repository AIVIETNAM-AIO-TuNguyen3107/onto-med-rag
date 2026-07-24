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

