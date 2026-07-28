from clinical_nlp.entity_finding import RuleEntityFinder, merge_proposals
from clinical_nlp.schemas import EntityType


TEXT = (
    "Danh sách thuốc trước nhập viện chính xác và đầy đủ.* *"
    "1. amlodipine 10 mg po daily* *"
    "2. aspirin 81 mg po daily* *"
    "3. metoprolol succinate xl 50 mg po daily* *"
    "4. guaifenesin ml po q6h:prn điều trị ho* *"
    "5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức* *"
    "6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau* *"
    "7. pravastatin 40 mg po daily* *"
    "8. docusate sodium 100 mg po bid điều trị táo bón* *"
    "9. senna 8.6 mg po bid:prn điều trị táo bón* *"
    "10. clonazepam 0.5 mg po qam:prn điều trị lo âu* *"
    "11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ"
)


EXPECTED = [
    ("amlodipine 10 mg po daily", EntityType.MEDICATION, (58, 83)),
    ("aspirin 81 mg po daily", EntityType.MEDICATION, (89, 111)),
    ("metoprolol succinate xl 50 mg po daily", EntityType.MEDICATION, (117, 155)),
    ("guaifenesin ml po q6h:prn", EntityType.MEDICATION, (161, 186)),
    ("ho", EntityType.SYMPTOM, (196, 198)),
    ("nystatin oral suspension 5 ml po qid:prn", EntityType.MEDICATION, (204, 244)),
    ("đau nhức", EntityType.SYMPTOM, (254, 262)),
    ("acetaminophen 325-650 mg po q6h:prn", EntityType.MEDICATION, (268, 303)),
    ("sốt đau", EntityType.SYMPTOM, (313, 320)),
    ("pravastatin 40 mg po daily", EntityType.MEDICATION, (326, 352)),
    ("docusate sodium 100 mg po bid", EntityType.MEDICATION, (358, 387)),
    ("táo bón", EntityType.SYMPTOM, (397, 404)),
    ("senna 8.6 mg po bid:prn", EntityType.MEDICATION, (410, 433)),
    ("táo bón", EntityType.SYMPTOM, (443, 450)),
    ("clonazepam 0.5 mg po qam:prn", EntityType.MEDICATION, (457, 485)),
    ("lo âu", EntityType.SYMPTOM, (495, 500)),
    ("clonazepam 1.5 mg po qhs", EntityType.MEDICATION, (507, 531)),
    ("lo âu", EntityType.SYMPTOM, (541, 546)),
    ("mất ngủ", EntityType.SYMPTOM, (547, 554)),
]


def test_golden_spans() -> None:
    proposals = merge_proposals(RuleEntityFinder().find(TEXT))
    actual = [(row.text, row.type, (row.start, row.end)) for row in proposals]
    assert actual == EXPECTED

