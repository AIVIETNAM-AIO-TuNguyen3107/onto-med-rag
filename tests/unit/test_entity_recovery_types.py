from __future__ import annotations

from clinical_nlp.pipeline import EntityRecoveryResponse


def test_recovery_drops_unknown_entity_types() -> None:
    parsed = EntityRecoveryResponse.model_validate(
        {
            "entities": [
                {"text": "ho", "occurrence": 1, "type": "TRIỆU_CHỨNG"},
                {"text": "nội soi", "occurrence": 1, "type": "THỦ_TỤC"},
                {"text": "THA", "occurrence": 1, "type": "CHẨN_ĐOÁN"},
            ]
        }
    )
    assert [row.type.value for row in parsed.entities] == [
        "TRIỆU_CHỨNG",
        "CHẨN_ĐOÁN",
    ]

