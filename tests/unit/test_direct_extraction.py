from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_nlp.direct_extraction import (
    CandidateOverride,
    OccurrenceOverride,
    RawEntity,
    LocalRxNormCatalog,
    apply_occurrence_overrides,
    automatic_selection,
    convert,
    link_rows,
    load_correction_overlay,
    load_raw_batches,
    reconstruct,
    select_retrieved_candidates,
    validate_strict_output_rows,
)
from clinical_nlp.icd_linking.index import ICDConcept, ICDIndex
from clinical_nlp.schemas import Document, EntityType, LinkCandidate


def _candidate(
    identifier: str,
    score: float,
    sources: list[str] | None = None,
) -> LinkCandidate:
    return LinkCandidate(
        identifier=identifier,
        name=identifier,
        score=score,
        retrieval_sources=sources or ["fuzzy"],
    )


def test_reconstruct_resolves_occurrences_and_keeps_assertions() -> None:
    document = Document(id="1", text="sốt cao rồi hết sốt cao sau đó")
    entities = [
        RawEntity(
            text="sốt cao",
            occurrence=2,
            type=EntityType.SYMPTOM,
            assertions=["isHistorical"],
        ),
        RawEntity(text="sốt cao", occurrence=1, type=EntityType.SYMPTOM),
    ]

    rows, failures = reconstruct(document, entities)

    assert failures == []
    assert [row["position"] for row in rows] == [[0, 7], [16, 23]]
    assert rows[0]["assertions"] == []
    assert rows[1]["assertions"] == ["isHistorical"]
    for row in rows:
        start, end = row["position"]
        assert document.text[start:end] == row["text"]


def test_reconstruct_reports_unresolvable_spans_without_guessing() -> None:
    document = Document(id="1", text="bệnh nhân sốt cao")
    entities = [
        RawEntity(text="sốt cao", occurrence=3, type=EntityType.SYMPTOM),
        RawEntity(text="không có ở đây", occurrence=1, type=EntityType.SYMPTOM),
    ]

    rows, failures = reconstruct(document, entities)

    assert rows == []
    assert len(failures) == 2
    assert {failure["text"] for failure in failures} == {"sốt cao", "không có ở đây"}


def test_automatic_selection_declines_ambiguous_and_empty() -> None:
    assert automatic_selection([]) == []
    # Two close candidates: no margin, so no code rather than the top hit.
    assert (
        automatic_selection([_candidate("A", 0.80), _candidate("B", 0.78)]) == []
    )
    # A single weak candidate is still not enough.
    assert automatic_selection([_candidate("A", 0.60)]) == []


def test_automatic_selection_accepts_exact_single_and_clear_margin() -> None:
    assert automatic_selection([_candidate("A", 0.95, ["exact"])]) == ["A"]
    assert automatic_selection([_candidate("A", 0.72)]) == ["A"]
    assert (
        automatic_selection([_candidate("A", 0.90), _candidate("B", 0.50)]) == ["A"]
    )


def test_top_or_drop_forces_only_an_existing_retrieved_candidate() -> None:
    weak = [_candidate("A", 0.20), _candidate("B", 0.19)]

    selected, mode, detail = select_retrieved_candidates(
        weak,
        threshold=0.55,
        policy="top-or-drop",
    )
    empty, empty_mode, empty_detail = select_retrieved_candidates(
        [],
        threshold=0.55,
        policy="top-or-drop",
    )

    assert selected == ["A"]
    assert mode == "forced_top"
    assert detail == weak[0]
    assert empty == []
    assert empty_mode == "no_candidate"
    assert empty_detail is None


def test_top_or_empty_forces_hits_but_keeps_no_hit_selection_empty() -> None:
    weak = [_candidate("A", 0.20)]

    selected, mode, _ = select_retrieved_candidates(
        weak,
        threshold=0.55,
        policy="top-or-empty",
    )
    empty, empty_mode, _ = select_retrieved_candidates(
        [],
        threshold=0.55,
        policy="top-or-empty",
    )

    assert selected == ["A"]
    assert mode == "forced_top"
    assert empty == []
    assert empty_mode == "no_candidate"


def test_occurrence_override_keeps_text_and_moves_to_standalone_mention() -> None:
    document = Document(id="7", text="BS nói em ăn nhiều ói bấy nhiêu")
    raw = [RawEntity(text="ói", occurrence=1, type=EntityType.SYMPTOM)]
    expected_start = document.text.index("ói", document.text.index("nói") + 3)
    override = OccurrenceOverride(
        document_id="7",
        text="ói",
        type=EntityType.SYMPTOM,
        from_occurrence=1,
        to_occurrence=2,
        expected_position=(expected_start, expected_start + 2),
        rationale="select the standalone symptom",
    )

    adjusted, applied = apply_occurrence_overrides("7", raw, [override])
    rows, failures = reconstruct(document, adjusted)

    assert failures == []
    assert rows[0]["text"] == "ói"
    assert rows[0]["position"] == [expected_start, expected_start + 2]
    assert applied[0]["from_occurrence"] == 1
    assert applied[0]["to_occurrence"] == 2


def test_v2_correction_overlay_locks_reviewed_code_fixes() -> None:
    root = Path(__file__).resolve().parents[2]
    overlay = load_correction_overlay(
        root / "configs" / "direct_extraction_v2_strict.corrections.json"
    )
    by_target = {
        (row.document_id, row.text): row.candidates
        for row in overlay.candidate_overrides
    }

    assert by_target[("31", "vô sinh thứ phát")] == ["N46"]
    assert by_target[("81", "tắc mạch do cục máu đông")] == ["I82.9"]
    assert by_target[("16", "bumetanide 2mg iv")] == ["1808"]
    assert by_target[("16", "levofloxacin 750mg iv")] == ["82122"]
    assert by_target[("33", "80mg lasix iv")] == ["202991"]
    assert by_target[("57", "iv lasix 40 mg once")] == ["202991"]
    assert by_target[("58", "iv lasix 40 mg once")] == ["202991"]


def _index(tmp_path: Path) -> tuple[ICDIndex, LocalRxNormCatalog]:
    icd = ICDIndex(
        {
            "I50": ICDConcept(code="I50", names=("suy tim",)),
            "K58": ICDConcept(code="K58", names=("hội chứng ruột kích thích",)),
        }
    )
    return icd, _catalog(tmp_path)


def _catalog(tmp_path: Path) -> LocalRxNormCatalog:
    directory = tmp_path / "rxnorm"
    directory.mkdir(exist_ok=True)
    (directory / "in.json").write_text(
        json.dumps([{"rxcui": "1191", "name": "aspirin", "tty": "IN"}]),
        encoding="utf-8",
    )
    (directory / "scd.json").write_text(
        json.dumps(
            [
                {
                    "rxcui": "243670",
                    "name": "aspirin 81 MG Oral Tablet",
                    "tty": "SCD",
                },
                {
                    "rxcui": "308135",
                    "name": "amlodipine 10 MG Oral Tablet",
                    "tty": "SCD",
                },
                {
                    "rxcui": "999999",
                    "name": "amlodipine 10 MG Oral Capsule",
                    "tty": "SCD",
                },
                {
                    "rxcui": "692836",
                    "name": "acetaminophen 325 MG / aspirin 81 MG Oral Powder",
                    "tty": "SCD",
                },
            ]
        ),
        encoding="utf-8",
    )
    (directory / "sbd.json").write_text(
        json.dumps(
            [
                {
                    "rxcui": "200809",
                    "name": "furosemide 40 MG Oral Tablet [Lasix]",
                    "tty": "SBD",
                }
            ]
        ),
        encoding="utf-8",
    )
    return LocalRxNormCatalog(directory)


def test_catalog_reproduces_the_organizer_gold_codes(tmp_path: Path) -> None:
    """The two SCD codes REVIEW cites as gold must come back exactly."""
    catalog = _catalog(tmp_path)

    assert catalog.match("amlodipine 10 mg po daily") == ["308135"]
    assert catalog.match("aspirin 81 MG") == ["243670"]


def test_catalog_prefers_scd_over_ingredient_and_declines_combinations(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)

    # A dose-bearing aspirin span reaches the SCD, not ingredient 1191.
    assert catalog.match("aspirin 81 MG x 1") == ["243670"]
    # Bare aspirin has no strength, so it falls back to the ingredient.
    assert catalog.match("aspirin") == ["1191"]
    # The 325 MG strength only appears inside a combination product, which must
    # never be coded from a single-ingredient span.
    assert catalog.match("acetaminophen 325mg") == []


def test_catalog_matches_a_brand_name_with_strength(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    assert catalog.match("80mg po lasix") == []
    assert catalog.match("iv lasix 40 mg once") == ["200809"]


def test_catalog_converts_units(tmp_path: Path) -> None:
    directory = tmp_path / "units"
    directory.mkdir()
    (directory / "in.json").write_text("[]", encoding="utf-8")
    (directory / "sbd.json").write_text("[]", encoding="utf-8")
    (directory / "scd.json").write_text(
        json.dumps(
            [
                {
                    "rxcui": "1665021",
                    "name": "ceftriaxone 1000 MG Injection",
                    "tty": "SCD",
                },
                {
                    "rxcui": "966223",
                    "name": "levothyroxine sodium 0.075 MG Oral Tablet",
                    "tty": "SCD",
                },
            ]
        ),
        encoding="utf-8",
    )
    catalog = LocalRxNormCatalog(directory)

    assert catalog.match("Ceftriaxone 1g") == ["1665021"]
    assert catalog.match("levothyroxine với liều 75 microgam/ngày") == ["966223"]


def test_link_rows_leaves_non_linkable_types_uncoded(tmp_path: Path) -> None:
    icd, rxnorm = _index(tmp_path)
    rows = [
        {
            "text": "WBC",
            "type": EntityType.TEST_NAME.value,
            "assertions": [],
            "position": [0, 3],
        },
        {
            "text": "14,43",
            "type": EntityType.TEST_RESULT.value,
            "assertions": [],
            "position": [4, 9],
        },
    ]

    entities = link_rows(rows, icd_index=icd, rxnorm_catalog=rxnorm)

    assert [entity.output_dict()["candidates"] for entity in entities] == [[], []]


def test_strict_output_requires_conditional_candidate_shape() -> None:
    rows = [
        {
            "text": "suy tim",
            "type": "CHẨN_ĐOÁN",
            "candidates": ["I50"],
            "assertions": [],
            "position": [0, 7],
        },
        {
            "text": "WBC",
            "type": "TÊN_XÉT_NGHIỆM",
            "assertions": [],
            "position": [8, 11],
        },
    ]

    validate_strict_output_rows(rows)

    with pytest.raises(ValueError, match="must omit"):
        validate_strict_output_rows(
            [{**rows[1], "candidates": []}]
        )
    with pytest.raises(ValueError, match="exactly one"):
        validate_strict_output_rows(
            [{**rows[0], "candidates": []}]
        )
    validate_strict_output_rows(
        [{**rows[0], "candidates": []}],
        require_linked_candidates=False,
    )


def test_link_rows_codes_an_exact_diagnosis(tmp_path: Path) -> None:
    icd, rxnorm = _index(tmp_path)
    rows = [
        {
            "text": "Suy tim",
            "type": EntityType.DIAGNOSIS.value,
            "assertions": ["isHistorical"],
            "position": [0, 7],
        }
    ]

    entities = link_rows(rows, icd_index=icd, rxnorm_catalog=rxnorm)

    assert entities[0].candidates == ["I50"]
    assert entities[0].output_dict()["assertions"] == ["isHistorical"]


def test_link_rows_applies_candidate_override_and_audits_it(
    tmp_path: Path,
) -> None:
    icd, rxnorm = _index(tmp_path)
    rows = [
        {
            "text": "suy tim",
            "type": EntityType.DIAGNOSIS.value,
            "assertions": [],
            "position": [0, 7],
        }
    ]
    override = CandidateOverride(
        document_id="1",
        text="suy tim",
        type=EntityType.DIAGNOSIS,
        position=(0, 7),
        candidates=["K58"],
        rationale="exercise the reviewed override path",
    )
    audit = {"dropped": [], "forced_candidates": [], "overrides": []}

    entities = link_rows(
        rows,
        icd_index=icd,
        rxnorm_catalog=rxnorm,
        candidate_policy="top-or-drop",
        document_id="1",
        candidate_overrides={
            ("1", EntityType.DIAGNOSIS.value, 0, 7): override
        },
        audit=audit,
    )

    assert entities[0].candidates == ["K58"]
    assert audit["overrides"][0]["from_candidates"] == ["I50"]
    assert audit["overrides"][0]["to_candidates"] == ["K58"]


def test_load_raw_batches_rejects_a_duplicated_document(tmp_path: Path) -> None:
    payload = [{"document_id": "1", "entities": []}]
    (tmp_path / "batch-01.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "batch-02.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="more than one batch"):
        load_raw_batches(tmp_path)


def test_convert_is_lossless_and_writes_competition_shape(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Bệnh nhân bị suy tim nặng.", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("batch-01.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "1",
                    "entities": [
                        {
                            "text": "suy tim",
                            "occurrence": 1,
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    icd_path = tmp_path / "icd.json"
    ICDIndex({"I50": ICDConcept(code="I50", names=("suy tim",))}).save(icd_path)

    result = convert(
        raw_dir=raw_dir,
        input_dir=input_dir,
        run_dir=tmp_path / "run",
        icd_index_path=icd_path,
        rxnorm_dir=_catalog(tmp_path).directory,
        audit_path=tmp_path / "audit.json",
    )

    assert result["lossless"] is True
    assert result["written_entities"] == 1
    assert result["unresolved"] == 0
    written = json.loads(
        (tmp_path / "run" / "outputs" / "1.json").read_text("utf-8")
    )
    assert written == [
        {
            "text": "suy tim",
            "type": "CHẨN_ĐOÁN",
            "candidates": ["I50"],
            "assertions": ["isHistorical"],
            "position": [13, 20],
        }
    ]
    manifest = json.loads(
        (tmp_path / "run" / "source_manifest.json").read_text("utf-8")
    )
    assert manifest["selection"]["document_ids"] == ["1"]


def test_convert_top_or_drop_writes_strict_shape_and_accounts_for_drop(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("ho bệnh lạ", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("batch-01.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "1",
                    "entities": [
                        {
                            "text": "ho",
                            "occurrence": 1,
                            "type": "TRIỆU_CHỨNG",
                            "assertions": [],
                        },
                        {
                            "text": "bệnh lạ",
                            "occurrence": 1,
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                        },
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    icd_path = tmp_path / "icd.json"
    ICDIndex({}).save(icd_path)
    run_dir = tmp_path / "run"
    audit_path = run_dir / "audit.json"

    result = convert(
        raw_dir=raw_dir,
        input_dir=input_dir,
        run_dir=run_dir,
        icd_index_path=icd_path,
        rxnorm_dir=_catalog(tmp_path).directory,
        audit_path=audit_path,
        candidate_policy="top-or-drop",
        omit_nonlinkable_candidates=True,
    )

    assert result["lossless"] is False
    assert result["accounted"] is True
    assert result["written_entities"] == 1
    assert result["dropped"] == 1
    assert result["dropped_assertions"] == 1
    assert json.loads((run_dir / "outputs" / "1.json").read_text("utf-8")) == [
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [0, 2],
        }
    ]
    audit = json.loads(audit_path.read_text("utf-8"))
    assert audit["counts"]["accounted"] is True
    assert audit["dropped"][0]["text"] == "bệnh lạ"


def test_convert_top_or_empty_retains_no_hit_linkable_entity(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("ho bệnh lạ", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("batch-01.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "1",
                    "entities": [
                        {
                            "text": "ho",
                            "occurrence": 1,
                            "type": "TRIỆU_CHỨNG",
                            "assertions": [],
                        },
                        {
                            "text": "bệnh lạ",
                            "occurrence": 1,
                            "type": "CHẨN_ĐOÁN",
                            "assertions": ["isHistorical"],
                        },
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    icd_path = tmp_path / "icd.json"
    ICDIndex({}).save(icd_path)
    run_dir = tmp_path / "run"
    audit_path = run_dir / "audit.json"

    result = convert(
        raw_dir=raw_dir,
        input_dir=input_dir,
        run_dir=run_dir,
        icd_index_path=icd_path,
        rxnorm_dir=_catalog(tmp_path).directory,
        audit_path=audit_path,
        candidate_policy="top-or-empty",
        omit_nonlinkable_candidates=True,
    )

    assert result["lossless"] is True
    assert result["accounted"] is True
    assert result["written_entities"] == 2
    assert result["dropped"] == 0
    assert result["retained_unlinked"] == 1
    assert json.loads((run_dir / "outputs" / "1.json").read_text("utf-8")) == [
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [0, 2],
        },
        {
            "text": "bệnh lạ",
            "type": "CHẨN_ĐOÁN",
            "candidates": [],
            "assertions": ["isHistorical"],
            "position": [3, 10],
        },
    ]
    audit = json.loads(audit_path.read_text("utf-8"))
    assert audit["counts"]["retained_unlinked"] == 1
    assert audit["retained_unlinked"][0]["text"] == "bệnh lạ"
