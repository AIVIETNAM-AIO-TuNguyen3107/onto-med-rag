from __future__ import annotations

import time
from pathlib import Path

import pytest

from clinical_nlp.config import PipelineConfig
from clinical_nlp.entity_finding import merge_proposals
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.llm.base import LLMTask
from clinical_nlp.ner.base import NoopNERBackend
from clinical_nlp.pipeline import (
    BatchCandidateSelectionResponse,
    CandidateSelection,
    ClinicalPipeline,
)
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.schemas import (
    Document,
    EntityType,
    LinkCandidate,
    SpanProposal,
)


class FakeLLM:
    name = "fake"

    def __init__(self, responses=None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def generate_json(self, task, messages, response_schema, **kwargs):
        self.calls.append({"task": task, "messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


def _pipeline(tmp_path: Path, llm: FakeLLM | None = None) -> ClinicalPipeline:
    config = PipelineConfig()
    config.run.llm_review_mode = "selective"
    config.llm.max_concurrency = 1
    return ClinicalPipeline(
        config,
        ICDIndex({}),
        RxNormIndex(tmp_path / "rxnorm.json", use_api=False),
        NoopNERBackend(),
        llm or FakeLLM(),
    )


def _proposal(
    text: str,
    *,
    start: int = 0,
    entity_type: EntityType = EntityType.DIAGNOSIS,
    source: str = "rule_dictionary",
    score: float = 0.8,
    evidence: dict | None = None,
) -> SpanProposal:
    return SpanProposal(
        start=start,
        end=start + len(text),
        text=text,
        type=entity_type,
        source=source,
        score=score,
        evidence=evidence or {},
    )


def _candidate(
    identifier: str,
    score: float,
    *,
    source: str = "fuzzy",
) -> LinkCandidate:
    return LinkCandidate(
        identifier=identifier,
        name=identifier,
        terminology_type="ICD10:disease",
        score=score,
        retrieval_sources=[source],
    )


def test_merge_records_cross_source_type_conflicts() -> None:
    rows = merge_proposals(
        [
            _proposal("ho", entity_type=EntityType.SYMPTOM, source="gliner"),
            _proposal(
                "ho",
                entity_type=EntityType.DIAGNOSIS,
                source="llm_recovery",
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0].evidence["type_conflict"] is True
    assert rows[0].evidence["supporting_sources"] == ["gliner", "llm_recovery"]
    assert rows[0].evidence["alternative_types"] == [
        EntityType.DIAGNOSIS.value,
        EntityType.SYMPTOM.value,
    ]


@pytest.mark.parametrize(
    "text,source",
    [
        ("băng phiến", "llm_recovery"),
        ("long não", "rule_dictionary"),
        ("thuốc", "gliner"),
        ("thuốc đông y", "structured_medication_rule"),
    ],
)
def test_post_merge_medication_filter_is_source_independent(
    tmp_path: Path,
    text: str,
    source: str,
) -> None:
    pipeline = _pipeline(tmp_path)
    kept, audit = pipeline._filter_merged_proposals(
        [
            _proposal(
                text,
                entity_type=EntityType.MEDICATION,
                source=source,
            )
        ]
    )

    assert kept == []
    assert audit[0]["decision_source"] == "deterministic_filter"


@pytest.mark.parametrize(
    "text,entity_type",
    [
        ("Xq28", EntityType.TEST_NAME),
        ("máu khô", EntityType.TEST_RESULT),
        ("xét nghiệm chuyên sâu", EntityType.TEST_NAME),
        ("đậu tằm", EntityType.SYMPTOM),
        ("nhận xét", EntityType.SYMPTOM),
        ("hiến máu", EntityType.DIAGNOSIS),
        (
            "Glucose-6-Phosphate Dehydrogenase",
            EntityType.TEST_NAME,
        ),
    ],
)
def test_post_merge_generic_non_entities_are_source_independent(
    tmp_path: Path,
    text: str,
    entity_type: EntityType,
) -> None:
    pipeline = _pipeline(tmp_path)
    kept, audit = pipeline._filter_merged_proposals(
        [
            _proposal(
                text,
                entity_type=entity_type,
                source="llm_recovery",
            )
        ]
    )

    assert kept == []
    assert audit[0]["reason"] == "source_independent_non_entity_exclusion"


def test_selective_uncertainty_gates(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    document = Document(id="1", text="Không có ho. 5,0 (cao)")
    cases = [
        (
            _proposal("ho", start=9, source="llm_recovery"),
            [],
            {"llm_recovery", "nearby_assertion_cue"},
        ),
        (
            _proposal("ho", start=9, source="gliner", score=0.5),
            [],
            {"low_confidence_gliner_only", "nearby_assertion_cue"},
        ),
        (
            _proposal(
                "ho",
                start=9,
                evidence={"type_conflict": True},
            ),
            [],
            {"type_conflict", "nearby_assertion_cue"},
        ),
        (
            _proposal(
                "5,0 (cao)",
                start=13,
                entity_type=EntityType.TEST_RESULT,
            ),
            [],
            {"laboratory_span_policy"},
        ),
    ]
    for proposal, assertions, expected in cases:
        reasons = pipeline._selective_review_reasons(
            document,
            proposal,
            assertions,
            [],
        )
        assert expected <= set(reasons)


@pytest.mark.parametrize(
    "text",
    [
        "6.3",
        "14,43",
        "38.3°C",
        "139/68 mmHg",
    ],
)
def test_valid_laboratory_result_spans_are_accepted(
    tmp_path: Path,
    text: str,
) -> None:
    pipeline = _pipeline(tmp_path)

    assert pipeline._valid_laboratory_result_span(text) is True


@pytest.mark.parametrize("text", ["5mg", "1 viên"])
def test_ambiguous_laboratory_result_spans_still_require_review(
    tmp_path: Path,
    text: str,
) -> None:
    pipeline = _pipeline(tmp_path)
    proposal = _proposal(
        text,
        entity_type=EntityType.TEST_RESULT,
        source="gliner",
        score=0.8,
    )

    reasons = pipeline._selective_review_reasons(
        Document(id="1", text=text),
        proposal,
        [],
        [],
    )

    assert "laboratory_span_policy" in reasons


def test_valid_low_confidence_gliner_result_bypasses_review(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path)
    proposal = _proposal(
        "80%",
        entity_type=EntityType.TEST_RESULT,
        source="gliner",
        score=0.5,
    )

    reasons = pipeline._selective_review_reasons(
        Document(id="1", text="80%"),
        proposal,
        [],
        [],
    )

    assert "low_confidence_gliner_only" not in reasons
    assert "laboratory_span_policy" not in reasons
    assert reasons == []


def test_exact_linking_threshold_and_normalized_deduplication(
    tmp_path: Path,
) -> None:
    llm = FakeLLM()
    pipeline = _pipeline(tmp_path, llm)
    first = _proposal("Thiếu men G6PD", start=0)
    second = _proposal("thiếu men g6pd", start=30)
    exact = _candidate("D55.0", 1.0, source="exact")
    weak = _candidate("E61.3", 0.49)

    selected, decisions = pipeline._select_terminology(
        LLMTask.ICD_RERANK,
        Document(id="1", text=""),
        [(first, [exact, weak]), (second, [exact, weak])],
        limit=3,
        checkpoint_dir=None,
    )

    assert [row.identifier for row in selected[(0, len(first.text))]] == ["D55.0"]
    assert [row.identifier for row in selected[(30, 30 + len(second.text))]] == [
        "D55.0"
    ]
    assert decisions[(0, len(first.text))]["decision_source"] == "automatic"
    assert (
        decisions[(30, 30 + len(second.text))]["decision_source"]
        == "reused_automatic"
    )
    assert llm.calls == []


def test_weak_links_become_empty_instead_of_hallucinated(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    proposal = _proposal("mơ hồ")

    selected, decisions = pipeline._select_terminology(
        LLMTask.ICD_RERANK,
        Document(id="1", text="mơ hồ"),
        [(proposal, [_candidate("A00", 0.4)])],
        limit=3,
        checkpoint_dir=None,
    )

    assert selected[(0, len(proposal.text))] == []
    assert decisions[(0, len(proposal.text))]["reason"] == "empty_after_threshold"


def test_ambiguous_normalized_mentions_share_one_llm_decision(
    tmp_path: Path,
) -> None:
    response = BatchCandidateSelectionResponse(
        selections=[
            CandidateSelection(position=(0, 6), candidates=["A00"])
        ]
    )
    llm = FakeLLM([response])
    pipeline = _pipeline(tmp_path, llm)
    first = _proposal("bệnh x", start=0)
    second = _proposal("BỆNH X", start=20)
    candidates = [_candidate("A00", 0.8), _candidate("A01", 0.7)]

    selected, decisions = pipeline._select_terminology(
        LLMTask.ICD_RERANK,
        Document(id="1", text="bệnh x"),
        [(first, candidates), (second, candidates)],
        limit=3,
        checkpoint_dir=None,
    )

    assert len(llm.calls) == 1
    assert [row.identifier for row in selected[(0, 6)]] == ["A00"]
    assert [row.identifier for row in selected[(20, 26)]] == ["A00"]
    assert decisions[(20, 26)]["decision_source"] == "reused_llm"


def test_parallel_map_preserves_input_order(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.config.llm.max_concurrency = 2

    def finish_out_of_order(value: int) -> int:
        time.sleep(0.01 * (3 - value))
        return value

    assert pipeline._parallel_map([1, 2, 3], finish_out_of_order) == [1, 2, 3]


def test_review_mode_legacy_mapping_and_conflicts() -> None:
    assert (
        PipelineConfig.model_validate(
            {"run": {"llm_full_review": True}}
        ).run.llm_review_mode
        == "full"
    )
    assert (
        PipelineConfig.model_validate(
            {"run": {"llm_review_mode": "selective"}}
        ).run.llm_review_mode
        == "selective"
    )
    with pytest.raises(ValueError, match="conflicts"):
        PipelineConfig.model_validate(
            {
                "run": {
                    "llm_full_review": True,
                    "llm_review_mode": "selective",
                }
            }
        )
