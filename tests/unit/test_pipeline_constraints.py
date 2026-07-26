from __future__ import annotations

from pathlib import Path

import pytest

from clinical_nlp.config import PipelineConfig
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.llm.base import LLMDecisionError
from clinical_nlp.ner.base import NoopNERBackend
from clinical_nlp.pipeline import (
    BatchCandidateSelectionResponse,
    CandidateSelection,
    ClinicalPipeline,
    EntityRecoveryResponse,
    EntityReviewResponse,
    RecoveredEntity,
    ReviewedEntity,
)
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.schemas import (
    Assertion,
    Document,
    EntityType,
    LinkCandidate,
    SpanProposal,
)


class FakeLLM:
    name = "fake"

    def __init__(self, response) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[dict] = []

    def generate_json(
        self,
        task,
        messages,
        response_schema,
        max_new_tokens=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "task": task,
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                **kwargs,
            }
        )
        response = (
            self.responses.pop(0)
            if len(self.responses) > 1
            else self.responses[0]
        )
        if isinstance(response, Exception):
            raise response
        return response


def _pipeline(tmp_path: Path, response) -> ClinicalPipeline:
    return ClinicalPipeline(
        PipelineConfig(),
        ICDIndex({}),
        RxNormIndex(tmp_path / "rx.json", use_api=False),
        NoopNERBackend(),
        FakeLLM(response),
    )


def test_entity_review_invalid_positions_preserve_original_entity(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(1, 3),
                    keep=True,
                    type=EntityType.SYMPTOM,
                )
            ]
        ),
    )
    document = Document(id="x", text="ho")
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="test",
    )

    warnings: list[str] = []
    reviewed, assertions, artifacts = pipeline._review_entities(
        document,
        [proposal],
        {(0, 2): []},
        warnings=warnings,
    )

    assert reviewed == [proposal]
    assert assertions == {(0, 2): []}
    assert artifacts[0]["decision_source"] == "deterministic_fallback"
    assert "positions" in artifacts[0]["decision_error"]
    assert "deterministic safe defaults" in warnings[0]
    assert len(pipeline.llm_backend.calls) == 2


def test_entity_review_prompt_is_recall_biased_and_type_conservative(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(0, 2),
                    keep=True,
                    type=EntityType.SYMPTOM,
                )
            ]
        ),
    )
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="llm_recovery",
    )

    pipeline._review_entities(
        Document(id="x", text="ho"),
        [proposal],
        {(0, 2): []},
    )

    prompt = pipeline.llm_backend.calls[0]["messages"][0]["content"]
    assert "Prefer keep=true" in prompt
    assert "only for clearly generic, non-clinical, or unsupported" in prompt
    assert "Preserve the supplied type" in prompt
    assert "never retype speculatively" in prompt


def test_invalid_lab_assertions_preserve_original_entity(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(0, 2),
                    keep=True,
                    type=EntityType.TEST_RESULT,
                    assertions=[Assertion.HISTORICAL],
                )
            ]
        ),
    )
    document = Document(id="x", text="12")
    proposal = SpanProposal(
        start=0,
        end=2,
        text="12",
        type=EntityType.TEST_RESULT,
        source="test",
    )

    reviewed, assertions, artifacts = pipeline._review_entities(
        document,
        [proposal],
        {(0, 2): []},
    )

    assert reviewed == [proposal]
    assert assertions == {(0, 2): []}
    assert artifacts[0]["decision_source"] == "deterministic_fallback"


def test_entity_review_allows_unsupported_type_only_on_rejected_row(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(0, 2),
                    keep=False,
                    type="OTHER",
                )
            ]
        ),
    )
    document = Document(id="x", text="ho")
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="test",
    )
    warnings: list[str] = []

    reviewed, reviewed_assertions, artifacts = pipeline._review_entities(
        document,
        [proposal],
        {(0, 2): []},
        warnings=warnings,
    )

    assert reviewed == []
    assert reviewed_assertions == {}
    assert artifacts[0]["returned_type"] == "OTHER"
    assert artifacts[0]["reviewed_type"] == EntityType.SYMPTOM.value
    assert artifacts[0]["unsupported_returned_type"] is True
    assert "unsupported type labels" in warnings[0]
    assert len(pipeline.llm_backend.calls) == 1


def test_entity_review_retries_then_preserves_unsupported_kept_type(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(0, 2),
                    keep=True,
                    type="OTHER",
                )
            ]
        ),
    )
    document = Document(id="x", text="ho")
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="test",
    )

    reviewed, assertions, artifacts = pipeline._review_entities(
        document,
        [proposal],
        {(0, 2): []},
    )

    assert reviewed == [proposal]
    assert assertions == {(0, 2): []}
    assert artifacts[0]["decision_source"] == "deterministic_fallback"
    assert len(pipeline.llm_backend.calls) == 2
    assert pipeline.llm_backend.calls[0]["reasoning_enabled"] is True
    assert pipeline.llm_backend.calls[1]["reasoning_enabled"] is False


def test_batch_rerank_invented_ids_fall_back_to_empty(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        BatchCandidateSelectionResponse(
            selections=[
                CandidateSelection(
                    position=(0, 6),
                    candidates=["INVENTED"],
                    confidence=1.0,
                )
            ]
        ),
    )
    proposal = SpanProposal(
        start=0,
        end=6,
        text="bệnh x",
        type=EntityType.DIAGNOSIS,
        source="test",
    )
    candidate = LinkCandidate(
        identifier="A00",
        name="Bệnh tả",
        terminology_type="ICD10:disease",
    )

    selected = pipeline._batch_rerank(
        task="icd_rerank",
        document=Document(id="x", text="bệnh x"),
        entries=[(proposal, [candidate])],
        limit=3,
    )

    assert selected[(0, 6)] == []
    assert len(pipeline.llm_backend.calls) == 2


def test_entity_review_is_batched_at_ten(tmp_path: Path) -> None:
    proposals = [
        SpanProposal(
            start=index * 2,
            end=index * 2 + 1,
            text="x",
            type=EntityType.SYMPTOM,
            source="test",
        )
        for index in range(21)
    ]
    responses = [
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(row.start, row.end),
                    keep=True,
                    type=row.type,
                )
                for row in proposals[:10]
            ]
        ),
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(row.start, row.end),
                    keep=True,
                    type=row.type,
                )
                for row in proposals[10:20]
            ]
        ),
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(proposals[20].start, proposals[20].end),
                    keep=True,
                    type=proposals[20].type,
                )
            ]
        ),
    ]
    pipeline = _pipeline(tmp_path, responses)
    pipeline.config.llm.max_concurrency = 1

    pipeline._review_entities(
        Document(id="x", text="x " * 21),
        proposals,
        {(row.start, row.end): [] for row in proposals},
    )

    assert len(pipeline.llm_backend.calls) == 3
    assert {
        row["max_new_tokens"] for row in pipeline.llm_backend.calls
    } == {2048}


def test_recovery_audits_unknown_types_and_uses_chunk_budget(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityRecoveryResponse(
            entities=[
                RecoveredEntity(
                    text="ho",
                    occurrence=1,
                    type=EntityType.SYMPTOM.value,
                ),
                RecoveredEntity(
                    text="thủ thuật",
                    occurrence=1,
                    type="THỦ_TỤC",
                ),
            ]
        ),
    )
    document = Document(id="x", text="ho và thủ thuật")
    from clinical_nlp.text import chunk_document

    chunks = chunk_document(document, max_chars=100, overlap_chars=10)
    warnings: list[str] = []
    proposals, audit = pipeline._recover_entities(
        document,
        chunks,
        [],
        warnings,
    )

    assert [(row.text, row.type) for row in proposals] == [
        ("ho", EntityType.SYMPTOM)
    ]
    assert audit[0]["reason"] == "unsupported_entity_type"
    assert "unsupported entity types" in warnings[0]
    assert pipeline.llm_backend.calls[0]["max_new_tokens"] == 4096
    assert pipeline.llm_backend.calls[0]["reasoning_enabled"] is False
    assert (
        pipeline.model_metadata()["llm"]["task_reasoning"]["entity_recovery"]
        is False
    )


def test_recovery_schema_constrains_model_to_allowed_entity_types() -> None:
    schema = RecoveredEntity.model_json_schema()

    assert set(schema["properties"]["type"]["enum"]) == {
        entity_type.value for entity_type in EntityType
    }


def test_recovery_scans_independently_and_tracks_existing_occurrences(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityRecoveryResponse(
            entities=[
                RecoveredEntity(
                    text="ho",
                    occurrence=2,
                    type=EntityType.SYMPTOM.value,
                )
            ]
        ),
    )
    document = Document(id="x", text="ho rồi ho")
    existing = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="gliner",
    )
    from clinical_nlp.text import chunk_document

    proposals, _ = pipeline._recover_entities(
        document,
        chunk_document(document, max_chars=100, overlap_chars=10),
        [existing],
        [],
    )

    assert [(row.text, row.start, row.end) for row in proposals] == [
        ("ho", 7, 9)
    ]
    system_prompt = pipeline.llm_backend.calls[0]["messages"][0]["content"]
    user_prompt = pipeline.llm_backend.calls[0]["messages"][1]["content"]
    assert "Independently and exhaustively" in system_prompt
    assert "not an exclusion list" in system_prompt
    assert "Return every entity" in system_prompt
    assert "Never generate offsets" in system_prompt
    assert '"occurrence": 1' in user_prompt
    assert '"text": "ho"' in user_prompt


def test_recovery_corrects_only_unambiguous_near_match_entity_types(
    tmp_path: Path,
) -> None:
    text = "đại tiện ra máu đỏ tươi gián đoạn"
    pipeline = _pipeline(
        tmp_path,
        EntityRecoveryResponse(
            entities=[
                RecoveredEntity(
                    text=text,
                    occurrence=1,
                    type="TRIỆU_CHUGHT",
                )
            ]
        ),
    )
    document = Document(id="x", text=text)
    from clinical_nlp.text import chunk_document

    proposals, audit = pipeline._recover_entities(
        document,
        chunk_document(document, max_chars=100, overlap_chars=10),
        [],
        [],
    )

    assert [(row.text, row.type) for row in proposals] == [
        (text, EntityType.SYMPTOM)
    ]
    assert audit[0]["status"] == "corrected"
    assert audit[0]["corrected_type"] == EntityType.SYMPTOM.value


def test_suspicious_recovery_chunk_retries_once_without_reasoning(
    tmp_path: Path,
) -> None:
    pathological = EntityRecoveryResponse(
        entities=[
            RecoveredEntity(
                text="x",
                occurrence=1,
                type=EntityType.SYMPTOM.value,
            )
            for _ in range(41)
        ]
    )
    clean = EntityRecoveryResponse(
        entities=[
            RecoveredEntity(
                text="ho",
                occurrence=1,
                type=EntityType.SYMPTOM.value,
            )
        ]
    )
    pipeline = _pipeline(tmp_path, [pathological, clean])
    document = Document(id="x", text="ho")
    from clinical_nlp.text import chunk_document

    warnings: list[str] = []
    proposals, audit = pipeline._recover_entities(
        document,
        chunk_document(document, max_chars=100, overlap_chars=10),
        [],
        warnings,
    )

    assert [row.text for row in proposals] == ["ho"]
    assert audit[0]["status"] == "retried"
    assert audit[0]["initial_row_count"] == 41
    assert "retried 1 suspicious" in warnings[0]
    assert len(pipeline.llm_backend.calls) == 2
    assert [
        row["reasoning_enabled"] for row in pipeline.llm_backend.calls
    ] == [False, False]
    assert pipeline.llm_backend.calls[1]["call_id"].endswith(
        "-quality-fallback"
    )


def test_repeated_suspicious_recovery_chunk_is_discarded(
    tmp_path: Path,
) -> None:
    pathological = EntityRecoveryResponse(
        entities=[
            RecoveredEntity(
                text="x",
                occurrence=1,
                type=EntityType.SYMPTOM.value,
            )
            for _ in range(41)
        ]
    )
    pipeline = _pipeline(tmp_path, [pathological, pathological])
    document = Document(id="x", text="ho")
    from clinical_nlp.text import chunk_document

    warnings: list[str] = []
    proposals, audit = pipeline._recover_entities(
        document,
        chunk_document(document, max_chars=100, overlap_chars=10),
        [],
        warnings,
    )

    assert proposals == []
    assert audit[0]["status"] == "rejected"
    assert audit[0]["retry_row_count"] == 41
    assert "discarded 1 suspicious" in warnings[0]
    assert len(pipeline.llm_backend.calls) == 2


def test_incomplete_review_retries_once_then_preserves_original(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path, EntityReviewResponse(entities=[]))
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="test",
    )

    warnings: list[str] = []
    reviewed, assertions, artifacts = pipeline._review_entities(
        Document(id="x", text="ho"),
        [proposal],
        {(0, 2): []},
        warnings=warnings,
    )

    assert reviewed == [proposal]
    assert assertions == {(0, 2): []}
    assert artifacts[0]["decision_source"] == "deterministic_fallback"
    assert "omitted required positions" in artifacts[0]["decision_error"]
    assert "deterministic safe defaults" in warnings[0]
    assert len(pipeline.llm_backend.calls) == 2


def test_structured_review_failures_preserve_original(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        [
            LLMDecisionError("invalid structured response"),
            LLMDecisionError("invalid structured fallback"),
        ],
    )
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="test",
    )
    warnings: list[str] = []

    reviewed, assertions, artifacts = pipeline._review_entities(
        Document(id="x", text="ho"),
        [proposal],
        {(0, 2): []},
        warnings=warnings,
    )

    assert reviewed == [proposal]
    assert assertions == {(0, 2): []}
    assert artifacts[0]["decision_source"] == "deterministic_fallback"
    assert "structured response failure" in warnings[0]


def test_terminal_review_fallback_drops_invalid_lab_span(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        EntityReviewResponse(
            entities=[
                ReviewedEntity(
                    position=(0, 6),
                    keep=True,
                    type="OTHER",
                )
            ]
        ),
    )
    proposal = SpanProposal(
        start=0,
        end=6,
        text="1 viên",
        type=EntityType.TEST_RESULT,
        source="gliner",
    )

    reviewed, assertions, artifacts = pipeline._review_entities(
        Document(id="x", text="1 viên"),
        [proposal],
        {(0, 6): []},
    )

    assert reviewed == []
    assert assertions == {}
    assert artifacts[0]["decision_source"] == "deterministic_fallback"
    assert artifacts[0]["keep"] is False


def test_terminology_reranking_is_batched_at_ten(tmp_path: Path) -> None:
    proposals = [
        SpanProposal(
            start=index * 2,
            end=index * 2 + 1,
            text="x",
            type=EntityType.DIAGNOSIS,
            source="test",
        )
        for index in range(11)
    ]
    entries = [
        (
            proposal,
            [
                LinkCandidate(
                    identifier=f"A{index}",
                    name=f"Diagnosis {index}",
                    terminology_type="ICD10:disease",
                )
            ],
        )
        for index, proposal in enumerate(proposals)
    ]
    responses = [
        BatchCandidateSelectionResponse(
            selections=[
                CandidateSelection(
                    position=(proposal.start, proposal.end),
                    candidates=[f"A{index}"],
                )
                for index, proposal in enumerate(proposals[:10])
            ]
        ),
        BatchCandidateSelectionResponse(
            selections=[
                CandidateSelection(
                    position=(proposals[10].start, proposals[10].end),
                    candidates=["A10"],
                )
            ]
        ),
    ]
    pipeline = _pipeline(tmp_path, responses)
    pipeline.config.llm.max_concurrency = 1

    selected = pipeline._batch_rerank(
        task="icd_rerank",
        document=Document(id="x", text="x " * 11),
        entries=entries,
        limit=3,
    )

    assert len(pipeline.llm_backend.calls) == 2
    assert {
        row["max_new_tokens"] for row in pipeline.llm_backend.calls
    } == {1536}
    assert selected[(20, 21)][0].identifier == "A10"


def test_duplicate_candidate_decision_retries_then_succeeds(
    tmp_path: Path,
) -> None:
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.DIAGNOSIS,
        source="test",
    )
    candidate = LinkCandidate(
        identifier="A1",
        name="Diagnosis",
        terminology_type="ICD10:disease",
    )
    duplicate = BatchCandidateSelectionResponse(
        selections=[
            CandidateSelection(
                position=(0, 2),
                candidates=["A1", "A1"],
            )
        ]
    )
    valid = BatchCandidateSelectionResponse(
        selections=[
            CandidateSelection(
                position=(0, 2),
                candidates=["A1"],
            )
        ]
    )
    pipeline = _pipeline(tmp_path, [duplicate, valid])

    selected = pipeline._batch_rerank(
        task="icd_rerank",
        document=Document(id="x", text="ho"),
        entries=[(proposal, [candidate])],
        limit=3,
    )

    assert len(pipeline.llm_backend.calls) == 2
    assert selected[(0, 2)][0].identifier == "A1"
