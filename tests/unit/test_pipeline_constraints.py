from __future__ import annotations

from pathlib import Path

import pytest

from clinical_nlp.config import PipelineConfig
from clinical_nlp.icd_linking import ICDIndex
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
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def _pipeline(tmp_path: Path, response) -> ClinicalPipeline:
    return ClinicalPipeline(
        PipelineConfig(),
        ICDIndex({}),
        RxNormIndex(tmp_path / "rx.json", use_api=False),
        NoopNERBackend(),
        FakeLLM(response),
    )


def test_entity_review_cannot_change_positions(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="positions"):
        pipeline._review_entities(document, [proposal], {(0, 2): []})


def test_entity_review_rejects_assertions_on_lab_results(tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeError, match="invalid"):
        pipeline._review_entities(document, [proposal], {(0, 2): []})


def test_batch_rerank_cannot_invent_candidate_ids(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="invented"):
        pipeline._batch_rerank(
            task="icd_rerank",
            document=Document(id="x", text="bệnh x"),
            entries=[(proposal, [candidate])],
            limit=3,
        )


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
    assert pipeline.llm_backend.calls[0]["max_new_tokens"] == 1536


def test_incomplete_review_retries_once_then_fails(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, EntityReviewResponse(entities=[]))
    proposal = SpanProposal(
        start=0,
        end=2,
        text="ho",
        type=EntityType.SYMPTOM,
        source="test",
    )

    with pytest.raises(RuntimeError, match="invalid"):
        pipeline._review_entities(
            Document(id="x", text="ho"),
            [proposal],
            {(0, 2): []},
        )

    assert len(pipeline.llm_backend.calls) == 2


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
