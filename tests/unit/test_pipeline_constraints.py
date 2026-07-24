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
    EntityReviewResponse,
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
        self.response = response

    def generate_json(self, task, messages, response_schema):
        return self.response


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

    with pytest.raises(ValueError, match="ineligible"):
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
