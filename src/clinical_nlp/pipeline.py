from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from clinical_nlp.assertion_detection import AssertionDetector
from clinical_nlp.config import PipelineConfig
from clinical_nlp.entity_finding import RuleEntityFinder, merge_proposals
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.llm.base import LLMBackend, LLMTask, NoopLLMBackend
from clinical_nlp.ner.base import NERBackend
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.rxnorm_linking.index import query_variants
from clinical_nlp.schemas import (
    Document,
    Entity,
    EntityType,
    LinkCandidate,
    SpanProposal,
)
from clinical_nlp.text import chunk_document, find_occurrence
from clinical_nlp.validation import validate_entities


class RecoveredEntity(BaseModel):
    text: str
    occurrence: int = 1
    type: EntityType


class EntityRecoveryResponse(BaseModel):
    entities: list[RecoveredEntity] = Field(default_factory=list)


class RankedCandidatesResponse(BaseModel):
    candidates: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class DocumentArtifacts(BaseModel):
    chunks: list[dict[str, Any]]
    rule_proposals: list[dict[str, Any]]
    ner_proposals: list[dict[str, Any]]
    llm_proposals: list[dict[str, Any]]
    merged_entities: list[dict[str, Any]]
    assertions: list[dict[str, Any]]
    icd_candidates: list[dict[str, Any]]
    rxnorm_candidates: list[dict[str, Any]]
    reranked_entities: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


class ClinicalPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        icd_index: ICDIndex,
        rxnorm_index: RxNormIndex,
        ner_backend: NERBackend,
        llm_backend: LLMBackend,
    ) -> None:
        self.config = config
        self.icd_index = icd_index
        self.rxnorm_index = rxnorm_index
        self.ner_backend = ner_backend
        self.llm_backend = llm_backend
        self.rules = RuleEntityFinder(icd_index=icd_index)
        self.assertions = AssertionDetector()

    def process(self, document: Document) -> tuple[list[Entity], DocumentArtifacts]:
        warnings: list[str] = []
        chunks = chunk_document(
            document,
            max_chars=self.config.entity_finding.chunk_chars,
            overlap_chars=self.config.entity_finding.chunk_overlap_chars,
        )
        rule_proposals = self.rules.find(document.text)
        try:
            ner_proposals = self.ner_backend.predict(
                document,
                chunks,
                self.config.entity_finding.ner_threshold,
            )
        except Exception as exc:
            if self.config.run.fail_on_model_unavailable:
                raise
            warnings.append(f"NER unavailable: {type(exc).__name__}: {exc}")
            ner_proposals = []
        ner_proposals = self._filter_ner_proposals(ner_proposals)

        llm_proposals: list[SpanProposal] = []
        if not isinstance(self.llm_backend, NoopLLMBackend):
            try:
                llm_proposals = self._recover_entities(
                    document, rule_proposals + ner_proposals
                )
            except Exception as exc:
                if self.config.run.fail_on_model_unavailable:
                    raise
                warnings.append(f"LLM recovery unavailable: {type(exc).__name__}: {exc}")

        merged = merge_proposals(rule_proposals + ner_proposals + llm_proposals)
        assertion_rows: list[dict[str, Any]] = []
        for proposal in merged:
            labels = self.assertions.detect(document.text, proposal)
            assertion_rows.append(
                {
                    "position": [proposal.start, proposal.end],
                    "assertions": [value.value for value in labels],
                }
            )

        icd_artifacts: list[dict[str, Any]] = []
        rxnorm_artifacts: list[dict[str, Any]] = []
        entities: list[Entity] = []
        for proposal, assertion_row in zip(merged, assertion_rows):
            candidate_ids: list[str] | None = None
            if proposal.type == EntityType.DIAGNOSIS:
                candidates = self.icd_index.retrieve(proposal.text, limit=10)
                candidates = self._rerank_if_needed(
                    LLMTask.ICD_RERANK,
                    proposal,
                    candidates,
                    self.config.linking.icd_max_candidates,
                    warnings,
                )
                candidate_ids = [row.identifier for row in candidates]
                icd_artifacts.append(
                    self._candidate_artifact(proposal, candidates)
                )
            elif proposal.type == EntityType.MEDICATION:
                candidates = self._retrieve_rxnorm(proposal.text, warnings)
                candidates = self._rerank_if_needed(
                    LLMTask.RXNORM_RERANK,
                    proposal,
                    candidates,
                    self.config.linking.rxnorm_max_candidates,
                    warnings,
                )
                candidate_ids = [row.identifier for row in candidates]
                rxnorm_artifacts.append(
                    self._candidate_artifact(proposal, candidates)
                )
            entities.append(
                Entity(
                    text=proposal.text,
                    type=proposal.type,
                    candidates=candidate_ids,
                    assertions=assertion_row["assertions"],
                    position=(proposal.start, proposal.end),
                )
            )
        validate_entities(document, entities)
        artifacts = DocumentArtifacts(
            chunks=[row.model_dump(mode="json") for row in chunks],
            rule_proposals=[
                row.model_dump(mode="json") for row in rule_proposals
            ],
            ner_proposals=[row.model_dump(mode="json") for row in ner_proposals],
            llm_proposals=[row.model_dump(mode="json") for row in llm_proposals],
            merged_entities=[row.model_dump(mode="json") for row in merged],
            assertions=assertion_rows,
            icd_candidates=icd_artifacts,
            rxnorm_candidates=rxnorm_artifacts,
            reranked_entities=[row.output_dict() for row in entities],
            warnings=warnings,
        )
        return entities, artifacts

    def _recover_entities(
        self,
        document: Document,
        existing: list[SpanProposal],
    ) -> list[SpanProposal]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract only explicit clinical entities. Copy text exactly. "
                    "Never generate offsets. Return missing entities only. Think "
                    "carefully, then return JSON only in the final answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TEXT:\n<<<\n{document.text}\n>>>\n\n"
                    "EXISTING_ENTITIES:\n"
                    + str(
                        [
                            {"text": row.text, "type": row.type.value}
                            for row in existing
                        ]
                    )
                    + '\nReturn {"entities":[{"text":"exact substring",'
                    '"occurrence":1,"type":"allowed type"}]}.'
                ),
            },
        ]
        response = self.llm_backend.generate_json(
            LLMTask.ENTITY_RECOVERY,
            messages,
            EntityRecoveryResponse,
        )
        proposals: list[SpanProposal] = []
        for row in response.entities:
            try:
                start, end = find_occurrence(
                    document.text,
                    row.text,
                    row.occurrence,
                )
            except ValueError:
                continue
            proposals.append(
                SpanProposal(
                    start=start,
                    end=end,
                    text=row.text,
                    type=row.type,
                    source="llm_recovery",
                    score=0.70,
                )
            )
        return proposals

    @staticmethod
    def _filter_ner_proposals(
        proposals: list[SpanProposal],
    ) -> list[SpanProposal]:
        generic = {
            "dấu hiệu",
            "triệu chứng",
            "xét nghiệm",
            "xét nghiệm máu",
            "sàng lọc",
            "bệnh bẩm sinh",
            "nhiễm sắc thể x",
            "g6pd",
        }
        non_medications = {
            "băng phiến",
            "long não",
            "thuốc",
            "thuốc nam",
            "thuốc đông y",
        }
        kept: list[SpanProposal] = []
        for row in proposals:
            normalized = row.text.casefold().strip()
            if "\n" in row.text or normalized in generic:
                continue
            if row.type == EntityType.MEDICATION and normalized in non_medications:
                continue
            if (
                row.type == EntityType.TEST_NAME
                and "glucose-6-phosphate dehydrogenase" in normalized
            ):
                continue
            kept.append(row)
        return kept

    def _retrieve_rxnorm(
        self,
        mention: str,
        warnings: list[str],
    ) -> list[LinkCandidate]:
        merged: dict[str, LinkCandidate] = {}
        for variant in query_variants(mention):
            try:
                rows = self.rxnorm_index.retrieve(variant, limit=10)
            except Exception as exc:
                warnings.append(
                    f"RxNorm retrieval failed for {variant!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            for row in rows:
                old = merged.get(row.identifier)
                if old is None or row.score > old.score:
                    merged[row.identifier] = row
        return sorted(
            merged.values(),
            key=lambda row: (-row.score, row.identifier),
        )[:10]

    def _rerank_if_needed(
        self,
        task: LLMTask,
        proposal: SpanProposal,
        candidates: list[LinkCandidate],
        limit: int,
        warnings: list[str],
    ) -> list[LinkCandidate]:
        if not candidates:
            return []
        if candidates[0].score >= 0.92 and (
            len(candidates) == 1
            or candidates[0].score - candidates[1].score >= 0.12
        ):
            return candidates[:1]
        if isinstance(self.llm_backend, NoopLLMBackend):
            return candidates[:limit]
        allowed = {row.identifier: row for row in candidates}
        messages = [
            {
                "role": "system",
                "content": (
                    "Rank only supplied terminology candidates. Never invent or "
                    "modify an identifier. Think carefully and return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"MENTION: {proposal.text!r}\nCANDIDATES:\n"
                    + "\n".join(
                        f"{row.identifier} — {row.name} ({row.terminology_type})"
                        for row in candidates
                    )
                    + f'\nReturn at most {limit}: '
                    '{"candidates":["ID"],"confidence":0.0}'
                ),
            },
        ]
        try:
            response = self.llm_backend.generate_json(
                task,
                messages,
                RankedCandidatesResponse,
            )
        except Exception as exc:
            warnings.append(f"{task.value} failed: {type(exc).__name__}: {exc}")
            return candidates[:limit]
        selected = [allowed[value] for value in response.candidates if value in allowed]
        return selected[:limit] or candidates[:limit]

    @staticmethod
    def _candidate_artifact(
        proposal: SpanProposal,
        candidates: list[LinkCandidate],
    ) -> dict[str, Any]:
        return {
            "position": [proposal.start, proposal.end],
            "text": proposal.text,
            "candidates": [row.model_dump(mode="json") for row in candidates],
        }
