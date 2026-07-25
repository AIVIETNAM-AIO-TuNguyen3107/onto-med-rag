from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clinical_nlp.assertion_detection import AssertionDetector
from clinical_nlp.config import PipelineConfig
from clinical_nlp.entity_finding import RuleEntityFinder, merge_proposals
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.llm.base import LLMBackend, LLMTask, NoopLLMBackend
from clinical_nlp.ner.base import NERBackend
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.rxnorm_linking.index import query_variants
from clinical_nlp.schemas import (
    Assertion,
    Chunk,
    Document,
    Entity,
    EntityType,
    LinkCandidate,
    SpanProposal,
)
from clinical_nlp.text import chunk_document, find_occurrence
from clinical_nlp.validation import validate_entities


RECOVERY_MAX_NEW_TOKENS = 1536
REVIEW_MAX_NEW_TOKENS = 2048
RERANK_MAX_NEW_TOKENS = 1536
ENTITY_REVIEW_BATCH_SIZE = 20
TERMINOLOGY_RERANK_BATCH_SIZE = 10


class RecoveredEntity(BaseModel):
    text: str = Field(min_length=1)
    occurrence: int = Field(default=1, ge=1)
    type: str


class EntityRecoveryResponse(BaseModel):
    entities: list[RecoveredEntity] = Field(default_factory=list)


class RankedCandidatesResponse(BaseModel):
    candidates: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ReviewedEntity(BaseModel):
    position: tuple[int, int]
    keep: bool
    type: EntityType
    assertions: list[Assertion] = Field(default_factory=list)


class EntityReviewResponse(BaseModel):
    entities: list[ReviewedEntity] = Field(default_factory=list)


class CandidateSelection(BaseModel):
    position: tuple[int, int]
    candidates: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class BatchCandidateSelectionResponse(BaseModel):
    selections: list[CandidateSelection] = Field(default_factory=list)


class OnlinePreflightResponse(BaseModel):
    status: str
    sum: int


class DocumentArtifacts(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    chunks: list[dict[str, Any]]
    rule_proposals: list[dict[str, Any]]
    ner_proposals: list[dict[str, Any]]
    llm_proposals: list[dict[str, Any]]
    llm_recovery_audit: list[dict[str, Any]]
    merged_entities: list[dict[str, Any]]
    llm_reviews: list[dict[str, Any]]
    assertions: list[dict[str, Any]]
    icd_candidates: list[dict[str, Any]]
    rxnorm_candidates: list[dict[str, Any]]
    reranked_entities: list[dict[str, Any]]
    model_metadata: dict[str, Any] = Field(default_factory=dict)
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
        llm_recovery_audit: list[dict[str, Any]] = []
        if not isinstance(self.llm_backend, NoopLLMBackend):
            try:
                llm_proposals, llm_recovery_audit = self._recover_entities(
                    document,
                    chunks,
                    rule_proposals + ner_proposals,
                    warnings,
                )
            except Exception as exc:
                if self.config.run.fail_on_model_unavailable:
                    raise
                warnings.append(f"LLM recovery unavailable: {type(exc).__name__}: {exc}")

        merged = merge_proposals(rule_proposals + ner_proposals + llm_proposals)
        assertion_by_position: dict[tuple[int, int], list[str]] = {}
        for proposal in merged:
            labels = self.assertions.detect(document.text, proposal)
            assertion_by_position[(proposal.start, proposal.end)] = [
                value.value for value in labels
            ]

        llm_reviews: list[dict[str, Any]] = []
        if self.config.run.llm_full_review:
            merged, assertion_by_position, llm_reviews = self._review_entities(
                document,
                merged,
                assertion_by_position,
            )

        icd_artifacts: list[dict[str, Any]] = []
        rxnorm_artifacts: list[dict[str, Any]] = []
        icd_entries: list[tuple[SpanProposal, list[LinkCandidate]]] = []
        rxnorm_entries: list[tuple[SpanProposal, list[LinkCandidate]]] = []
        for proposal in merged:
            if proposal.type == EntityType.DIAGNOSIS:
                candidates = self.icd_index.retrieve(
                    proposal.text,
                    limit=self.config.linking.retrieval_candidates,
                )
                icd_entries.append((proposal, candidates))
            elif proposal.type == EntityType.MEDICATION:
                candidates = self._retrieve_rxnorm(proposal.text, warnings)
                rxnorm_entries.append((proposal, candidates))

        if self.config.run.llm_full_review:
            selected_icd = self._batch_rerank(
                LLMTask.ICD_RERANK,
                document,
                icd_entries,
                self.config.linking.icd_max_candidates,
            )
            selected_rxnorm = self._batch_rerank(
                LLMTask.RXNORM_RERANK,
                document,
                rxnorm_entries,
                self.config.linking.rxnorm_max_candidates,
            )
        else:
            selected_icd = {
                (proposal.start, proposal.end): self._rerank_if_needed(
                    LLMTask.ICD_RERANK,
                    proposal,
                    candidates,
                    self.config.linking.icd_max_candidates,
                    warnings,
                )
                for proposal, candidates in icd_entries
            }
            selected_rxnorm = {
                (proposal.start, proposal.end): self._rerank_if_needed(
                    LLMTask.RXNORM_RERANK,
                    proposal,
                    candidates,
                    self.config.linking.rxnorm_max_candidates,
                    warnings,
                )
                for proposal, candidates in rxnorm_entries
            }

        for proposal, candidates in icd_entries:
            selected = selected_icd[(proposal.start, proposal.end)]
            icd_artifacts.append(
                self._candidate_artifact(proposal, candidates, selected)
            )
        for proposal, candidates in rxnorm_entries:
            selected = selected_rxnorm[(proposal.start, proposal.end)]
            rxnorm_artifacts.append(
                self._candidate_artifact(proposal, candidates, selected)
            )

        entities: list[Entity] = []
        for proposal in merged:
            position = (proposal.start, proposal.end)
            candidate_ids: list[str] | None = None
            if proposal.type == EntityType.DIAGNOSIS:
                candidate_ids = [
                    row.identifier for row in selected_icd.get(position, [])
                ]
            elif proposal.type == EntityType.MEDICATION:
                candidate_ids = [
                    row.identifier for row in selected_rxnorm.get(position, [])
                ]
            entities.append(
                Entity(
                    text=proposal.text,
                    type=proposal.type,
                    candidates=candidate_ids,
                    assertions=assertion_by_position[position],
                    position=position,
                )
            )
        validate_entities(document, entities)
        assertion_rows = [
            {
                "position": [proposal.start, proposal.end],
                "assertions": assertion_by_position[(proposal.start, proposal.end)],
            }
            for proposal in merged
        ]
        artifacts = DocumentArtifacts(
            chunks=[row.model_dump(mode="json") for row in chunks],
            rule_proposals=[
                row.model_dump(mode="json") for row in rule_proposals
            ],
            ner_proposals=[row.model_dump(mode="json") for row in ner_proposals],
            llm_proposals=[row.model_dump(mode="json") for row in llm_proposals],
            llm_recovery_audit=llm_recovery_audit,
            merged_entities=[row.model_dump(mode="json") for row in merged],
            llm_reviews=llm_reviews,
            assertions=assertion_rows,
            icd_candidates=icd_artifacts,
            rxnorm_candidates=rxnorm_artifacts,
            reranked_entities=[row.output_dict() for row in entities],
            model_metadata=self.model_metadata(),
            warnings=warnings,
        )
        return entities, artifacts

    def model_metadata(self) -> dict[str, Any]:
        return {
            "ner": {
                "backend": getattr(self.ner_backend, "name", "unknown"),
                "model_id": getattr(self.ner_backend, "model_id", None),
            },
            "llm": {
                "backend": getattr(self.llm_backend, "name", "unknown"),
                "model_id": self.config.llm.model_id,
                "endpoint": self.config.llm.endpoint,
                "thinking": self.config.llm.thinking,
                "last_response": getattr(
                    self.llm_backend,
                    "last_response_metadata",
                    {},
                ),
            },
        }

    def online_preflight(self) -> dict[str, Any]:
        if getattr(self.ner_backend, "name", None) == "noop":
            raise RuntimeError("NER preflight failed: noop backend is active")
        if isinstance(self.llm_backend, NoopLLMBackend):
            raise RuntimeError("LLM preflight failed: noop backend is active")
        response = self.llm_backend.generate_json(
            LLMTask.ENTITY_RECOVERY,
            [
                {
                    "role": "system",
                    "content": "Think carefully and return JSON only.",
                },
                {
                    "role": "user",
                    "content": 'Return exactly {"status":"ok","sum":4}.',
                },
            ],
            OnlinePreflightResponse,
            max_new_tokens=128,
        )
        if response.status != "ok" or response.sum != 4:
            raise RuntimeError("LLM preflight returned an unexpected response")
        llm_metadata = self.model_metadata()["llm"]
        response_model = str(
            llm_metadata.get("last_response", {}).get("response_model") or ""
        )
        requested_model = self.config.llm.model_id.split(":", 1)[0]
        requested_leaf = requested_model.rsplit("/", 1)[-1]
        if requested_leaf.casefold() not in response_model.casefold():
            raise RuntimeError(
                "LLM preflight response did not confirm the requested Qwen model: "
                f"{response_model!r}"
            )
        rxnorm = self.rxnorm_index.retrieve("amlodipine 10 mg po daily", limit=5)
        if not rxnorm:
            raise RuntimeError("RxNorm preflight returned no candidates")
        return {
            "status": "ok",
            "models": self.model_metadata(),
            "icd_concepts": len(self.icd_index.concepts),
            "icd_source_sha256": self.icd_index.source_sha256,
            "rxnorm_candidate_count": len(rxnorm),
            "rxnorm_candidate_ids": [row.identifier for row in rxnorm],
        }

    def _review_entities(
        self,
        document: Document,
        proposals: list[SpanProposal],
        initial_assertions: dict[tuple[int, int], list[str]],
    ) -> tuple[
        list[SpanProposal],
        dict[tuple[int, int], list[str]],
        list[dict[str, Any]],
    ]:
        if isinstance(self.llm_backend, NoopLLMBackend):
            raise RuntimeError("full LLM review requires an active LLM backend")
        decisions: dict[tuple[int, int], ReviewedEntity] = {}
        batch_by_position: dict[tuple[int, int], int] = {}
        for batch_index, start in enumerate(
            range(0, len(proposals), ENTITY_REVIEW_BATCH_SIZE)
        ):
            batch = proposals[start : start + ENTITY_REVIEW_BATCH_SIZE]
            supplied = [
                {
                    "position": [row.start, row.end],
                    "text": row.text,
                    "type": row.type.value,
                    "assertions": initial_assertions[(row.start, row.end)],
                }
                for row in batch
            ]
            expected = {(row.start, row.end) for row in batch}
            last_error: ValueError | None = None
            for _ in range(self.config.llm.max_retries + 1):
                response = self.llm_backend.generate_json(
                    LLMTask.ASSERTION_ADJUDICATION,
                    [
                        {
                            "role": "system",
                            "content": (
                                "Review supplied Vietnamese clinical entity spans. "
                                "Think carefully, then return JSON only. Positions "
                                "are immutable. Return exactly one decision for every "
                                "supplied position. Set keep=false for generic or "
                                "non-clinical false positives. Use only the five "
                                "allowed entity types. Assertions may only be "
                                "isNegated, isFamily, isHistorical and are permitted "
                                "only for symptoms, diagnoses, and medications. "
                                "Family assertions require contextual evidence, not "
                                "merely a kinship word."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"TEXT:\n<<<\n{document.text}\n>>>\n\n"
                                "ENTITIES:\n"
                                f"{json.dumps(supplied, ensure_ascii=False)}\n\n"
                                'Return {"entities":[{"position":[start,end],'
                                '"keep":true,"type":"ALLOWED_TYPE",'
                                '"assertions":[]}]}.'
                            ),
                        },
                    ],
                    EntityReviewResponse,
                    max_new_tokens=REVIEW_MAX_NEW_TOKENS,
                )
                returned = [tuple(row.position) for row in response.entities]
                returned_set = set(returned)
                invented = returned_set - expected
                if invented:
                    raise ValueError(
                        "LLM entity review invented or changed positions: "
                        f"{sorted(invented)}"
                    )
                if len(returned) != len(returned_set):
                    last_error = ValueError(
                        "LLM entity review returned duplicate positions"
                    )
                    continue
                if returned_set != expected:
                    last_error = ValueError(
                        "LLM entity review omitted required positions"
                    )
                    continue
                invalid_assertion = any(
                    row.type
                    not in {
                        EntityType.SYMPTOM,
                        EntityType.DIAGNOSIS,
                        EntityType.MEDICATION,
                    }
                    and row.assertions
                    for row in response.entities
                )
                if invalid_assertion:
                    last_error = ValueError(
                        "LLM assigned assertions to an ineligible entity type"
                    )
                    continue
                for row in response.entities:
                    position = tuple(row.position)
                    decisions[position] = row
                    batch_by_position[position] = batch_index
                break
            else:
                raise RuntimeError(
                    "LLM entity review remained incomplete or invalid after "
                    f"{self.config.llm.max_retries + 1} attempts"
                ) from last_error

        reviewed: list[SpanProposal] = []
        reviewed_assertions: dict[tuple[int, int], list[str]] = {}
        artifacts: list[dict[str, Any]] = []
        eligible = {
            EntityType.SYMPTOM,
            EntityType.DIAGNOSIS,
            EntityType.MEDICATION,
        }
        for proposal in proposals:
            position = (proposal.start, proposal.end)
            decision = decisions[position]
            if decision.type not in eligible and decision.assertions:
                raise ValueError("LLM assigned assertions to an ineligible entity type")
            artifacts.append(
                {
                    "position": list(position),
                    "text": proposal.text,
                    "initial_type": proposal.type.value,
                    "initial_assertions": initial_assertions[position],
                    "keep": decision.keep,
                    "reviewed_type": decision.type.value,
                    "reviewed_assertions": [
                        value.value for value in decision.assertions
                    ],
                    "batch_index": batch_by_position[position],
                }
            )
            if not decision.keep:
                continue
            evidence = dict(proposal.evidence)
            evidence["llm_reviewed"] = True
            reviewed.append(
                proposal.model_copy(
                    update={
                        "type": decision.type,
                        "evidence": evidence,
                    }
                )
            )
            reviewed_assertions[position] = [
                value.value for value in decision.assertions
            ]
        return reviewed, reviewed_assertions, artifacts

    def _batch_rerank(
        self,
        task: LLMTask,
        document: Document,
        entries: list[tuple[SpanProposal, list[LinkCandidate]]],
        limit: int,
    ) -> dict[tuple[int, int], list[LinkCandidate]]:
        selected: dict[tuple[int, int], list[LinkCandidate]] = {
            (proposal.start, proposal.end): []
            for proposal, _ in entries
        }
        with_candidates = [
            (proposal, candidates)
            for proposal, candidates in entries
            if candidates
        ]
        if not with_candidates:
            return selected
        if isinstance(self.llm_backend, NoopLLMBackend):
            raise RuntimeError("batch terminology reranking requires an active LLM")

        policy = (
            "For ICD-10, choose the most specific code explicitly supported by "
            "the mention and document. Hierarchy/status Z codes are valid only "
            "when the text expresses that factor rather than a disease."
            if task == LLMTask.ICD_RERANK
            else
            "For RxNorm, prefer generic SCD when ingredient, strength, and form "
            "are supported; choose SBD only for an explicit brand; use IN when "
            "the text supports only the ingredient. Never assume missing details."
        )
        candidates_by_position = {
            (proposal.start, proposal.end): {
                row.identifier: row for row in candidates
            }
            for proposal, candidates in with_candidates
        }
        for start in range(0, len(with_candidates), TERMINOLOGY_RERANK_BATCH_SIZE):
            batch = with_candidates[start : start + TERMINOLOGY_RERANK_BATCH_SIZE]
            payload = [
                {
                    "position": [proposal.start, proposal.end],
                    "mention": proposal.text,
                    "candidates": [
                        {
                            "id": row.identifier,
                            "name": row.name,
                            "type": row.terminology_type,
                            "score": row.score,
                            "metadata": row.metadata,
                        }
                        for row in candidates
                    ],
                }
                for proposal, candidates in batch
            ]
            expected = {
                (proposal.start, proposal.end) for proposal, _ in batch
            }
            last_error: ValueError | None = None
            for _ in range(self.config.llm.max_retries + 1):
                response = self.llm_backend.generate_json(
                    task,
                    [
                        {
                            "role": "system",
                            "content": (
                                "Rerank only supplied terminology candidates. Think "
                                "carefully, then return JSON only. Never invent or "
                                f"modify an identifier. Return at most {limit} IDs "
                                f"per position. {policy}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"TEXT:\n<<<\n{document.text}\n>>>\n\n"
                                "MENTIONS:\n"
                                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                                'Return {"selections":[{"position":[start,end],'
                                '"candidates":["ID"],"confidence":0.0}]}. Return '
                                "exactly one selection for every supplied position."
                            ),
                        },
                    ],
                    BatchCandidateSelectionResponse,
                    max_new_tokens=RERANK_MAX_NEW_TOKENS,
                )
                returned = [tuple(row.position) for row in response.selections]
                returned_set = set(returned)
                invented_positions = returned_set - expected
                if invented_positions:
                    raise ValueError(
                        "LLM terminology reranking invented or changed positions: "
                        f"{sorted(invented_positions)}"
                    )
                if len(returned) != len(returned_set):
                    last_error = ValueError(
                        "LLM terminology reranking returned duplicate positions"
                    )
                    continue
                if returned_set != expected:
                    last_error = ValueError(
                        "LLM terminology reranking omitted required positions"
                    )
                    continue
                invalid_decision = False
                for row in response.selections:
                    position = tuple(row.position)
                    if len(row.candidates) > limit or len(row.candidates) != len(
                        set(row.candidates)
                    ):
                        last_error = ValueError(
                            "LLM returned too many or duplicate candidate IDs"
                        )
                        invalid_decision = True
                        break
                    allowed = candidates_by_position[position]
                    if any(
                        identifier not in allowed for identifier in row.candidates
                    ):
                        raise ValueError("LLM invented a terminology candidate ID")
                if invalid_decision:
                    continue
                for row in response.selections:
                    position = tuple(row.position)
                    allowed = candidates_by_position[position]
                    selected[position] = [
                        allowed[identifier] for identifier in row.candidates
                    ]
                break
            else:
                raise RuntimeError(
                    "LLM terminology reranking remained incomplete after "
                    f"{self.config.llm.max_retries + 1} attempts"
                ) from last_error
        return selected

    def _recover_entities(
        self,
        document: Document,
        chunks: list[Chunk],
        existing: list[SpanProposal],
        warnings: list[str],
    ) -> tuple[list[SpanProposal], list[dict[str, Any]]]:
        proposals: list[SpanProposal] = []
        audit: list[dict[str, Any]] = []
        unsupported_types = 0
        missing_substrings = 0
        for chunk in chunks:
            chunk_existing = [
                {"text": row.text, "type": row.type.value}
                for row in existing
                if row.start >= chunk.start and row.end <= chunk.end
            ]
            response = self.llm_backend.generate_json(
                LLMTask.ENTITY_RECOVERY,
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract only explicit clinical entities. Copy text "
                            "exactly. Never generate offsets. Return missing entities "
                            "only. Think carefully, then return JSON only in the "
                            "final answer. Allowed types are TRIỆU_CHỨNG, "
                            "TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"TEXT_CHUNK:\n<<<\n{chunk.text}\n>>>\n\n"
                            "EXISTING_ENTITIES:\n"
                            f"{json.dumps(chunk_existing, ensure_ascii=False)}\n"
                            'Return {"entities":[{"text":"exact substring",'
                            '"occurrence":1,"type":"allowed type"}]}.'
                        ),
                    },
                ],
                EntityRecoveryResponse,
                max_new_tokens=RECOVERY_MAX_NEW_TOKENS,
            )
            for row in response.entities:
                type_value = row.type.strip()
                try:
                    entity_type = EntityType(type_value)
                except ValueError:
                    unsupported_types += 1
                    audit.append(
                        {
                            "chunk_index": chunk.index,
                            "text": row.text,
                            "type": row.type,
                            "occurrence": row.occurrence,
                            "status": "rejected",
                            "reason": "unsupported_entity_type",
                        }
                    )
                    continue
                try:
                    relative_start, relative_end = find_occurrence(
                        chunk.text,
                        row.text,
                        row.occurrence,
                    )
                except ValueError:
                    missing_substrings += 1
                    audit.append(
                        {
                            "chunk_index": chunk.index,
                            "text": row.text,
                            "type": type_value,
                            "occurrence": row.occurrence,
                            "status": "rejected",
                            "reason": "substring_not_found_in_chunk",
                        }
                    )
                    continue
                start = chunk.start + relative_start
                end = chunk.start + relative_end
                proposals.append(
                    SpanProposal(
                        start=start,
                        end=end,
                        text=document.text[start:end],
                        type=entity_type,
                        source="llm_recovery",
                        score=0.70,
                        evidence={"chunk_index": chunk.index},
                    )
                )
        if unsupported_types:
            warnings.append(
                "LLM recovery rejected "
                f"{unsupported_types} row(s) with unsupported entity types"
            )
        if missing_substrings:
            warnings.append(
                "LLM recovery rejected "
                f"{missing_substrings} row(s) not found exactly in their chunks"
            )
        return proposals, audit

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
                rows = self.rxnorm_index.retrieve(
                    variant,
                    limit=self.config.linking.retrieval_candidates,
                )
            except Exception as exc:
                if self.config.run.fail_on_model_unavailable:
                    raise
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
        )[: self.config.linking.retrieval_candidates]

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
                max_new_tokens=RERANK_MAX_NEW_TOKENS,
            )
        except Exception as exc:
            warnings.append(f"{task.value} failed: {type(exc).__name__}: {exc}")
            return candidates[:limit]
        selected = [allowed[value] for value in response.candidates if value in allowed]
        return selected[:limit] or candidates[:limit]

    @staticmethod
    def _candidate_artifact(
        proposal: SpanProposal,
        retrieved: list[LinkCandidate],
        selected: list[LinkCandidate],
    ) -> dict[str, Any]:
        return {
            "position": [proposal.start, proposal.end],
            "text": proposal.text,
            "retrieved_candidates": [
                row.model_dump(mode="json") for row in retrieved
            ],
            "selected_candidates": [
                row.model_dump(mode="json") for row in selected
            ],
        }
