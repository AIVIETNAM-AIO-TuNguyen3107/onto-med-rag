from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clinical_nlp.assertion_detection import AssertionDetector
from clinical_nlp.config import PipelineConfig
from clinical_nlp.entity_finding import RuleEntityFinder, merge_proposals
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.llm.base import (
    LLMBackend,
    LLMDecisionError,
    LLMTask,
    NoopLLMBackend,
)
from clinical_nlp.ner.base import NERBackend
from clinical_nlp.normalization import normalize_search
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


RECOVERY_MAX_NEW_TOKENS = 4096
PREFLIGHT_MAX_NEW_TOKENS = 512
REVIEW_MAX_NEW_TOKENS = 2048
RERANK_MAX_NEW_TOKENS = 1536
FALLBACK_MAX_NEW_TOKENS = 2048
ENTITY_REVIEW_BATCH_SIZE = 10
TERMINOLOGY_RERANK_BATCH_SIZE = 10
POST_MERGE_NON_MEDICATIONS = {
    "băng phiến",
    "long não",
    "thuốc",
    "thuốc nam",
    "thuốc đông y",
}
SOURCE_INDEPENDENT_NON_ENTITIES = {
    "dấu hiệu",
    "triệu chứng",
    "nhiễm sắc thể x",
    "xq28",
    "máu khô",
    "đậu tằm",
    "nhận xét",
    "hiến máu",
}
QUALITATIVE_RESULT_RE = re.compile(r"(?i)^(?:âm\s+tính|dương\s+tính)$")
_RESULT_UNIT = (
    r"%|°\s*[CF]|g/L|mg/L|mg/dL|mmol/L|µmol/L|umol/L|U/L|IU/L|"
    r"10\^?\d+/L|x10\^?\d+/L|mmHg|bpm|lần/phút|nhịp/phút|kg|cm|mm|ml|cc"
)
NUMERIC_RESULT_RE = re.compile(
    r"(?ix)^(?:[<>]=?\s*)?[+-]?\d+(?:[.,]\d+)?"
    r"(?:\s*[-–/]\s*[+-]?\d+(?:[.,]\d+)?)?"
    rf"(?:\s*(?:{_RESULT_UNIT}))?$"
)


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
    type: str
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
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
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
        self._retrieval_lock = threading.Lock()
        self._retrieval_cache: dict[
            tuple[str, str],
            list[LinkCandidate],
        ] = {}

    def process(
        self,
        document: Document,
        *,
        checkpoint_dir: Path | None = None,
    ) -> tuple[list[Entity], DocumentArtifacts]:
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
                    checkpoint_dir,
                )
            except Exception as exc:
                if self.config.run.fail_on_model_unavailable:
                    raise
                warnings.append(f"LLM recovery unavailable: {type(exc).__name__}: {exc}")

        merged = merge_proposals(rule_proposals + ner_proposals + llm_proposals)
        merged, filter_decisions = self._filter_merged_proposals(merged)
        assertion_by_position: dict[tuple[int, int], list[str]] = {}
        for proposal in merged:
            labels = self.assertions.detect(document.text, proposal)
            assertion_by_position[(proposal.start, proposal.end)] = [
                value.value for value in labels
            ]

        llm_reviews: list[dict[str, Any]] = list(filter_decisions)
        review_mode = self.config.run.llm_review_mode or "off"
        if review_mode != "off":
            merged, assertion_by_position, llm_reviews = self._review_entities(
                document,
                merged,
                assertion_by_position,
                warnings,
                checkpoint_dir,
                review_mode,
                prefix_artifacts=filter_decisions,
            )
        else:
            llm_reviews.extend(
                self._automatic_review_artifact(
                    proposal,
                    assertion_by_position[(proposal.start, proposal.end)],
                    reason="review_mode_off",
                )
                for proposal in merged
            )

        icd_artifacts: list[dict[str, Any]] = []
        rxnorm_artifacts: list[dict[str, Any]] = []
        icd_entries: list[tuple[SpanProposal, list[LinkCandidate]]] = []
        rxnorm_entries: list[tuple[SpanProposal, list[LinkCandidate]]] = []
        for proposal in merged:
            if proposal.type == EntityType.DIAGNOSIS:
                candidates = self._retrieve_icd(proposal.text)
                icd_entries.append((proposal, candidates))
            elif proposal.type == EntityType.MEDICATION:
                candidates = self._retrieve_rxnorm(proposal.text, warnings)
                rxnorm_entries.append((proposal, candidates))

        terminology_decisions: dict[tuple[int, int], dict[str, Any]] = {}
        if review_mode == "selective":
            selected_icd, icd_decisions = self._select_terminology(
                LLMTask.ICD_RERANK,
                document,
                icd_entries,
                self.config.linking.icd_max_candidates,
                checkpoint_dir,
            )
            selected_rxnorm, rxnorm_decisions = self._select_terminology(
                LLMTask.RXNORM_RERANK,
                document,
                rxnorm_entries,
                self.config.linking.rxnorm_max_candidates,
                checkpoint_dir,
            )
            terminology_decisions.update(icd_decisions)
            terminology_decisions.update(rxnorm_decisions)
        elif review_mode == "full":
            selected_icd = self._batch_rerank(
                LLMTask.ICD_RERANK,
                document,
                icd_entries,
                self.config.linking.icd_max_candidates,
                checkpoint_dir,
            )
            selected_rxnorm = self._batch_rerank(
                LLMTask.RXNORM_RERANK,
                document,
                rxnorm_entries,
                self.config.linking.rxnorm_max_candidates,
                checkpoint_dir,
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
                self._candidate_artifact(
                    proposal,
                    candidates,
                    self._eligible_candidates(LLMTask.ICD_RERANK, candidates),
                    selected,
                    terminology_decisions.get((proposal.start, proposal.end)),
                )
            )
        for proposal, candidates in rxnorm_entries:
            selected = selected_rxnorm[(proposal.start, proposal.end)]
            rxnorm_artifacts.append(
                self._candidate_artifact(
                    proposal,
                    candidates,
                    self._eligible_candidates(LLMTask.RXNORM_RERANK, candidates),
                    selected,
                    terminology_decisions.get((proposal.start, proposal.end)),
                )
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
            llm_calls=self._document_call_audits(document.id, checkpoint_dir),
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
                "review_mode": self.config.run.llm_review_mode,
                "task_reasoning": {
                    "entity_recovery": False,
                    "entity_review": self.config.llm.reasoning_enabled,
                    "terminology_rerank": self.config.llm.reasoning_enabled,
                },
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
            max_new_tokens=PREFLIGHT_MAX_NEW_TOKENS,
            reasoning_enabled=False,
            call_id="preflight/model-json",
            cache_enabled=False,
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
        warnings: list[str] | None = None,
        checkpoint_dir: Path | None = None,
        review_mode: str = "full",
        *,
        prefix_artifacts: list[dict[str, Any]] | None = None,
    ) -> tuple[
        list[SpanProposal],
        dict[tuple[int, int], list[str]],
        list[dict[str, Any]],
    ]:
        warnings = warnings if warnings is not None else []
        if isinstance(self.llm_backend, NoopLLMBackend):
            raise RuntimeError("LLM review requires an active LLM backend")
        review_reasons: dict[tuple[int, int], list[str]] = {}
        review_proposals: list[SpanProposal] = []
        for proposal in proposals:
            position = (proposal.start, proposal.end)
            reasons = (
                ["full_review"]
                if review_mode == "full"
                else self._selective_review_reasons(
                    document,
                    proposal,
                    initial_assertions[position],
                    warnings,
                )
            )
            review_reasons[position] = reasons
            if reasons:
                review_proposals.append(proposal)

        decisions: dict[tuple[int, int], ReviewedEntity] = {}
        batch_by_position: dict[tuple[int, int], int] = {}
        fallback_by_position: dict[tuple[int, int], str] = {}
        batches = [
            review_proposals[start : start + ENTITY_REVIEW_BATCH_SIZE]
            for start in range(0, len(review_proposals), ENTITY_REVIEW_BATCH_SIZE)
        ]
        jobs = list(enumerate(batches))

        def review_job(
            job: tuple[int, list[SpanProposal]],
        ) -> tuple[int, list[ReviewedEntity], str | None]:
            batch_index, batch = job
            try:
                response = self._review_batch(
                    document,
                    batch,
                    initial_assertions,
                    batch_index,
                    checkpoint_dir,
                    reasoning_enabled=True,
                    call_suffix="",
                )
            except LLMDecisionError as exc:
                response = None
                error = f"structured response failure: {exc}"
            else:
                error = self._review_response_error(batch, response)
            if error is not None and self.config.llm.decision_retries:
                try:
                    response = self._review_batch(
                        document,
                        batch,
                        initial_assertions,
                        batch_index,
                        checkpoint_dir,
                        reasoning_enabled=False,
                        call_suffix="-decision-fallback",
                    )
                except LLMDecisionError as exc:
                    response = None
                    error = f"structured response failure: {exc}"
                else:
                    error = self._review_response_error(batch, response)
            if error is not None:
                preserved = [
                    ReviewedEntity(
                        position=(proposal.start, proposal.end),
                        keep=not (
                            proposal.type == EntityType.TEST_RESULT
                            and not self._valid_laboratory_result_span(
                                proposal.text
                            )
                        ),
                        type=proposal.type.value,
                        assertions=[
                            Assertion(value)
                            for value in initial_assertions[
                                (proposal.start, proposal.end)
                            ]
                        ],
                    )
                    for proposal in batch
                ]
                return batch_index, preserved, error
            return batch_index, response.entities, None

        for batch_index, rows, fallback_error in self._parallel_map(
            jobs,
            review_job,
        ):
            if fallback_error is not None:
                warnings.append(
                    "LLM entity review batch "
                    f"{batch_index} invalid after decision fallback; "
                    f"applied deterministic safe defaults: {fallback_error}"
                )
            for row in rows:
                position = tuple(row.position)
                decisions[position] = row
                batch_by_position[position] = batch_index
                if fallback_error is not None:
                    fallback_by_position[position] = fallback_error

        reviewed: list[SpanProposal] = []
        reviewed_assertions: dict[tuple[int, int], list[str]] = {}
        artifacts: list[dict[str, Any]] = list(prefix_artifacts or [])
        unsupported_rejected_types = 0
        eligible = {
            EntityType.SYMPTOM,
            EntityType.DIAGNOSIS,
            EntityType.MEDICATION,
        }
        for proposal in proposals:
            position = (proposal.start, proposal.end)
            decision = decisions.get(position)
            if decision is None:
                reviewed.append(proposal)
                reviewed_assertions[position] = initial_assertions[position]
                artifacts.append(
                    self._automatic_review_artifact(
                        proposal,
                        initial_assertions[position],
                        reason="high_confidence",
                    )
                )
                continue
            try:
                reviewed_type = EntityType(decision.type)
                unsupported_returned_type = False
            except ValueError:
                if decision.keep:
                    raise ValueError(
                        "LLM kept an entity with an unsupported entity type"
                    )
                reviewed_type = proposal.type
                unsupported_returned_type = True
                unsupported_rejected_types += 1
            if reviewed_type not in eligible and decision.assertions:
                raise ValueError("LLM assigned assertions to an ineligible entity type")
            artifacts.append(
                {
                    "position": list(position),
                    "text": proposal.text,
                    "initial_type": proposal.type.value,
                    "initial_assertions": initial_assertions[position],
                    "keep": decision.keep,
                    "returned_type": decision.type,
                    "reviewed_type": reviewed_type.value,
                    "unsupported_returned_type": unsupported_returned_type,
                    "reviewed_assertions": [
                        value.value for value in decision.assertions
                    ],
                    "batch_index": batch_by_position[position],
                    "decision_source": (
                        "deterministic_fallback"
                        if position in fallback_by_position
                        else "llm"
                    ),
                    "decision_error": fallback_by_position.get(position),
                    "escalation_reasons": review_reasons[position],
                }
            )
            if not decision.keep:
                continue
            if position in fallback_by_position:
                reviewed.append(proposal)
                reviewed_assertions[position] = initial_assertions[position]
                continue
            evidence = dict(proposal.evidence)
            evidence["llm_reviewed"] = True
            reviewed.append(
                proposal.model_copy(
                    update={
                        "type": reviewed_type,
                        "evidence": evidence,
                    }
                )
            )
            reviewed_assertions[position] = [
                value.value for value in decision.assertions
            ]
        if unsupported_rejected_types:
            warnings.append(
                "LLM review rejected "
                f"{unsupported_rejected_types} row(s) using unsupported type labels"
            )
        return reviewed, reviewed_assertions, artifacts

    def _review_batch(
        self,
        document: Document,
        batch: list[SpanProposal],
        initial_assertions: dict[tuple[int, int], list[str]],
        batch_index: int,
        checkpoint_dir: Path | None,
        *,
        reasoning_enabled: bool,
        call_suffix: str,
    ) -> EntityReviewResponse:
        supplied = [
            {
                "position": [row.start, row.end],
                "text": row.text,
                "type": row.type.value,
                "assertions": initial_assertions[(row.start, row.end)],
                "section": self.assertions.section_at(document.text, row.start),
                **self._local_context(document, row.start, row.end),
            }
            for row in batch
        ]
        return self.llm_backend.generate_json(
            LLMTask.ASSERTION_ADJUDICATION,
            [
                {
                    "role": "system",
                    "content": (
                        "Review supplied Vietnamese clinical entity spans. "
                        "Think carefully, then return JSON only. Positions are "
                        "immutable. Return exactly one decision for every supplied "
                        "position. Prefer keep=true for any explicit, plausible "
                        "mention of one of the five allowed clinical entity types. "
                        "Set keep=false only for clearly generic, non-clinical, or "
                        "unsupported spans. Preserve the supplied type unless the "
                        "local context strongly supports a different allowed type; "
                        "never retype speculatively. Use only the five allowed "
                        "entity types. "
                        "Assertions may only be isNegated, isFamily, isHistorical "
                        "and are permitted only for symptoms, diagnoses, and "
                        "medications. Family assertions require contextual evidence, "
                        "not merely a kinship word."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "ENTITIES_WITH_LOCAL_CONTEXT:\n"
                        f"{json.dumps(supplied, ensure_ascii=False)}\n\n"
                        'Return {"entities":[{"position":[start,end],'
                        '"keep":true,"type":"ALLOWED_TYPE","assertions":[]}]}.'
                    ),
                },
            ],
            EntityReviewResponse,
            max_new_tokens=(
                REVIEW_MAX_NEW_TOKENS
                if reasoning_enabled
                else FALLBACK_MAX_NEW_TOKENS
            ),
            reasoning_enabled=reasoning_enabled,
            call_id=(
                f"{document.id}/review-{batch_index:03d}{call_suffix}"
            ),
            checkpoint_dir=checkpoint_dir,
        )

    @staticmethod
    def _review_response_error(
        batch: list[SpanProposal],
        response: EntityReviewResponse,
    ) -> str | None:
        expected = {(row.start, row.end) for row in batch}
        returned = [tuple(row.position) for row in response.entities]
        returned_set = set(returned)
        invented = returned_set - expected
        if invented:
            return (
                "invented or changed positions: "
                f"{sorted(invented)}"
            )
        if len(returned) != len(returned_set):
            return "duplicate positions"
        if returned_set != expected:
            return "omitted required positions"
        assertion_eligible = {
            EntityType.SYMPTOM,
            EntityType.DIAGNOSIS,
            EntityType.MEDICATION,
        }
        for row in response.entities:
            try:
                entity_type = EntityType(row.type)
            except ValueError:
                if row.keep:
                    return "unsupported entity type on a kept entity"
                if row.assertions:
                    return "assertions assigned to a rejected unsupported type"
                continue
            if entity_type not in assertion_eligible and row.assertions:
                return "assertions assigned to an ineligible entity type"
        return None

    def _batch_rerank(
        self,
        task: LLMTask,
        document: Document,
        entries: list[tuple[SpanProposal, list[LinkCandidate]]],
        limit: int,
        checkpoint_dir: Path | None = None,
    ) -> dict[tuple[int, int], list[LinkCandidate]]:
        task = LLMTask(task)
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

        candidates_by_position = {
            (proposal.start, proposal.end): {
                row.identifier: row for row in candidates
            }
            for proposal, candidates in with_candidates
        }
        batches = [
            with_candidates[start : start + TERMINOLOGY_RERANK_BATCH_SIZE]
            for start in range(
                0,
                len(with_candidates),
                TERMINOLOGY_RERANK_BATCH_SIZE,
            )
        ]

        def rerank_job(
            job: tuple[int, list[tuple[SpanProposal, list[LinkCandidate]]]],
        ) -> tuple[int, list[CandidateSelection]]:
            batch_index, batch = job
            try:
                response = self._rerank_batch(
                    task,
                    document,
                    batch,
                    limit,
                    batch_index,
                    checkpoint_dir,
                    reasoning_enabled=True,
                    call_suffix="",
                )
            except LLMDecisionError as exc:
                response = None
                error = f"structured response failure: {exc}"
            else:
                error = self._rerank_response_error(batch, response, limit)
            if error is not None and self.config.llm.decision_retries:
                try:
                    response = self._rerank_batch(
                        task,
                        document,
                        batch,
                        limit,
                        batch_index,
                        checkpoint_dir,
                        reasoning_enabled=False,
                        call_suffix="-decision-fallback",
                    )
                except LLMDecisionError as exc:
                    response = None
                    error = f"structured response failure: {exc}"
                else:
                    error = self._rerank_response_error(batch, response, limit)
            if error is not None:
                return (
                    batch_index,
                    [
                        CandidateSelection(
                            position=(proposal.start, proposal.end),
                            candidates=[],
                            confidence=0.0,
                        )
                        for proposal, _ in batch
                    ],
                )
            return batch_index, response.selections

        for _, rows in self._parallel_map(list(enumerate(batches)), rerank_job):
            for row in rows:
                position = tuple(row.position)
                allowed = candidates_by_position[position]
                selected[position] = [
                    allowed[identifier] for identifier in row.candidates
                ]
        return selected

    def _rerank_batch(
        self,
        task: LLMTask,
        document: Document,
        batch: list[tuple[SpanProposal, list[LinkCandidate]]],
        limit: int,
        batch_index: int,
        checkpoint_dir: Path | None,
        *,
        reasoning_enabled: bool,
        call_suffix: str,
    ) -> BatchCandidateSelectionResponse:
        policy = (
            "For ICD-10, choose the most specific code explicitly supported by "
            "the mention and local context. Status Z codes are valid only when "
            "the context expresses that factor rather than a disease."
            if task == LLMTask.ICD_RERANK
            else
            "For RxNorm, prefer generic SCD when ingredient, strength, and form "
            "are explicit; choose SBD only for an explicit brand; use IN when "
            "only the ingredient is supported. Never assume missing details."
        )
        payload = [
            {
                "position": [proposal.start, proposal.end],
                "mention": proposal.text,
                "section": self.assertions.section_at(document.text, proposal.start),
                **self._local_context(document, proposal.start, proposal.end),
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
        return self.llm_backend.generate_json(
            task,
            [
                {
                    "role": "system",
                    "content": (
                        "Rerank only supplied terminology candidates. Think "
                        "carefully, then return JSON only. Never invent or modify "
                        f"an identifier. Return at most {limit} IDs per position. "
                        "Every returned ID must appear in the candidates array for "
                        "that same position. If the clinically best ID is absent, "
                        "return an empty candidates list for that position. "
                        f"{policy}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "MENTIONS_WITH_LOCAL_CONTEXT:\n"
                        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                        'Return {"selections":[{"position":[start,end],'
                        '"candidates":["ID"],"confidence":0.0}]}. Return exactly '
                        "one selection for every supplied position."
                    ),
                },
            ],
            BatchCandidateSelectionResponse,
            max_new_tokens=(
                RERANK_MAX_NEW_TOKENS
                if reasoning_enabled
                else FALLBACK_MAX_NEW_TOKENS
            ),
            reasoning_enabled=reasoning_enabled,
            call_id=(
                f"{document.id}/{task.value}-{batch_index:03d}{call_suffix}"
            ),
            checkpoint_dir=checkpoint_dir,
        )

    @staticmethod
    def _rerank_response_error(
        batch: list[tuple[SpanProposal, list[LinkCandidate]]],
        response: BatchCandidateSelectionResponse,
        limit: int,
    ) -> str | None:
        expected = {
            (proposal.start, proposal.end): {
                row.identifier for row in candidates
            }
            for proposal, candidates in batch
        }
        returned = [tuple(row.position) for row in response.selections]
        returned_set = set(returned)
        invented = returned_set - set(expected)
        if invented:
            return (
                "invented or changed positions: "
                f"{sorted(invented)}"
            )
        if len(returned) != len(returned_set):
            return "duplicate positions"
        if returned_set != set(expected):
            return "omitted required positions"
        for row in response.selections:
            if (
                len(row.candidates) > limit
                or len(row.candidates) != len(set(row.candidates))
            ):
                return "too many or duplicate candidate IDs"
            if any(
                identifier not in expected[tuple(row.position)]
                for identifier in row.candidates
            ):
                return "invented a terminology candidate ID"
        return None

    def _recover_entities(
        self,
        document: Document,
        chunks: list[Chunk],
        existing: list[SpanProposal],
        warnings: list[str],
        checkpoint_dir: Path | None = None,
    ) -> tuple[list[SpanProposal], list[dict[str, Any]]]:
        proposals: list[SpanProposal] = []
        audit: list[dict[str, Any]] = []
        unsupported_types = 0
        corrected_types = 0
        missing_substrings = 0
        quality_retries = 0
        quality_rejections = 0

        def recovery_job(
            chunk: Chunk,
        ) -> tuple[Chunk, EntityRecoveryResponse, dict[str, Any] | None]:
            chunk_existing_by_key: dict[
                tuple[str, int, str],
                dict[str, str | int],
            ] = {}
            for row in existing:
                if row.start < chunk.start or row.end > chunk.end:
                    continue
                occurrence = self._occurrence_at(
                    chunk.text,
                    row.text,
                    row.start - chunk.start,
                )
                key = (row.text, occurrence, row.type.value)
                chunk_existing_by_key[key] = {
                    "text": row.text,
                    "occurrence": occurrence,
                    "type": row.type.value,
                }
            chunk_existing = [
                chunk_existing_by_key[key]
                for key in sorted(
                    chunk_existing_by_key,
                    key=lambda value: (value[1], value[0], value[2]),
                )
            ]
            try:
                response = self._recover_chunk(
                    document,
                    chunk,
                    chunk_existing,
                    checkpoint_dir,
                    strict=False,
                )
            except LLMDecisionError as exc:
                return (
                    chunk,
                    EntityRecoveryResponse(entities=[]),
                    {
                        "chunk_index": chunk.index,
                        "status": "rejected",
                        "reason": "invalid_structured_response",
                        "detail": str(exc),
                        "initial_row_count": 0,
                        "retry_row_count": 0,
                    },
                )
            quality_error = self._recovery_quality_error(response)
            if quality_error is None:
                return chunk, response, None
            try:
                retry = self._recover_chunk(
                    document,
                    chunk,
                    chunk_existing,
                    checkpoint_dir,
                    strict=True,
                )
            except LLMDecisionError as exc:
                return (
                    chunk,
                    EntityRecoveryResponse(entities=[]),
                    {
                        "chunk_index": chunk.index,
                        "status": "rejected",
                        "reason": "invalid_structured_response",
                        "detail": str(exc),
                        "initial_row_count": len(response.entities),
                        "retry_row_count": 0,
                    },
                )
            retry_error = self._recovery_quality_error(retry)
            if retry_error is None:
                return (
                    chunk,
                    retry,
                    {
                        "chunk_index": chunk.index,
                        "status": "retried",
                        "reason": "suspicious_recovery_response",
                        "detail": quality_error,
                        "initial_row_count": len(response.entities),
                        "retry_row_count": len(retry.entities),
                    },
                )
            return (
                chunk,
                EntityRecoveryResponse(entities=[]),
                {
                    "chunk_index": chunk.index,
                    "status": "rejected",
                    "reason": "suspicious_recovery_response",
                    "detail": retry_error,
                    "initial_row_count": len(response.entities),
                    "retry_row_count": len(retry.entities),
                },
            )

        for chunk, response, quality_audit in self._parallel_map(
            chunks,
            recovery_job,
        ):
            if quality_audit is not None:
                audit.append(quality_audit)
                if quality_audit["status"] == "retried":
                    quality_retries += 1
                else:
                    quality_rejections += 1
            for row in response.entities:
                type_value = row.type.strip()
                try:
                    entity_type = EntityType(type_value)
                except ValueError:
                    entity_type = self._closest_entity_type(type_value)
                    if entity_type is None:
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
                    corrected_types += 1
                    audit.append(
                        {
                            "chunk_index": chunk.index,
                            "text": row.text,
                            "type": row.type,
                            "corrected_type": entity_type.value,
                            "occurrence": row.occurrence,
                            "status": "corrected",
                            "reason": "unambiguous_near_match_entity_type",
                        }
                    )
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
        if corrected_types:
            warnings.append(
                "LLM recovery corrected "
                f"{corrected_types} unambiguous near-match entity type(s)"
            )
        if missing_substrings:
            warnings.append(
                "LLM recovery rejected "
                f"{missing_substrings} row(s) not found exactly in their chunks"
            )
        if quality_retries:
            warnings.append(
                "LLM recovery retried "
                f"{quality_retries} suspicious chunk response(s)"
            )
        if quality_rejections:
            warnings.append(
                "LLM recovery discarded "
                f"{quality_rejections} suspicious chunk response(s) after retry"
            )
        return proposals, audit

    def _recover_chunk(
        self,
        document: Document,
        chunk: Chunk,
        chunk_existing: list[dict[str, str | int]],
        checkpoint_dir: Path | None,
        *,
        strict: bool,
    ) -> EntityRecoveryResponse:
        quality_policy = (
            "The previous extraction was pathologically dense. Return at most "
            "30 high-confidence entities. Never label ordinary words, "
            "punctuation, anatomy alone, headings, durations, ages, percentages, "
            "foods, activities, or treatment instructions as entities. A "
            "single-word span is valid only when it is independently a clear "
            "clinical symptom, diagnosis, medication, test name, or result."
            if strict
            else ""
        )
        return self.llm_backend.generate_json(
            LLMTask.ENTITY_RECOVERY,
            [
                {
                    "role": "system",
                    "content": (
                        "Independently and exhaustively extract every explicit "
                        "clinical entity mention in TEXT_CHUNK. First scan the "
                        "entire chunk. EXISTING_ENTITIES contains supplemental "
                        "hints, not an exclusion list and not a complete inventory. "
                        "Return every entity, including entities listed there; the "
                        "host will de-duplicate them. For repeated text, report its "
                        "one-based occurrence within TEXT_CHUNK. Copy one contiguous "
                        "source span exactly: never omit intervening words, "
                        "paraphrase, normalize, or join separate phrases. Never "
                        "generate offsets. "
                        "Return JSON only. Allowed types are TRIỆU_CHỨNG, "
                        "TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC. "
                        f"{quality_policy}"
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
            reasoning_enabled=False,
            call_id=(
                f"{document.id}/recovery-{chunk.index:03d}"
                + ("-quality-fallback" if strict else "")
            ),
            checkpoint_dir=checkpoint_dir,
        )

    @staticmethod
    def _closest_entity_type(value: str) -> EntityType | None:
        normalized = normalize_search(value)
        ranked = sorted(
            (
                (
                    SequenceMatcher(
                        None,
                        normalized,
                        normalize_search(candidate.value),
                    ).ratio(),
                    candidate,
                )
                for candidate in EntityType
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        top_score, top_candidate = ranked[0]
        second_score = ranked[1][0]
        if top_score < 0.75 or top_score - second_score < 0.10:
            return None
        return top_candidate

    @staticmethod
    def _occurrence_at(text: str, substring: str, start: int) -> int:
        if text[start : start + len(substring)] != substring:
            raise ValueError("span is not an exact substring at the supplied start")
        occurrence = 0
        cursor = 0
        while True:
            found = text.find(substring, cursor)
            if found < 0 or found > start:
                raise ValueError("span start is not a reconstructable occurrence")
            occurrence += 1
            if found == start:
                return occurrence
            cursor = found + len(substring)

    @staticmethod
    def _recovery_quality_error(
        response: EntityRecoveryResponse,
    ) -> str | None:
        rows = response.entities
        if len(rows) > 40:
            return f"row_count={len(rows)} exceeds 40"
        if len(rows) < 20:
            return None
        single_token_rows = sum(
            len(normalize_search(row.text).split()) <= 1 for row in rows
        )
        ratio = single_token_rows / len(rows)
        if ratio > 0.65:
            return (
                f"single_token_ratio={ratio:.3f} exceeds 0.65 "
                f"across {len(rows)} rows"
            )
        return None

    def _filter_merged_proposals(
        self,
        proposals: list[SpanProposal],
    ) -> tuple[list[SpanProposal], list[dict[str, Any]]]:
        kept: list[SpanProposal] = []
        artifacts: list[dict[str, Any]] = []
        for proposal in proposals:
            normalized = normalize_search(proposal.text)
            reject_reason: str | None = None
            if normalized in SOURCE_INDEPENDENT_NON_ENTITIES:
                reject_reason = "source_independent_non_entity_exclusion"
            elif proposal.type == EntityType.MEDICATION and (
                normalized in POST_MERGE_NON_MEDICATIONS
                or bool(
                    re.fullmatch(
                        r"thuốc(?:\s+(?:đang\s+dùng|trước\s+nhập\s+viện|"
                        r"điều\s+trị|kê\s+đơn))?",
                        normalized,
                    )
                )
            ):
                reject_reason = "source_independent_non_medication_exclusion"
            if reject_reason is None:
                kept.append(proposal)
                continue
            artifacts.append(
                {
                    "position": [proposal.start, proposal.end],
                    "text": proposal.text,
                    "initial_type": proposal.type.value,
                    "keep": False,
                    "decision_source": "deterministic_filter",
                    "reason": reject_reason,
                    "supporting_sources": proposal.evidence.get(
                        "supporting_sources",
                        [proposal.source],
                    ),
                }
            )
        return kept, artifacts

    @staticmethod
    def _automatic_review_artifact(
        proposal: SpanProposal,
        assertions: list[str],
        *,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "position": [proposal.start, proposal.end],
            "text": proposal.text,
            "initial_type": proposal.type.value,
            "initial_assertions": assertions,
            "keep": True,
            "reviewed_type": proposal.type.value,
            "reviewed_assertions": assertions,
            "decision_source": "automatic",
            "reason": reason,
            "supporting_sources": proposal.evidence.get(
                "supporting_sources",
                [proposal.source],
            ),
        }

    def _selective_review_reasons(
        self,
        document: Document,
        proposal: SpanProposal,
        initial_assertions: list[str],
        warnings: list[str],
    ) -> list[str]:
        sources = set(
            proposal.evidence.get(
                "supporting_sources",
                proposal.evidence.get("sources", [proposal.source]),
            )
        )
        sources.add(proposal.source)
        reasons: list[str] = []
        if "llm_recovery" in sources:
            reasons.append("llm_recovery")
        if (
            sources <= {"gliner"}
            and proposal.score < self.config.run.gliner_review_threshold
            and not (
                proposal.type == EntityType.TEST_RESULT
                and self._valid_laboratory_result_span(proposal.text)
            )
        ):
            reasons.append("low_confidence_gliner_only")
        if proposal.evidence.get("type_conflict"):
            reasons.append("type_conflict")
        if initial_assertions:
            reasons.append("assertion_rule")
        elif self.assertions.has_context_cue(document.text, proposal):
            reasons.append("nearby_assertion_cue")
        if proposal.type == EntityType.MEDICATION:
            structured_sources = {
                "structured_medication_rule",
                "structured_medication_sig",
            }
            if not sources & structured_sources and not self._has_strong_rxnorm(
                proposal.text,
                warnings,
            ):
                reasons.append("unstructured_medication_without_strong_rxnorm")
        if (
            proposal.type == EntityType.TEST_RESULT
            and not self._valid_laboratory_result_span(proposal.text)
        ):
            reasons.append("laboratory_span_policy")
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _valid_laboratory_result_span(text: str) -> bool:
        value = text.strip()
        return bool(
            QUALITATIVE_RESULT_RE.fullmatch(value)
            or NUMERIC_RESULT_RE.fullmatch(value)
        )

    def _has_strong_rxnorm(
        self,
        mention: str,
        warnings: list[str],
    ) -> bool:
        candidates = self._retrieve_rxnorm(mention, warnings)
        if not candidates:
            return False
        return (
            self._is_exact_candidate(candidates[0])
            or candidates[0].score >= self.config.linking.auto_exact_score
        )

    def _retrieve_icd(self, mention: str) -> list[LinkCandidate]:
        key = ("icd", normalize_search(mention))
        with self._retrieval_lock:
            cached = self._retrieval_cache.get(key)
        if cached is not None:
            return list(cached)
        rows = self.icd_index.retrieve(
            mention,
            limit=self.config.linking.retrieval_candidates,
        )
        with self._retrieval_lock:
            self._retrieval_cache[key] = list(rows)
        return rows

    def _eligible_candidates(
        self,
        task: LLMTask,
        candidates: list[LinkCandidate],
    ) -> list[LinkCandidate]:
        threshold = (
            self.config.linking.icd_min_score
            if task == LLMTask.ICD_RERANK
            else self.config.linking.rxnorm_min_score
        )
        return [
            row
            for row in candidates
            if self._is_exact_candidate(row) or row.score >= threshold
        ]

    @staticmethod
    def _is_exact_candidate(candidate: LinkCandidate) -> bool:
        return any(
            source in {"exact", "rxnav_exact_or_normalized"}
            or "exact_or_normalized" in source
            for source in candidate.retrieval_sources
        )

    def _automatic_candidate_selection(
        self,
        candidates: list[LinkCandidate],
    ) -> tuple[list[str] | None, str]:
        if not candidates:
            return [], "empty_after_threshold"
        top = candidates[0]
        if (
            self._is_exact_candidate(top)
            and top.score >= self.config.linking.auto_exact_score
        ):
            return [top.identifier], "exact_or_normalized"
        if (
            len(candidates) == 1
            and top.score >= self.config.linking.auto_single_score
        ):
            return [top.identifier], "single_strong_candidate"
        if (
            len(candidates) >= 2
            and top.score >= self.config.linking.auto_top_score
            and top.score - candidates[1].score
            >= self.config.linking.auto_score_margin
        ):
            return [top.identifier], "strong_top_margin"
        return None, "ambiguous"

    def _select_terminology(
        self,
        task: LLMTask,
        document: Document,
        entries: list[tuple[SpanProposal, list[LinkCandidate]]],
        limit: int,
        checkpoint_dir: Path | None,
    ) -> tuple[
        dict[tuple[int, int], list[LinkCandidate]],
        dict[tuple[int, int], dict[str, Any]],
    ]:
        selected = {
            (proposal.start, proposal.end): []
            for proposal, _ in entries
        }
        decisions: dict[tuple[int, int], dict[str, Any]] = {}
        groups: dict[
            tuple[str, tuple[str, ...]],
            list[tuple[SpanProposal, list[LinkCandidate]]],
        ] = {}
        for proposal, retrieved in entries:
            eligible = self._eligible_candidates(task, retrieved)
            key = (
                normalize_search(proposal.text),
                tuple(row.identifier for row in eligible),
            )
            groups.setdefault(key, []).append((proposal, eligible))

        ambiguous_representatives: list[
            tuple[SpanProposal, list[LinkCandidate]]
        ] = []
        ambiguous_groups: dict[
            tuple[int, int],
            list[tuple[SpanProposal, list[LinkCandidate]]],
        ] = {}
        for grouped_entries in groups.values():
            representative, eligible = grouped_entries[0]
            candidate_ids, reason = self._automatic_candidate_selection(eligible)
            if candidate_ids is None:
                ambiguous_representatives.append((representative, eligible))
                ambiguous_groups[
                    (representative.start, representative.end)
                ] = grouped_entries
                continue
            for index, (proposal, proposal_candidates) in enumerate(grouped_entries):
                position = (proposal.start, proposal.end)
                by_id = {
                    row.identifier: row for row in proposal_candidates
                }
                selected[position] = [
                    by_id[identifier]
                    for identifier in candidate_ids
                    if identifier in by_id
                ]
                decisions[position] = {
                    "decision_source": (
                        "automatic" if index == 0 else "reused_automatic"
                    ),
                    "reason": reason,
                    "deduplication_key": [
                        normalize_search(proposal.text),
                        list(by_id),
                    ],
                }

        if ambiguous_representatives:
            reranked = self._batch_rerank(
                task,
                document,
                ambiguous_representatives,
                limit,
                checkpoint_dir,
            )
            for representative, _ in ambiguous_representatives:
                representative_position = (
                    representative.start,
                    representative.end,
                )
                chosen_ids = [
                    row.identifier
                    for row in reranked[representative_position]
                ]
                for index, (proposal, proposal_candidates) in enumerate(
                    ambiguous_groups[representative_position]
                ):
                    position = (proposal.start, proposal.end)
                    by_id = {
                        row.identifier: row for row in proposal_candidates
                    }
                    selected[position] = [
                        by_id[identifier]
                        for identifier in chosen_ids
                        if identifier in by_id
                    ]
                    decisions[position] = {
                        "decision_source": (
                            "llm" if index == 0 else "reused_llm"
                        ),
                        "reason": "ambiguous_candidates",
                        "representative_position": list(
                            representative_position
                        ),
                    }
        return selected, decisions

    def _local_context(
        self,
        document: Document,
        start: int,
        end: int,
    ) -> dict[str, Any]:
        context_chars = self.config.run.review_context_chars
        context_start = max(0, start - context_chars)
        context_end = min(len(document.text), end + context_chars)
        return {
            "context_start": context_start,
            "context_end": context_end,
            "left_context": document.text[context_start:start],
            "right_context": document.text[end:context_end],
        }

    def _parallel_map(self, items: list[Any], function: Any) -> list[Any]:
        if len(items) <= 1 or self.config.llm.max_concurrency <= 1:
            return [function(item) for item in items]
        workers = min(self.config.llm.max_concurrency, len(items))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(function, items))

    def _document_call_audits(
        self,
        document_id: str,
        checkpoint_dir: Path | None,
    ) -> list[dict[str, Any]]:
        prefix = f"{document_id}/"
        if checkpoint_dir is not None:
            audit_dir = checkpoint_dir / "audits"
            if audit_dir.exists():
                return sorted(
                    (
                        json.loads(path.read_text("utf-8"))
                        for path in audit_dir.glob("*.json")
                    ),
                    key=lambda row: (row["call_id"], row["cache_key"]),
                )
        method = getattr(self.llm_backend, "call_audits", None)
        if method is None:
            return []
        return [
            row
            for row in method()
            if str(row.get("call_id", "")).startswith(prefix)
        ]

    @staticmethod
    def _filter_ner_proposals(
        proposals: list[SpanProposal],
    ) -> list[SpanProposal]:
        kept: list[SpanProposal] = []
        for row in proposals:
            normalized = normalize_search(row.text)
            if (
                "\n" in row.text
                or normalized in SOURCE_INDEPENDENT_NON_ENTITIES
            ):
                continue
            if (
                row.type == EntityType.MEDICATION
                and normalized in POST_MERGE_NON_MEDICATIONS
            ):
                continue
            kept.append(row)
        return kept

    def _retrieve_rxnorm(
        self,
        mention: str,
        warnings: list[str],
    ) -> list[LinkCandidate]:
        cache_key = ("rxnorm", normalize_search(mention))
        with self._retrieval_lock:
            cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        merged: dict[str, LinkCandidate] = {}
        for variant in query_variants(mention):
            try:
                with self._retrieval_lock:
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
        result = sorted(
            merged.values(),
            key=lambda row: (-row.score, row.identifier),
        )[: self.config.linking.retrieval_candidates]
        with self._retrieval_lock:
            self._retrieval_cache[cache_key] = list(result)
        return result

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
        eligible: list[LinkCandidate],
        selected: list[LinkCandidate],
        decision: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "position": [proposal.start, proposal.end],
            "text": proposal.text,
            "retrieved_candidates": [
                row.model_dump(mode="json") for row in retrieved
            ],
            "eligible_candidates": [
                row.model_dump(mode="json") for row in eligible
            ],
            "selected_candidates": [
                row.model_dump(mode="json") for row in selected
            ],
            "decision": decision
            or {
                "decision_source": "legacy",
                "reason": "review_mode_not_selective",
            },
        }
