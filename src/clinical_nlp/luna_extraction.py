from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from clinical_nlp.assertion_detection import AssertionDetector
from clinical_nlp.atomic import atomic_write_json
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.normalization import normalize_search
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.rxnorm_linking.index import query_variants
from clinical_nlp.schemas import (
    Assertion,
    Document,
    Entity,
    EntityType,
    SpanProposal,
)
from clinical_nlp.submission_variants import (
    _document_sort_key,
    _entity_key,
    load_output_directory,
)
from clinical_nlp.text import find_occurrence, is_masked_span
from clinical_nlp.validation import validate_entities, validate_output_directory


DEFAULT_MODEL = "gpt-5.6-sol"
CHECKPOINT_VERSION = 1
MAX_DENSITY_PER_1000 = 45.0
FULL_MIN_ENTITIES = 2300
FULL_MAX_ENTITIES = 4500
BASELINE_WER = 66.9345
BEST_SUBMISSION_SCORE = 29.5991
BASELINE_CANDIDATE_SCORE = 20.9721
BEST_ASSERTION_SCORE = 37.6352

_TRANSIENT_MARKERS = (
    "connection reset",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "internal server error",
    "overloaded",
)
_LIMIT_MARKERS = (
    "rate limit",
    "usage limit",
    "quota",
    "too many requests",
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;\n]")
_CONTRAST_RE = re.compile(r"(?i)\b(?:nhưng|tuy\s+nhiên|however|but)\b")
_HISTORY_RE = re.compile(
    r"(?i)(?:tiền\s+sử|trước\s+đây|đã\s+từng|"
    r"history\s+of|previously|past\s+medical\s+history)"
)


class LunaEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    occurrence: int = Field(ge=1)
    type: EntityType


class LunaExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    entities: list[LunaEntity]


class LunaPreflightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["gpt-5.6-sol"]
    status: Literal["ready"]


class LunaAuditDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^c\d{4}$")
    keep: bool
    type: EntityType


class LunaAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    decisions: list[LunaAuditDecision]


class SubscriptionLimitError(RuntimeError):
    """The authenticated subscription cannot accept more work right now."""


class LunaCallError(RuntimeError):
    """A schema-constrained Luna call failed after its retry budget."""


class AuditCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    occurrence: int
    type: EntityType
    position: tuple[int, int]
    source: Literal["baseline", "luna"]
    change_kind: Literal["addition", "removal"]

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.position[0], self.position[1], self.type.value)


class LunaLeaderboardMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wer: float = Field(ge=0)
    assertions_score: float = Field(ge=0)
    candidates_score: float = Field(ge=0)
    final_score: float = Field(ge=0)


ResponseT = TypeVar("ResponseT", bound=BaseModel)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _occurrence_at(text: str, substring: str, start: int) -> int:
    if text[start : start + len(substring)] != substring:
        raise ValueError("span is not an exact substring at its supplied start")
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


def _mark_order_relaxed_occurrence(
    text: str,
    substring: str,
    occurrence: int,
) -> tuple[int, int]:
    def keyed(
        value: str,
    ) -> tuple[str, list[int], list[int]]:
        parts: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        start = 0
        while start < len(value):
            end = start + 1
            while end < len(value) and unicodedata.combining(value[end]):
                end += 1
            nfd = unicodedata.normalize("NFD", value[start:end])
            bases = [char for char in nfd if not unicodedata.combining(char)]
            marks = sorted(
                (char for char in nfd if unicodedata.combining(char)),
                key=lambda char: (unicodedata.combining(char), ord(char)),
            )
            cluster_key = "".join([*bases, *marks])
            parts.append(cluster_key)
            starts.extend([start] * len(cluster_key))
            ends.extend([end] * len(cluster_key))
            start = end
        return "".join(parts), starts, ends

    text_key, starts, ends = keyed(text)
    substring_key, _, _ = keyed(substring)
    cursor = 0
    found = 0
    while True:
        match = text_key.find(substring_key, cursor)
        if match < 0:
            raise ValueError("mark-order-relaxed substring not found")
        found += 1
        if found == occurrence:
            match_end = match + len(substring_key)
            return starts[match], ends[match_end - 1]
        cursor = match + max(1, len(substring_key))


def reconstruct_luna_entities(
    document: Document,
    response: LunaExtractionResponse,
) -> list[dict[str, Any]]:
    if response.document_id != document.id:
        raise ValueError(
            f"Luna returned document {response.document_id!r} for {document.id!r}"
        )
    located: list[tuple[LunaEntity, int, int]] = []
    failures: list[dict[str, Any]] = []
    for proposed in response.entities:
        try:
            start, end = find_occurrence(
                document.text,
                proposed.text,
                proposed.occurrence,
            )
        except ValueError:
            failure = {
                "text": proposed.text,
                "occurrence": proposed.occurrence,
                "type": proposed.type.value,
            }
            try:
                relaxed_start, relaxed_end = _mark_order_relaxed_occurrence(
                    document.text,
                    proposed.text,
                    proposed.occurrence,
                )
            except ValueError:
                pass
            else:
                failure["exact_source_json"] = json.dumps(
                    document.text[relaxed_start:relaxed_end],
                    ensure_ascii=True,
                )
            failures.append(failure)
        else:
            located.append((proposed, start, end))
    if failures:
        raise ValueError(
            "unreconstructable exact substrings; copy the original Unicode "
            "codepoints without normalization: "
            + json.dumps(failures, ensure_ascii=False)
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for proposed, start, end in located:
        key = (start, end, proposed.type.value)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "text": document.text[start:end],
                "type": proposed.type.value,
                "position": [start, end],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["position"][0],
            row["position"][1],
            row["type"],
        ),
    )


def _json_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LunaCallError(
                f"Codex emitted invalid JSON event at line {line_number}"
            ) from exc
        if not isinstance(event, dict):
            raise LunaCallError("Codex JSON event must be an object")
        events.append(event)
    return events


def _final_agent_message(events: list[dict[str, Any]]) -> str:
    messages: list[str] = []
    for event in events:
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            messages.append(item["text"])
    if not messages:
        raise LunaCallError("Codex did not emit a final agent message")
    return messages[-1]


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        usage = event.get("usage")
        if isinstance(usage, dict):
            return usage
    return {}


class CodexLunaRunner:
    def __init__(
        self,
        *,
        repository: Path,
        run_dir: Path,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "high",
        codex_bin: str | Path | None = None,
        max_retries: int = 2,
        timeout_seconds: float = 900.0,
    ) -> None:
        resolved = (
            str(codex_bin)
            if codex_bin is not None
            else shutil.which("codex")
        )
        if not resolved:
            raise RuntimeError("codex executable is unavailable")
        if model != DEFAULT_MODEL:
            raise ValueError(
                f"refusing non-Luna model {model!r}; expected {DEFAULT_MODEL!r}"
            )
        self.codex_bin = resolved
        self.repository = repository.resolve()
        self.run_dir = run_dir
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def preflight_auth(self) -> dict[str, Any]:
        completed = subprocess.run(
            [self.codex_bin, "login", "status"],
            cwd=self.repository,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        combined = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0 or "Logged in using ChatGPT" not in combined:
            raise RuntimeError("Codex CLI is not authenticated through ChatGPT OAuth")
        return {
            "codex_bin": self.codex_bin,
            "authenticated_with": "ChatGPT",
            "requested_model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }

    def preflight_model(self) -> dict[str, Any]:
        result = self.preflight_auth()

        def validate(response: LunaPreflightResponse) -> None:
            if response.model != self.model or response.status != "ready":
                raise ValueError("Luna model preflight returned the wrong identity")

        response = self.run_structured(
            task="preflight",
            document_id="model",
            document_text=self.model,
            prompt=(
                "This is a model-availability preflight. Return exactly the "
                "schema-constrained object declaring model gpt-5.6-sol and "
                "status ready."
            ),
            response_type=LunaPreflightResponse,
            validator=validate,
        )
        return {
            **result,
            "confirmed_model": response.model,
            "model_available": response.status == "ready",
        }

    def _command(self, schema_path: Path) -> list[str]:
        return [
            self.codex_bin,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--sandbox",
            "read-only",
            "--cd",
            str(self.repository),
            "--output-schema",
            str(schema_path),
            "-",
        ]

    def run_structured(
        self,
        *,
        task: str,
        document_id: str,
        document_text: str,
        prompt: str,
        response_type: type[ResponseT],
        validator: Callable[[ResponseT], None],
    ) -> ResponseT:
        prompt_hash = _sha256_text(prompt)
        document_hash = _sha256_text(document_text)
        checkpoint_path = (
            self.run_dir / "checkpoints" / task / f"{document_id}.json"
        )
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
            if (
                checkpoint.get("version") == CHECKPOINT_VERSION
                and checkpoint.get("model") == self.model
                and checkpoint.get("prompt_sha256") == prompt_hash
                and checkpoint.get("document_sha256") == document_hash
                and checkpoint.get("status") == "complete"
            ):
                response = response_type.model_validate(checkpoint["response"])
                validator(response)
                return response

        schema_path = self.run_dir / "schemas" / f"{task}.json"
        atomic_write_json(schema_path, response_type.model_json_schema())
        attempts: list[dict[str, Any]] = []
        error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retry_prompt = prompt
            if attempt:
                retry_prompt += (
                    "\n\nSTRICT RETRY: The previous response failed schema or "
                    "host reconstruction. Return a complete corrected JSON "
                    "object only; do not omit repeated mentions."
                )
                if error is not None:
                    retry_prompt += (
                        "\nPrevious host validation error: "
                        f"{type(error).__name__}: {error}"
                        "\nThe original may use decomposed Unicode combining "
                        "marks. Do not normalize, retype, or visually recreate "
                        "text. Copy exact source codepoints. When the error "
                        "supplies exact_source_json, use that escaped JSON "
                        "string value exactly. If a long span is "
                        "unstable, choose a shorter correctly bounded exact "
                        "clinical substring rather than paraphrasing. Never "
                        "return the same (text, occurrence, type) row twice. "
                        "Genuine repeated mentions must use distinct "
                        "document-wide occurrence numbers."
                    )
            attempt_row: dict[str, Any] = {
                "attempt": attempt + 1,
            }
            try:
                completed = subprocess.run(
                    self._command(schema_path),
                    cwd=self.repository,
                    input=retry_prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                combined = (
                    f"{completed.stdout}\n{completed.stderr}".casefold()
                )
                attempt_row.update(
                    {
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
                if any(marker in combined for marker in _LIMIT_MARKERS):
                    raise SubscriptionLimitError(
                        "Luna subscription limit reached; resume this run later"
                    )
                if completed.returncode != 0:
                    marker = (
                        "transient Codex failure"
                        if any(value in combined for value in _TRANSIENT_MARKERS)
                        else "Codex process failed"
                    )
                    raise LunaCallError(
                        f"{marker} with exit code {completed.returncode}"
                    )
                events = _json_events(completed.stdout)
                raw_response = json.loads(_final_agent_message(events))
                response = response_type.model_validate(raw_response)
                validator(response)
                attempt_row["usage"] = _usage_from_events(events)
                attempt_row["event_count"] = len(events)
                attempt_row.pop("stdout", None)
                attempts.append(attempt_row)
                atomic_write_json(
                    checkpoint_path,
                    {
                        "version": CHECKPOINT_VERSION,
                        "status": "complete",
                        "task": task,
                        "document_id": document_id,
                        "model": self.model,
                        "confirmed_model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                        "prompt_sha256": prompt_hash,
                        "document_sha256": document_hash,
                        "attempts": attempts,
                        "response": response.model_dump(mode="json"),
                    },
                )
                return response
            except SubscriptionLimitError:
                attempt_row["error"] = "subscription_limit"
                attempts.append(attempt_row)
                atomic_write_json(
                    checkpoint_path,
                    {
                        "version": CHECKPOINT_VERSION,
                        "status": "subscription_limit",
                        "task": task,
                        "document_id": document_id,
                        "model": self.model,
                        "prompt_sha256": prompt_hash,
                        "document_sha256": document_hash,
                        "attempts": attempts,
                    },
                )
                raise
            except Exception as exc:
                error = exc
                attempt_row["error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(attempt_row)

        atomic_write_json(
            checkpoint_path,
            {
                "version": CHECKPOINT_VERSION,
                "status": "failed",
                "task": task,
                "document_id": document_id,
                "model": self.model,
                "prompt_sha256": prompt_hash,
                "document_sha256": document_hash,
                "attempts": attempts,
            },
        )
        raise LunaCallError(
            f"{task} failed for document {document_id} after "
            f"{self.max_retries + 1} attempts"
        ) from error


def _existing_entities_for_prompt(
    document: Document,
    baseline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "text": row["text"],
            "occurrence": _occurrence_at(
                document.text,
                row["text"],
                row["position"][0],
            ),
            "type": row["type"],
        }
        for row in baseline
        if not is_masked_span(row["text"])
    ]


def extraction_prompt(
    document: Document,
    baseline: list[dict[str, Any]],
    *,
    recovery_only: bool = False,
) -> str:
    mode = (
        "Find only missing entities not already listed, but still scan the "
        "entire document before comparing occurrences."
        if recovery_only
        else "Return the complete final entity inventory for this document."
    )
    return (
        "You are Luna, performing high-precision exhaustive Vietnamese clinical "
        "entity extraction for a leaderboard submission.\n\n"
        f"{mode}\n"
        "First scan the entire ORIGINAL_TEXT independently. Then compare against "
        "EXISTING_ENTITIES. Existing rows are hints and may be wrong. Never "
        "return headings, generic clinical words, redaction placeholders, "
        "unsupported fragments, or speculative diagnoses.\n\n"
        "Allowed types exactly: TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, "
        "KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC.\n"
        "Scan sentence by sentence and return every explicit occurrence, "
        "including repeated mentions. Explicit clinical states such as "
        "pregnancy with gestational age may be CHẨN_ĐOÁN; patient-described "
        "worry or anxiety may be TRIỆU_CHỨNG. Do not infer either when it is "
        "not written.\n"
        "Medication text includes a contiguous name plus dose, form, route, and "
        "frequency when written, but excludes the indication. Laboratory results "
        "must be a qualitative result or a numeric value/range with an optional "
        "measured unit. A standalone percentage in an efficacy, sensitivity, "
        "risk, or test-result statement (for example 80%) is a valid "
        "KẾT_QUẢ_XÉT_NGHIỆM. Copy text exactly. occurrence is one-based across the "
        "whole original document for that exact text. Return no offsets and no "
        "candidate IDs or assertions.\n\n"
        f"DOCUMENT_ID:\n{document.id}\n\n"
        "EXISTING_ENTITIES:\n"
        f"{json.dumps(_existing_entities_for_prompt(document, baseline), ensure_ascii=False)}"
        "\n\nORIGINAL_TEXT:\n"
        f"{document.text}\n\n"
        'Return {"document_id":"ID","entities":[{"text":"exact substring",'
        '"occurrence":1,"type":"allowed type"}]}.'
    )


def _candidate_rows(
    document: Document,
    baseline: list[dict[str, Any]],
    luna: list[dict[str, Any]],
) -> list[AuditCandidate]:
    baseline_by_key = {
        (
            row["position"][0],
            row["position"][1],
            row["type"],
        ): row
        for row in baseline
    }
    luna_by_key = {
        (
            row["position"][0],
            row["position"][1],
            row["type"],
        ): row
        for row in luna
    }
    candidates: list[AuditCandidate] = []
    luna_only_keys = luna_by_key.keys() - baseline_by_key.keys()
    conflicting_baseline_keys = {
        baseline_key
        for baseline_key in baseline_by_key
        for luna_key in luna_only_keys
        if (
            baseline_key[0] < luna_key[1]
            and luna_key[0] < baseline_key[1]
        )
    }
    baseline_review_keys = (
        baseline_by_key.keys() - luna_by_key.keys()
    ) | conflicting_baseline_keys
    changed = [
        ("baseline", "removal", key, baseline_by_key[key])
        for key in baseline_review_keys
    ] + [
        ("luna", "addition", key, luna_by_key[key])
        for key in luna_only_keys
    ]
    changed.sort(
        key=lambda row: (
            row[2][0],
            row[2][1],
            row[2][2],
            row[0],
        )
    )
    for index, (source, kind, key, row) in enumerate(changed, 1):
        candidates.append(
            AuditCandidate(
                id=f"c{index:04d}",
                text=row["text"],
                occurrence=_occurrence_at(
                    document.text,
                    row["text"],
                    row["position"][0],
                ),
                type=EntityType(row["type"]),
                position=tuple(row["position"]),
                source=source,
                change_kind=kind,
            )
        )
    return candidates


def audit_prompt(
    document: Document,
    candidates: list[AuditCandidate],
) -> str:
    payload = [
        {
            "id": row.id,
            "text": row.text,
            "occurrence": row.occurrence,
            "type": row.type.value,
            "source": row.source,
            "change_kind": row.change_kind,
        }
        for row in candidates
    ]
    conflicts = [
        [left.id, right.id]
        for index, left in enumerate(candidates)
        for right in candidates[index + 1 :]
        if (
            left.position[0] < right.position[1]
            and right.position[0] < left.position[1]
        )
    ]
    return (
        "Audit proposed changes to a Vietnamese clinical entity inventory. "
        "Review every candidate against ORIGINAL_TEXT. You cannot add a row, "
        "change its text or occurrence, or omit a decision.\n\n"
        "For a Luna addition, keep=true only for an explicit, correctly bounded "
        "mention of the supplied type. For a baseline removal, keep=true means "
        "retain the baseline row; keep=false means it is definitely a heading, "
        "generic word, placeholder, malformed span, or unsupported entity. "
        "Retain explicit numeric percentages in efficacy, sensitivity, risk, "
        "or test-result statements, including a standalone 80%. Explicit "
        "pregnancy with gestational age and patient-described anxiety are "
        "valid clinical mentions when their supplied type is plausible. "
        "When uncertain, reject an addition and retain a baseline row. Return "
        "the supplied type unchanged; retyping is represented by a separate "
        "Luna addition. Every pair in INCOMPATIBLE_CANDIDATE_PAIRS overlaps: "
        "never return keep=true for both IDs in a pair. Prefer the baseline "
        "candidate unless the new boundary is clearly superior.\n\n"
        f"DOCUMENT_ID:\n{document.id}\n\n"
        f"CANDIDATES:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "INCOMPATIBLE_CANDIDATE_PAIRS:\n"
        f"{json.dumps(conflicts, ensure_ascii=False)}\n\n"
        f"ORIGINAL_TEXT:\n{document.text}\n\n"
        'Return {"document_id":"ID","decisions":[{"id":"c0001",'
        '"keep":true,"type":"supplied type"}]}.'
    )


def _decision_map(
    document_id: str,
    candidates: list[AuditCandidate],
    response: LunaAuditResponse,
) -> dict[str, LunaAuditDecision]:
    if response.document_id != document_id:
        raise ValueError("audit response belongs to another document")
    decisions: dict[str, LunaAuditDecision] = {}
    by_id = {row.id: row for row in candidates}
    for decision in response.decisions:
        if decision.id in decisions:
            raise ValueError(f"duplicate audit decision {decision.id}")
        candidate = by_id.get(decision.id)
        if candidate is None:
            raise ValueError(f"audit invented candidate id {decision.id}")
        if decision.type != candidate.type:
            raise ValueError(f"audit retyped candidate {decision.id}")
        decisions[decision.id] = decision
    if decisions.keys() != by_id.keys():
        raise ValueError("audit decisions must cover every supplied candidate")
    return decisions


def merge_luna_inventory(
    document: Document,
    baseline: list[dict[str, Any]],
    luna: list[dict[str, Any]],
    candidates: list[AuditCandidate],
    audit: LunaAuditResponse,
) -> list[dict[str, Any]]:
    decisions = _decision_map(document.id, candidates, audit)
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if (
                left.position[0] < right.position[1]
                and right.position[0] < left.position[1]
                and decisions[left.id].keep
                and decisions[right.id].keep
            ):
                raise ValueError(
                    "audit retained incompatible candidates "
                    f"{left.id} and {right.id}"
                )
    candidate_by_key = {row.key: row for row in candidates}
    baseline_by_key = {
        (row["position"][0], row["position"][1], row["type"]): row
        for row in baseline
    }
    luna_by_key = {
        (row["position"][0], row["position"][1], row["type"]): row
        for row in luna
    }
    selected: list[dict[str, Any]] = []
    for key, row in baseline_by_key.items():
        candidate = candidate_by_key.get(key)
        if (
            candidate is None
            and key in luna_by_key
        ) or (
            candidate is not None
            and decisions[candidate.id].keep
        ):
            selected.append(row)
    for key, row in luna_by_key.items():
        if key in baseline_by_key:
            continue
        candidate = candidate_by_key[key]
        if decisions[candidate.id].keep:
            selected.append(row)
    selected.sort(
        key=lambda row: (
            row["position"][0],
            row["position"][1],
            row["type"],
        )
    )
    previous_end = -1
    for row in selected:
        if is_masked_span(row["text"]):
            raise ValueError("audit retained a redaction placeholder")
        if row["position"][0] < previous_end:
            raise ValueError("audit retained overlapping entities")
        previous_end = row["position"][1]
    return selected


def _exact_icd_candidates(index: ICDIndex, mention: str) -> list[str]:
    exact = {
        row.identifier
        for row in index.retrieve(mention, limit=20)
        if "exact" in row.retrieval_sources
    }
    return sorted(exact) if len(exact) == 1 else []


def _exact_rxnorm_candidates(index: RxNormIndex, mention: str) -> list[str]:
    variants = {normalize_search(value) for value in query_variants(mention)}
    exact = {
        identifier
        for identifier, concept in index.concepts.items()
        if normalize_search(concept.get("name", "")) in variants
    }
    return sorted(exact) if len(exact) == 1 else []


def _conservative_assertions(
    document: Document,
    row: dict[str, Any],
    detector: AssertionDetector,
) -> list[str]:
    entity_type = EntityType(row["type"])
    if entity_type not in {
        EntityType.SYMPTOM,
        EntityType.DIAGNOSIS,
        EntityType.MEDICATION,
    }:
        return []
    proposal = SpanProposal(
        start=row["position"][0],
        end=row["position"][1],
        text=row["text"],
        type=entity_type,
        source="luna",
        score=1.0,
    )
    detected = set(detector.detect(document.text, proposal))
    prefix = document.text[max(0, proposal.start - 180) : proposal.start]
    boundary = max(
        [match.end() for match in _CLAUSE_BOUNDARY_RE.finditer(prefix)] or [0]
    )
    clause = prefix[boundary:]
    contrasts = list(_CONTRAST_RE.finditer(clause))
    if contrasts:
        clause = clause[contrasts[-1].end() :]
    selected: list[Assertion] = []
    if Assertion.NEGATED in detected:
        selected.append(Assertion.NEGATED)
    if Assertion.FAMILY in detected:
        selected.append(Assertion.FAMILY)
    if Assertion.HISTORICAL in detected and (
        _HISTORY_RE.search(clause)
        or (
            entity_type == EntityType.MEDICATION
            and detector.section_at(document.text, proposal.start)
            == "medication_history"
        )
    ):
        selected.append(Assertion.HISTORICAL)
    order = [Assertion.NEGATED, Assertion.FAMILY, Assertion.HISTORICAL]
    return [item.value for item in order if item in selected]


def apply_metadata(
    document: Document,
    inventory: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    icd_index: ICDIndex,
    rxnorm_index: RxNormIndex,
    assertion_detector: AssertionDetector,
) -> list[dict[str, Any]]:
    baseline_by_key = {
        (
            row["position"][0],
            row["position"][1],
            row["type"],
        ): row
        for row in baseline
    }
    entities: list[Entity] = []
    for row in inventory:
        key = (row["position"][0], row["position"][1], row["type"])
        protected = baseline_by_key.get(key)
        if protected is not None:
            entities.append(Entity.model_validate(protected))
            continue
        entity_type = EntityType(row["type"])
        candidates: list[str] = []
        if entity_type == EntityType.DIAGNOSIS:
            candidates = _exact_icd_candidates(icd_index, row["text"])
        elif entity_type == EntityType.MEDICATION:
            candidates = _exact_rxnorm_candidates(rxnorm_index, row["text"])
        entities.append(
            Entity(
                text=row["text"],
                type=entity_type,
                candidates=candidates,
                assertions=_conservative_assertions(
                    document,
                    row,
                    assertion_detector,
                ),
                position=tuple(row["position"]),
            )
        )
    validate_entities(document, entities)
    return [entity.output_dict() for entity in entities]


def _inventory_diff(
    baseline: dict[str, list[dict[str, Any]]],
    variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    baseline_index = {
        _entity_key(document_id, row): row
        for document_id, rows in baseline.items()
        for row in rows
    }
    variant_index = {
        _entity_key(document_id, row): row
        for document_id, rows in variant.items()
        for row in rows
    }
    shared = baseline_index.keys() & variant_index.keys()
    baseline_by_span = {
        (document_id, row["position"][0], row["position"][1]): row
        for document_id, rows in baseline.items()
        for row in rows
    }
    variant_by_span = {
        (document_id, row["position"][0], row["position"][1]): row
        for document_id, rows in variant.items()
        for row in rows
    }
    retyped_spans = {
        key
        for key in baseline_by_span.keys() & variant_by_span.keys()
        if baseline_by_span[key]["type"] != variant_by_span[key]["type"]
    }
    metadata_changed = sum(
        (
            baseline_index[key]["candidates"]
            != variant_index[key]["candidates"]
        )
        or (
            baseline_index[key]["assertions"]
            != variant_index[key]["assertions"]
        )
        for key in shared
    )
    return {
        "baseline_entities": len(baseline_index),
        "variant_entities": len(variant_index),
        "added": len(variant_index.keys() - baseline_index.keys()),
        "removed": len(baseline_index.keys() - variant_index.keys()),
        "retyped": len(retyped_spans),
        "unchanged": len(shared),
        "metadata_changed": metadata_changed,
        "candidate_drift_on_shared": sum(
            baseline_index[key]["candidates"]
            != variant_index[key]["candidates"]
            for key in shared
        ),
        "assertion_drift_on_shared": sum(
            baseline_index[key]["assertions"]
            != variant_index[key]["assertions"]
            for key in shared
        ),
    }


def _inventory_change_manifest(
    baseline: dict[str, list[dict[str, Any]]],
    variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    def identity(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": row["text"],
            "type": row["type"],
            "position": list(row["position"]),
        }

    by_document: dict[str, Any] = {}
    type_counts: dict[str, dict[str, int]] = {
        kind: {entity_type.value: 0 for entity_type in EntityType}
        for kind in (
            "added",
            "removed",
            "retyped_from",
            "retyped_to",
            "unchanged",
            "metadata_changed",
        )
    }
    document_ids = sorted(
        baseline.keys() | variant.keys(),
        key=_document_sort_key,
    )
    for document_id in document_ids:
        baseline_rows = baseline.get(document_id, [])
        variant_rows = variant.get(document_id, [])
        baseline_exact = {
            (row["position"][0], row["position"][1], row["type"]): row
            for row in baseline_rows
        }
        variant_exact = {
            (row["position"][0], row["position"][1], row["type"]): row
            for row in variant_rows
        }
        baseline_spans = {
            (row["position"][0], row["position"][1]): row
            for row in baseline_rows
        }
        variant_spans = {
            (row["position"][0], row["position"][1]): row
            for row in variant_rows
        }
        retyped_spans = {
            span
            for span in baseline_spans.keys() & variant_spans.keys()
            if baseline_spans[span]["type"] != variant_spans[span]["type"]
        }
        unchanged_keys = sorted(
            baseline_exact.keys() & variant_exact.keys()
        )
        added_keys = sorted(
            key
            for key in variant_exact.keys() - baseline_exact.keys()
            if (key[0], key[1]) not in retyped_spans
        )
        removed_keys = sorted(
            key
            for key in baseline_exact.keys() - variant_exact.keys()
            if (key[0], key[1]) not in retyped_spans
        )
        metadata_keys = [
            key
            for key in unchanged_keys
            if (
                baseline_exact[key]["candidates"]
                != variant_exact[key]["candidates"]
                or baseline_exact[key]["assertions"]
                != variant_exact[key]["assertions"]
            )
        ]
        retyped = [
            {
                "text": variant_spans[span]["text"],
                "position": list(span),
                "from_type": baseline_spans[span]["type"],
                "to_type": variant_spans[span]["type"],
            }
            for span in sorted(retyped_spans)
        ]
        row = {
            "added": [identity(variant_exact[key]) for key in added_keys],
            "removed": [identity(baseline_exact[key]) for key in removed_keys],
            "retyped": retyped,
            "unchanged": [
                identity(variant_exact[key]) for key in unchanged_keys
            ],
            "metadata_changed": [
                {
                    **identity(variant_exact[key]),
                    "baseline_candidates": baseline_exact[key]["candidates"],
                    "variant_candidates": variant_exact[key]["candidates"],
                    "baseline_assertions": baseline_exact[key]["assertions"],
                    "variant_assertions": variant_exact[key]["assertions"],
                }
                for key in metadata_keys
            ],
        }
        by_document[document_id] = row
        for kind in ("added", "removed", "unchanged", "metadata_changed"):
            for changed in row[kind]:
                type_counts[kind][changed["type"]] += 1
        for changed in retyped:
            type_counts["retyped_from"][changed["from_type"]] += 1
            type_counts["retyped_to"][changed["to_type"]] += 1
    return {
        "summary": _inventory_diff(baseline, variant),
        "by_type": type_counts,
        "by_document": by_document,
    }


def run_luna_consensus(
    *,
    repository: Path,
    input_dir: Path,
    baseline_output: Path,
    run_dir: Path,
    document_ids: Iterable[str],
    codex_bin: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "high",
    max_retries: int = 2,
    timeout_seconds: float = 900.0,
    enforce_full_guard: bool = False,
) -> dict[str, Any]:
    runner = CodexLunaRunner(
        repository=repository,
        run_dir=run_dir,
        model=model,
        reasoning_effort=reasoning_effort,
        codex_bin=codex_bin,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    preflight = runner.preflight_model()
    baseline_all = load_output_directory(baseline_output)
    selected_ids = sorted(set(document_ids), key=_document_sort_key)
    if not selected_ids:
        raise ValueError("at least one document must be selected")
    missing = [
        document_id
        for document_id in selected_ids
        if document_id not in baseline_all
        or not (input_dir / f"{document_id}.txt").exists()
    ]
    if missing:
        raise ValueError(f"unknown selected documents: {missing}")

    icd_index = ICDIndex.load(repository / "artifacts/icd_index.json")
    rxnorm_index = RxNormIndex(
        repository / "artifacts/rxnorm_full100_cache.json",
        use_api=False,
    )
    detector = AssertionDetector()
    variant: dict[str, list[dict[str, Any]]] = {}
    per_document: dict[str, Any] = {}
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in selected_ids:
        document = Document(
            id=document_id,
            text=(input_dir / f"{document_id}.txt").read_text("utf-8"),
        )
        baseline = baseline_all[document_id]
        extract_prompt = extraction_prompt(document, baseline)

        reconstructed: list[dict[str, Any]] = []

        def validate_extraction(response: LunaExtractionResponse) -> None:
            nonlocal reconstructed
            reconstructed = reconstruct_luna_entities(document, response)

        extraction = runner.run_structured(
            task="extract",
            document_id=document_id,
            document_text=document.text,
            prompt=extract_prompt,
            response_type=LunaExtractionResponse,
            validator=validate_extraction,
        )
        candidates = _candidate_rows(document, baseline, reconstructed)
        if candidates:
            decisions: dict[str, LunaAuditDecision] = {}
            merged: list[dict[str, Any]] = []

            def validate_audit(response: LunaAuditResponse) -> None:
                nonlocal decisions, merged
                decisions = _decision_map(document_id, candidates, response)
                merged = merge_luna_inventory(
                    document,
                    baseline,
                    reconstructed,
                    candidates,
                    response,
                )

            audit = runner.run_structured(
                task="audit",
                document_id=document_id,
                document_text=document.text,
                prompt=audit_prompt(document, candidates),
                response_type=LunaAuditResponse,
                validator=validate_audit,
            )
        else:
            audit = LunaAuditResponse(document_id=document_id, decisions=[])
            merged = list(baseline)
        rows = apply_metadata(
            document,
            merged,
            baseline,
            icd_index=icd_index,
            rxnorm_index=rxnorm_index,
            assertion_detector=detector,
        )
        density = 1000.0 * len(rows) / max(1, len(document.text))
        if density > MAX_DENSITY_PER_1000:
            raise ValueError(
                f"document {document_id} density {density:.2f}/1000 exceeds "
                f"{MAX_DENSITY_PER_1000:.2f}"
            )
        atomic_write_json(output_dir / f"{document_id}.json", rows)
        atomic_write_json(
            run_dir / "documents" / document_id / "luna_extraction.json",
            extraction.model_dump(mode="json"),
        )
        atomic_write_json(
            run_dir / "documents" / document_id / "luna_audit.json",
            audit.model_dump(mode="json"),
        )
        per_document[document_id] = {
            "baseline_entities": len(baseline),
            "luna_proposals": len(extraction.entities),
            "luna_entities": len(reconstructed),
            "duplicate_proposals_collapsed": (
                len(extraction.entities) - len(reconstructed)
            ),
            "audit_candidates": len(candidates),
            "output_entities": len(rows),
            "density_per_1000": density,
        }
        variant[document_id] = rows

    total = sum(len(rows) for rows in variant.values())
    if enforce_full_guard:
        if len(selected_ids) != 100:
            raise ValueError("full guard requires exactly 100 selected documents")
        if not FULL_MIN_ENTITIES <= total <= FULL_MAX_ENTITIES:
            raise ValueError(
                f"full Luna inventory {total} falls outside "
                f"{FULL_MIN_ENTITIES}..{FULL_MAX_ENTITIES}"
            )
    validate_output_directory(
        output_dir,
        input_dir,
        expected_stems=set(selected_ids),
    )
    baseline_selected = {
        document_id: baseline_all[document_id]
        for document_id in selected_ids
    }
    changes = _inventory_change_manifest(baseline_selected, variant)
    manifest = {
        "kind": "luna_consensus",
        "preflight": preflight,
        "baseline_output": str(baseline_output),
        "input_dir": str(input_dir),
        "run_dir": str(run_dir),
        "documents": len(selected_ids),
        "entities": total,
        "full_guard_enforced": enforce_full_guard,
        "diff": changes["summary"],
        "changes": changes,
        "per_document": per_document,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest


def select_submission4_strategy(
    metrics: LunaLeaderboardMetrics,
) -> Literal["miss_recovery", "clear_changed_metadata", "precision_prune"]:
    if (
        metrics.wer < BASELINE_WER
        and metrics.final_score > BEST_SUBMISSION_SCORE
    ):
        return "miss_recovery"
    if metrics.wer < BASELINE_WER:
        return "clear_changed_metadata"
    return "precision_prune"


def build_score_adjusted_metadata_variant(
    *,
    baseline_output: Path,
    luna_output: Path,
    input_dir: Path,
    output_dir: Path,
    metrics: LunaLeaderboardMetrics,
) -> dict[str, Any]:
    strategy = select_submission4_strategy(metrics)
    if strategy != "clear_changed_metadata":
        raise ValueError(
            f"leaderboard metrics select {strategy}, not metadata clearing"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty {output_dir}")
    baseline = load_output_directory(baseline_output)
    luna = load_output_directory(luna_output)
    if baseline.keys() != luna.keys():
        raise ValueError("baseline and Luna document sets differ")
    baseline_index = {
        _entity_key(document_id, row): row
        for document_id, rows in baseline.items()
        for row in rows
    }
    clear_candidates = (
        metrics.candidates_score < BASELINE_CANDIDATE_SCORE
    )
    clear_assertions = (
        metrics.assertions_score < BEST_ASSERTION_SCORE
    )
    variant: dict[str, list[dict[str, Any]]] = {}
    changed_rows = 0
    for document_id in sorted(luna, key=_document_sort_key):
        rows: list[dict[str, Any]] = []
        for source in luna[document_id]:
            row = dict(source)
            if _entity_key(document_id, source) not in baseline_index:
                before = (list(row["candidates"]), list(row["assertions"]))
                if clear_candidates:
                    row["candidates"] = []
                if clear_assertions:
                    row["assertions"] = []
                after = (list(row["candidates"]), list(row["assertions"]))
                changed_rows += before != after
            rows.append(Entity.model_validate(row).output_dict())
        variant[document_id] = rows
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(variant, key=_document_sort_key):
        atomic_write_json(
            output_dir / f"{document_id}.json",
            variant[document_id],
        )
    validate_output_directory(
        output_dir,
        input_dir,
        expected_stems=set(variant),
    )
    changes = _inventory_change_manifest(baseline, variant)
    manifest = {
        "kind": "luna_score_adjusted_metadata",
        "strategy": strategy,
        "leaderboard_metrics": metrics.model_dump(mode="json"),
        "clear_candidates_on_changed_rows": clear_candidates,
        "clear_assertions_on_changed_rows": clear_assertions,
        "changed_rows": changed_rows,
        "changes": changes,
    }
    atomic_write_json(output_dir.parent / "manifest.json", manifest)
    return manifest


def _copy_corpus(
    corpus: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        document_id: [dict(row) for row in rows]
        for document_id, rows in corpus.items()
    }


def _has_overlap(
    row: dict[str, Any],
    others: Iterable[dict[str, Any]],
) -> bool:
    start, end = row["position"]
    return any(
        start < other["position"][1]
        and other["position"][0] < end
        for other in others
    )


def run_miss_only_recovery(
    *,
    repository: Path,
    input_dir: Path,
    base_output: Path,
    run_dir: Path,
    density_threshold: float = 18.0,
    codex_bin: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "high",
    max_retries: int = 2,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    runner = CodexLunaRunner(
        repository=repository,
        run_dir=run_dir,
        model=model,
        reasoning_effort=reasoning_effort,
        codex_bin=codex_bin,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    preflight = runner.preflight_model()
    base = load_output_directory(base_output)
    variant = _copy_corpus(base)
    selected_ids = [
        document_id
        for document_id in sorted(base, key=_document_sort_key)
        if (
            1000.0
            * len(base[document_id])
            / max(
                1,
                len(
                    (input_dir / f"{document_id}.txt").read_text("utf-8")
                ),
            )
            < density_threshold
        )
    ]
    icd_index = ICDIndex.load(repository / "artifacts/icd_index.json")
    rxnorm_index = RxNormIndex(
        repository / "artifacts/rxnorm_full100_cache.json",
        use_api=False,
    )
    detector = AssertionDetector()
    per_document: dict[str, Any] = {}
    for document_id in selected_ids:
        document = Document(
            id=document_id,
            text=(input_dir / f"{document_id}.txt").read_text("utf-8"),
        )
        existing = base[document_id]
        reconstructed: list[dict[str, Any]] = []

        def validate_recovery(response: LunaExtractionResponse) -> None:
            nonlocal reconstructed
            reconstructed = reconstruct_luna_entities(document, response)

        recovery = runner.run_structured(
            task="recovery_extract",
            document_id=document_id,
            document_text=document.text,
            prompt=extraction_prompt(
                document,
                existing,
                recovery_only=True,
            ),
            response_type=LunaExtractionResponse,
            validator=validate_recovery,
        )
        existing_keys = {
            (row["position"][0], row["position"][1], row["type"])
            for row in existing
        }
        additions = [
            row
            for row in reconstructed
            if (
                (
                    row["position"][0],
                    row["position"][1],
                    row["type"],
                )
                not in existing_keys
                and not _has_overlap(row, existing)
            )
        ]
        combined = [
            {
                "text": row["text"],
                "type": row["type"],
                "position": list(row["position"]),
            }
            for row in existing
        ] + additions
        combined.sort(
            key=lambda row: (
                row["position"][0],
                row["position"][1],
                row["type"],
            )
        )
        candidates = _candidate_rows(document, existing, combined)
        if candidates:
            merged: list[dict[str, Any]] = []

            def validate_recovery_audit(
                response: LunaAuditResponse,
            ) -> None:
                nonlocal merged
                if any(row.source != "luna" for row in candidates):
                    raise ValueError("miss recovery attempted a removal")
                merged = merge_luna_inventory(
                    document,
                    existing,
                    combined,
                    candidates,
                    response,
                )

            audit = runner.run_structured(
                task="recovery_audit",
                document_id=document_id,
                document_text=document.text,
                prompt=audit_prompt(document, candidates),
                response_type=LunaAuditResponse,
                validator=validate_recovery_audit,
            )
        else:
            audit = LunaAuditResponse(document_id=document_id, decisions=[])
            merged = list(existing)
        rows = apply_metadata(
            document,
            merged,
            existing,
            icd_index=icd_index,
            rxnorm_index=rxnorm_index,
            assertion_detector=detector,
        )
        density = 1000.0 * len(rows) / max(1, len(document.text))
        if density > MAX_DENSITY_PER_1000:
            raise ValueError(
                f"recovered document {document_id} exceeds density guard"
            )
        variant[document_id] = rows
        atomic_write_json(
            run_dir / "documents" / document_id / "luna_recovery.json",
            recovery.model_dump(mode="json"),
        )
        atomic_write_json(
            run_dir / "documents" / document_id / "luna_recovery_audit.json",
            audit.model_dump(mode="json"),
        )
        per_document[document_id] = {
            "base_entities": len(existing),
            "proposed_missing": len(reconstructed),
            "eligible_nonoverlapping_additions": len(additions),
            "output_entities": len(rows),
            "added": len(rows) - len(existing),
        }
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(variant, key=_document_sort_key):
        atomic_write_json(
            output_dir / f"{document_id}.json",
            variant[document_id],
        )
    validate_output_directory(output_dir, input_dir, expected_stems=set(base))
    total = sum(len(rows) for rows in variant.values())
    if not FULL_MIN_ENTITIES <= total <= FULL_MAX_ENTITIES:
        raise ValueError("recovered inventory falls outside full entity guard")
    changes = _inventory_change_manifest(base, variant)
    manifest = {
        "kind": "luna_miss_only_recovery",
        "preflight": preflight,
        "density_threshold": density_threshold,
        "selected_documents": selected_ids,
        "entities": total,
        "changes": changes,
        "per_document": per_document,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest


def run_precision_prune(
    *,
    repository: Path,
    input_dir: Path,
    baseline_output: Path,
    luna_run_dir: Path,
    run_dir: Path,
    codex_bin: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "high",
    max_retries: int = 2,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    runner = CodexLunaRunner(
        repository=repository,
        run_dir=run_dir,
        model=model,
        reasoning_effort=reasoning_effort,
        codex_bin=codex_bin,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    preflight = runner.preflight_model()
    baseline = load_output_directory(baseline_output)
    variant = _copy_corpus(baseline)
    per_document: dict[str, Any] = {}
    for document_id in sorted(baseline, key=_document_sort_key):
        document = Document(
            id=document_id,
            text=(input_dir / f"{document_id}.txt").read_text("utf-8"),
        )
        extraction_path = (
            luna_run_dir
            / "checkpoints"
            / "extract"
            / f"{document_id}.json"
        )
        audit_path = (
            luna_run_dir
            / "checkpoints"
            / "audit"
            / f"{document_id}.json"
        )
        if not extraction_path.exists():
            raise ValueError(f"missing full extraction for document {document_id}")
        extraction = LunaExtractionResponse.model_validate(
            json.loads(extraction_path.read_text("utf-8"))["response"]
        )
        luna = reconstruct_luna_entities(document, extraction)
        original_candidates = _candidate_rows(
            document,
            baseline[document_id],
            luna,
        )
        if original_candidates:
            if not audit_path.exists():
                raise ValueError(f"missing full audit for document {document_id}")
            original_audit = LunaAuditResponse.model_validate(
                json.loads(audit_path.read_text("utf-8"))["response"]
            )
            original_decisions = _decision_map(
                document_id,
                original_candidates,
                original_audit,
            )
        else:
            original_decisions = {}
        eligible = [
            row
            for row in original_candidates
            if (
                row.source == "baseline"
                and not original_decisions[row.id].keep
            )
        ]
        if not eligible:
            per_document[document_id] = {
                "eligible_rows": 0,
                "dropped_rows": 0,
            }
            continue
        focused: dict[str, LunaAuditDecision] = {}

        def validate_prune(response: LunaAuditResponse) -> None:
            nonlocal focused
            focused = _decision_map(document_id, eligible, response)

        runner.run_structured(
            task="prune_audit",
            document_id=document_id,
            document_text=document.text,
            prompt=(
                audit_prompt(document, eligible)
                + "\n\nFOCUSED PRECISION PRUNE: Each supplied baseline row "
                "was omitted by the independent full extraction and rejected "
                "by the first delta audit. Independently reassess it. "
                "keep=false only when it is certainly not a valid competition "
                "entity; otherwise keep=true."
            ),
            response_type=LunaAuditResponse,
            validator=validate_prune,
        )
        dropped_keys = {
            row.key
            for row in eligible
            if not focused[row.id].keep
        }
        variant[document_id] = [
            row
            for row in baseline[document_id]
            if (
                row["position"][0],
                row["position"][1],
                row["type"],
            )
            not in dropped_keys
        ]
        per_document[document_id] = {
            "eligible_rows": len(eligible),
            "dropped_rows": len(dropped_keys),
        }
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(variant, key=_document_sort_key):
        atomic_write_json(
            output_dir / f"{document_id}.json",
            variant[document_id],
        )
    validate_output_directory(
        output_dir,
        input_dir,
        expected_stems=set(baseline),
    )
    changes = _inventory_change_manifest(baseline, variant)
    manifest = {
        "kind": "luna_precision_prune",
        "preflight": preflight,
        "eligibility": (
            "baseline row omitted by full Luna extraction and rejected by "
            "the independent delta audit"
        ),
        "entities": sum(len(rows) for rows in variant.values()),
        "changes": changes,
        "per_document": per_document,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest


def _parse_document_ids(value: str, input_dir: Path) -> list[str]:
    if value == "all":
        return sorted(
            (path.stem for path in input_dir.glob("*.txt")),
            key=_document_sort_key,
        )
    return [part.strip() for part in value.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinical-nlp-luna")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--repository", type=Path, default=Path.cwd())
    preflight.add_argument("--run-dir", type=Path, required=True)
    preflight.add_argument("--codex-bin")
    preflight.add_argument("--model", default=DEFAULT_MODEL)
    preflight.add_argument("--reasoning-effort", default="high")

    run = subparsers.add_parser("run")
    run.add_argument("--repository", type=Path, default=Path.cwd())
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--baseline-output", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--documents", default="all")
    run.add_argument("--codex-bin")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--reasoning-effort", default="high")
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--timeout-seconds", type=float, default=900.0)
    run.add_argument("--full-guard", action="store_true")

    metadata = subparsers.add_parser("submission4-metadata")
    metadata.add_argument("--baseline-output", type=Path, required=True)
    metadata.add_argument("--luna-output", type=Path, required=True)
    metadata.add_argument("--input-dir", type=Path, required=True)
    metadata.add_argument("--output-dir", type=Path, required=True)
    metadata.add_argument("--wer", type=float, required=True)
    metadata.add_argument("--assertions-score", type=float, required=True)
    metadata.add_argument("--candidates-score", type=float, required=True)
    metadata.add_argument("--final-score", type=float, required=True)

    recovery = subparsers.add_parser("submission4-recover")
    recovery.add_argument("--repository", type=Path, default=Path.cwd())
    recovery.add_argument("--input-dir", type=Path, required=True)
    recovery.add_argument("--base-output", type=Path, required=True)
    recovery.add_argument("--run-dir", type=Path, required=True)
    recovery.add_argument("--density-threshold", type=float, default=18.0)
    recovery.add_argument("--codex-bin")
    recovery.add_argument("--model", default=DEFAULT_MODEL)
    recovery.add_argument("--reasoning-effort", default="high")
    recovery.add_argument("--max-retries", type=int, default=2)
    recovery.add_argument("--timeout-seconds", type=float, default=900.0)

    prune = subparsers.add_parser("submission4-prune")
    prune.add_argument("--repository", type=Path, default=Path.cwd())
    prune.add_argument("--input-dir", type=Path, required=True)
    prune.add_argument("--baseline-output", type=Path, required=True)
    prune.add_argument("--luna-run-dir", type=Path, required=True)
    prune.add_argument("--run-dir", type=Path, required=True)
    prune.add_argument("--codex-bin")
    prune.add_argument("--model", default=DEFAULT_MODEL)
    prune.add_argument("--reasoning-effort", default="high")
    prune.add_argument("--max-retries", type=int, default=2)
    prune.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        runner = CodexLunaRunner(
            repository=args.repository,
            run_dir=args.run_dir,
            codex_bin=args.codex_bin,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        result = runner.preflight_model()
    elif args.command == "run":
        result = run_luna_consensus(
            repository=args.repository,
            input_dir=args.input_dir,
            baseline_output=args.baseline_output,
            run_dir=args.run_dir,
            document_ids=_parse_document_ids(args.documents, args.input_dir),
            codex_bin=args.codex_bin,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
            enforce_full_guard=args.full_guard,
        )
    elif args.command == "submission4-metadata":
        result = build_score_adjusted_metadata_variant(
            baseline_output=args.baseline_output,
            luna_output=args.luna_output,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            metrics=LunaLeaderboardMetrics(
                wer=args.wer,
                assertions_score=args.assertions_score,
                candidates_score=args.candidates_score,
                final_score=args.final_score,
            ),
        )
    elif args.command == "submission4-recover":
        result = run_miss_only_recovery(
            repository=args.repository,
            input_dir=args.input_dir,
            base_output=args.base_output,
            run_dir=args.run_dir,
            density_threshold=args.density_threshold,
            codex_bin=args.codex_bin,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        result = run_precision_prune(
            repository=args.repository,
            input_dir=args.input_dir,
            baseline_output=args.baseline_output,
            luna_run_dir=args.luna_run_dir,
            run_dir=args.run_dir,
            codex_bin=args.codex_bin,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
