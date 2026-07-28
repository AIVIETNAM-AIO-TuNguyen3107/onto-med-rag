"""Convert direct-extraction raw batches into linked competition outputs.

The raw batches carry ``text`` plus a 1-based ``occurrence`` rather than character
offsets, so that the spans survive round-tripping through a model without the
offsets ever being guessed. This module reconstructs the offsets host-side with
:func:`clinical_nlp.text.find_occurrence`, then fills ICD and RxNorm candidates
with the deterministic linker only — no LLM is involved at any point.

Assertions come through from the raw batches verbatim. They were authored by hand
against the organizer conventions, so re-deriving them from
:class:`AssertionDetector` would silently discard that work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

import re

from clinical_nlp.atomic import atomic_write_json
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.normalization import normalize_search
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.schemas import Assertion, Document, Entity, EntityType, LinkCandidate
from clinical_nlp.text import find_occurrence, is_masked_span
from clinical_nlp.validation import validate_entities

# Mirrors LinkingConfig defaults in config.py. Duplicated rather than imported
# because instantiating PipelineConfig would pull in an LLM backend we do not use.
ICD_MIN_SCORE = 0.55
RXNORM_MIN_SCORE = 0.45
AUTO_EXACT_SCORE = 0.90
AUTO_SINGLE_SCORE = 0.70
AUTO_TOP_SCORE = 0.75
AUTO_SCORE_MARGIN = 0.25
RETRIEVAL_CANDIDATES = 20
CandidatePolicy = Literal["conservative", "top-or-drop", "top-or-empty"]

_EXACT_SOURCES = {"exact", "rxnav_exact_or_normalized"}

# Route, frequency and packaging noise that survives the alphabetic-token filter
# and must not be mistaken for an ingredient name.
_NOT_AN_INGREDIENT = frozenset(
    {
        "once",
        "daily",
        "nebs",
        "gram",
        "grams",
        "tablet",
        "oral",
        "inject",
        "injection",
        "capsule",
        "solution",
        "liều",
        "viên",
        "ống",
        "ngày",
        "lần",
    }
)

_STRENGTH_UNITS = {
    "mg": ("MG", 1.0),
    "milligram": ("MG", 1.0),
    "g": ("MG", 1000.0),
    "gram": ("MG", 1000.0),
    "mcg": ("MCG", 1.0),
    "microgam": ("MCG", 1.0),
    "microgram": ("MCG", 1.0),
}
_STRENGTH_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|milligram|gram|g|mcg|microgam|microgram)\b",
    re.IGNORECASE,
)


def _format_strength(value: float) -> str:
    return f"{value:g}"


class LocalRxNormCatalog:
    """Offline RxNorm matcher over a local ``in``/``scd``/``sbd`` crawl.

    RxNav's approximate matcher returns nothing at all for sig-bearing spans such
    as ``metoprolol 25mg po bid``, so the online path can only ever reach
    ingredient-level codes. Matching ingredient plus strength against a local
    catalog is what makes the SCD codes the organizer uses reachable.

    Selection is tiered most-specific-first — SCD, then SBD, then ingredient —
    and emits a single identifier, matching the one-code-per-medication shape of
    the organizer's own example. Anything ambiguous yields nothing.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.concepts: dict[str, dict[str, str]] = {}
        rows: list[dict[str, str]] = []
        for term_type in ("in", "scd", "sbd"):
            path = directory / f"{term_type}.json"
            if not path.exists():
                raise ValueError(f"missing RxNorm crawl file {path}")
            rows.extend(json.loads(path.read_text("utf-8")))
        for row in rows:
            self.concepts[row["rxcui"]] = row
        self.ingredients = [row for row in rows if row["tty"] == "IN"]
        self.branded = [row for row in rows if row["tty"] == "SBD"]
        # A combination product separates its active moieties with "/". Matching
        # a single-ingredient span against one would code the wrong concept.
        self.single_clinical = [
            row
            for row in rows
            if row["tty"] == "SCD" and "/" not in row["name"].split(" Oral")[0]
        ]
        self._by_ingredient_name = {
            normalize_search(row["name"]): row["rxcui"] for row in self.ingredients
        }

    @staticmethod
    def _tokens(span: str) -> list[str]:
        return [
            word
            for word in re.findall(r"[a-zA-Z]{4,}", span.lower())
            if word not in _NOT_AN_INGREDIENT
        ]

    @staticmethod
    def _strength(span: str) -> list[tuple[str, str]]:
        """Return the strength as ``(value, unit)`` pairs worth searching for.

        RxNorm is inconsistent about micrograms — levothyroxine is named in MG
        (``0.075 MG``) even where the prescription says 75 microgam — so a MCG
        reading also yields its MG equivalent.
        """
        match = _STRENGTH_RE.search(span)
        if match is None:
            return []
        value = float(match.group(1).replace(",", ".")) * _STRENGTH_UNITS[
            match.group(2).lower()
        ][1]
        unit = _STRENGTH_UNITS[match.group(2).lower()][0]
        readings = [(_format_strength(value), unit)]
        if unit == "MCG":
            readings.append((_format_strength(value / 1000.0), "MG"))
        return readings

    @staticmethod
    def _matches_strength(name: str, readings: list[tuple[str, str]]) -> bool:
        return any(
            re.search(rf"(?<![\d.]){re.escape(value)}\s*{unit}\b", name, re.IGNORECASE)
            for value, unit in readings
        )

    def _clinical_drug(
        self, tokens: list[str], readings: list[tuple[str, str]]
    ) -> list[str]:
        hits = [
            row
            for row in self.single_clinical
            if self._matches_strength(row["name"], readings)
            and any(token in row["name"].lower() for token in tokens)
        ]
        if len(hits) == 1:
            return [hits[0]["rxcui"]]
        if not hits:
            return []
        # Several forms of the same drug: take the plain oral tablet, which is
        # the form the organizer's example uses (amlodipine 10 MG Oral Tablet).
        plain = [
            row
            for row in hits
            if any(
                re.fullmatch(
                    rf"[a-z ]+ {re.escape(value)} {unit} Oral Tablet",
                    row["name"],
                    re.IGNORECASE,
                )
                for value, unit in readings
            )
        ]
        return [plain[0]["rxcui"]] if len(plain) == 1 else []

    def _branded_drug(
        self, tokens: list[str], readings: list[tuple[str, str]]
    ) -> list[str]:
        hits = [
            row
            for row in self.branded
            if self._matches_strength(row["name"], readings)
            and any(f"[{token}" in row["name"].lower() for token in tokens)
        ]
        return [hits[0]["rxcui"]] if len(hits) == 1 else []

    def match(self, span: str) -> list[str]:
        tokens = self._tokens(span)
        if not tokens:
            return []
        readings = self._strength(span)
        if readings:
            for tier in (self._clinical_drug, self._branded_drug):
                found = tier(tokens, readings)
                if found:
                    return found
        identifier = self._by_ingredient_name.get(normalize_search(span))
        return [identifier] if identifier else []


class RawEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    occurrence: int = Field(ge=1)
    type: EntityType
    assertions: list[Assertion] = Field(default_factory=list)


class RawDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    entities: list[RawEntity]


class OccurrenceOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    text: str = Field(min_length=1)
    type: EntityType
    from_occurrence: int = Field(ge=1)
    to_occurrence: int = Field(ge=1)
    expected_position: tuple[int, int]
    rationale: str = Field(min_length=1)


class CandidateOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    text: str = Field(min_length=1)
    type: EntityType
    position: tuple[int, int]
    candidates: list[str] = Field(min_length=1, max_length=1)
    rationale: str = Field(min_length=1)


class CorrectionOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    occurrence_overrides: list[OccurrenceOverride] = Field(default_factory=list)
    candidate_overrides: list[CandidateOverride] = Field(default_factory=list)


def load_correction_overlay(path: Path | None) -> CorrectionOverlay:
    if path is None:
        return CorrectionOverlay()
    return CorrectionOverlay.model_validate(json.loads(path.read_text("utf-8")))


def apply_occurrence_overrides(
    document_id: str,
    raw_entities: list[RawEntity],
    overrides: list[OccurrenceOverride],
) -> tuple[list[RawEntity], list[dict[str, Any]]]:
    """Apply document-scoped occurrence corrections without mutating raw batches."""
    adjusted = list(raw_entities)
    applied: list[dict[str, Any]] = []
    for override in overrides:
        if override.document_id != document_id:
            continue
        matching = [
            index
            for index, entity in enumerate(adjusted)
            if entity.text == override.text
            and entity.type == override.type
            and entity.occurrence == override.from_occurrence
        ]
        if len(matching) != 1:
            raise ValueError(
                "occurrence override must match exactly one raw entity: "
                f"document={document_id!r} text={override.text!r} "
                f"matches={len(matching)}"
            )
        index = matching[0]
        adjusted[index] = adjusted[index].model_copy(
            update={"occurrence": override.to_occurrence}
        )
        applied.append(
            {
                "kind": "occurrence",
                "document_id": document_id,
                "text": override.text,
                "type": override.type.value,
                "from_occurrence": override.from_occurrence,
                "to_occurrence": override.to_occurrence,
                "expected_position": list(override.expected_position),
                "rationale": override.rationale,
            }
        )
    return adjusted, applied


def load_raw_batches(raw_dir: Path) -> dict[str, list[RawEntity]]:
    """Read every batch file in ``raw_dir``, keyed by document id."""
    batches = sorted(raw_dir.glob("*.json"))
    if not batches:
        raise ValueError(f"no raw batch files under {raw_dir}")
    documents: dict[str, list[RawEntity]] = {}
    for path in batches:
        payload = json.loads(path.read_text("utf-8"))
        for row in payload:
            parsed = RawDocument.model_validate(row)
            if parsed.document_id in documents:
                raise ValueError(
                    f"document {parsed.document_id!r} appears in more than one batch"
                )
            documents[parsed.document_id] = parsed.entities
    return documents


def reconstruct(
    document: Document,
    raw_entities: list[RawEntity],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve ``(text, occurrence)`` pairs to offsets against ``document``.

    Returns the located rows sorted by position, and the rows that could not be
    resolved. Unresolvable spans are reported, never repaired — a span whose
    offsets we cannot derive is a span we do not know.
    """
    located: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw in raw_entities:
        try:
            start, end = find_occurrence(document.text, raw.text, raw.occurrence)
        except ValueError as error:
            failures.append(
                {
                    "document_id": document.id,
                    "text": raw.text,
                    "occurrence": raw.occurrence,
                    "type": raw.type.value,
                    "reason": str(error),
                }
            )
            continue
        if document.text[start:end] != raw.text:
            failures.append(
                {
                    "document_id": document.id,
                    "text": raw.text,
                    "occurrence": raw.occurrence,
                    "type": raw.type.value,
                    "reason": "resolved slice is not an exact match",
                    "resolved": document.text[start:end],
                }
            )
            continue
        located.append(
            {
                "text": document.text[start:end],
                "type": raw.type.value,
                "assertions": [item.value for item in raw.assertions],
                "position": [start, end],
            }
        )
    located.sort(key=lambda row: (row["position"][0], row["position"][1], row["type"]))
    return located, failures


def _is_exact_candidate(candidate: LinkCandidate) -> bool:
    return any(
        source in _EXACT_SOURCES or "exact_or_normalized" in source
        for source in candidate.retrieval_sources
    )


def _eligible_candidates(
    candidates: list[LinkCandidate],
    threshold: float,
) -> list[LinkCandidate]:
    return [
        row
        for row in candidates
        if _is_exact_candidate(row) or row.score >= threshold
    ]


def automatic_selection(candidates: list[LinkCandidate]) -> list[str]:
    """Pick a single identifier, or none at all.

    Mirrors ``Pipeline._automatic_candidate_selection``. Both of its ``None``
    outcomes — ambiguous, and the ICD escalation path — collapse to no code here,
    because the escalation target is an LLM this run deliberately does not use.
    A wrong code scores zero and so does an empty one, but a wrong code also
    costs the concept its Jaccard union.
    """
    if not candidates:
        return []
    top = candidates[0]
    if _is_exact_candidate(top) and top.score >= AUTO_EXACT_SCORE:
        return [top.identifier]
    if len(candidates) == 1 and top.score >= AUTO_SINGLE_SCORE:
        return [top.identifier]
    if (
        len(candidates) >= 2
        and top.score >= AUTO_TOP_SCORE
        and top.score - candidates[1].score >= AUTO_SCORE_MARGIN
    ):
        return [top.identifier]
    return []


def select_retrieved_candidates(
    candidates: list[LinkCandidate],
    *,
    threshold: float,
    policy: CandidatePolicy,
) -> tuple[list[str], str, LinkCandidate | None]:
    """Select conservatively, or force the top retrieved row when requested."""
    selected = automatic_selection(_eligible_candidates(candidates, threshold))
    if selected:
        top = next(row for row in candidates if row.identifier == selected[0])
        return selected, "confident", top
    if policy in {"top-or-drop", "top-or-empty"} and candidates:
        return [candidates[0].identifier], "forced_top", candidates[0]
    return [], "no_candidate", candidates[0] if candidates else None


def _candidate_details(candidate: LinkCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "identifier": candidate.identifier,
        "name": candidate.name,
        "terminology_type": candidate.terminology_type,
        "score": candidate.score,
        "retrieval_sources": list(candidate.retrieval_sources),
    }


def _validate_candidate_identifiers(
    *,
    entity_type: EntityType,
    candidates: list[str],
    icd_index: ICDIndex,
    rxnorm_catalog: LocalRxNormCatalog,
    rxnorm_fallback: RxNormIndex | None,
) -> None:
    if len(candidates) > 1:
        raise ValueError("strict direct extraction allows at most one candidate")
    if entity_type == EntityType.DIAGNOSIS:
        invalid = [value for value in candidates if not icd_index.contains(value)]
        if invalid:
            raise ValueError(f"unknown ICD-10 candidates: {invalid}")
    elif entity_type == EntityType.MEDICATION:
        invalid = [
            value
            for value in candidates
            if not value.isascii()
            or not value.isdigit()
            or (
                value not in rxnorm_catalog.concepts
                and (rxnorm_fallback is None or not rxnorm_fallback.contains(value))
            )
        ]
        if invalid:
            raise ValueError(f"unknown RxNorm candidates: {invalid}")


def link_rows(
    rows: list[dict[str, Any]],
    *,
    icd_index: ICDIndex,
    rxnorm_catalog: LocalRxNormCatalog,
    rxnorm_fallback: RxNormIndex | None = None,
    cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    candidate_policy: CandidatePolicy = "conservative",
    document_id: str | None = None,
    candidate_overrides: dict[
        tuple[str, str, int, int], CandidateOverride
    ] | None = None,
    audit: dict[str, list[dict[str, Any]]] | None = None,
) -> list[Entity]:
    """Attach candidates to reconstructed rows and build validated entities."""
    memo = cache if cache is not None else {}
    overrides = candidate_overrides or {}
    entities: list[Entity] = []
    for row in rows:
        entity_type = EntityType(row["type"])
        candidates: list[str] = []
        selection = "not_linkable"
        selected_detail: dict[str, Any] | None = None
        if entity_type in {EntityType.DIAGNOSIS, EntityType.MEDICATION}:
            key = (entity_type.value, row["text"])
            if key not in memo:
                if entity_type == EntityType.DIAGNOSIS:
                    retrieved = icd_index.retrieve(
                        row["text"], limit=RETRIEVAL_CANDIDATES
                    )
                    found, selection, detail = select_retrieved_candidates(
                        retrieved,
                        threshold=ICD_MIN_SCORE,
                        policy=candidate_policy,
                    )
                else:
                    found = rxnorm_catalog.match(row["text"])
                    if found:
                        concept = rxnorm_catalog.concepts[found[0]]
                        selection = "confident"
                        detail = LinkCandidate(
                            identifier=found[0],
                            name=concept["name"],
                            terminology_type=concept["tty"],
                            score=1.0,
                            retrieval_sources=["local_rxnorm_catalog"],
                        )
                    elif rxnorm_fallback is not None:
                        # The crawl carries no BN concepts, so a bare brand
                        # mention such as "seroquel" is only reachable through
                        # the previously cached RxNorm index.
                        retrieved = rxnorm_fallback.retrieve(
                            row["text"], limit=RETRIEVAL_CANDIDATES
                        )
                        found, selection, detail = select_retrieved_candidates(
                            retrieved,
                            threshold=RXNORM_MIN_SCORE,
                            policy=candidate_policy,
                        )
                    else:
                        found = []
                        selection = "no_candidate"
                        detail = None
                memo[key] = {
                    "candidates": list(found),
                    "selection": selection,
                    "detail": _candidate_details(detail),
                }
            decision = memo[key]
            candidates = list(decision["candidates"])
            selection = str(decision["selection"])
            selected_detail = decision["detail"]

            if document_id is not None:
                start, end = row["position"]
                override_key = (document_id, entity_type.value, start, end)
                override = overrides.get(override_key)
                if override is not None:
                    if override.text != row["text"]:
                        raise ValueError(
                            "candidate override text does not match reconstructed "
                            f"source at {override_key}: {override.text!r} != "
                            f"{row['text']!r}"
                        )
                    original_candidates = list(candidates)
                    candidates = list(override.candidates)
                    selection = "override"
                    selected_detail = None
                    if audit is not None:
                        audit["overrides"].append(
                            {
                                "kind": "candidate",
                                "document_id": document_id,
                                "text": row["text"],
                                "type": entity_type.value,
                                "position": list(row["position"]),
                                "from_candidates": original_candidates,
                                "to_candidates": list(candidates),
                                "rationale": override.rationale,
                            }
                        )

            _validate_candidate_identifiers(
                entity_type=entity_type,
                candidates=candidates,
                icd_index=icd_index,
                rxnorm_catalog=rxnorm_catalog,
                rxnorm_fallback=rxnorm_fallback,
            )
            if candidate_policy == "top-or-drop" and not candidates:
                if audit is not None:
                    audit["dropped"].append(
                        {
                            "document_id": document_id,
                            "text": row["text"],
                            "type": entity_type.value,
                            "position": list(row["position"]),
                            "assertions": list(row["assertions"]),
                            "removed_assertions": len(row["assertions"]),
                            "reason": "no_retrieved_candidate",
                            "top_retrieved": selected_detail,
                        }
                    )
                continue
            if candidate_policy == "top-or-empty" and not candidates:
                if audit is not None:
                    audit["retained_unlinked"].append(
                        {
                            "document_id": document_id,
                            "text": row["text"],
                            "type": entity_type.value,
                            "position": list(row["position"]),
                            "assertions": list(row["assertions"]),
                            "reason": "no_retrieved_candidate",
                        }
                    )
            if selection == "forced_top" and audit is not None:
                audit["forced_candidates"].append(
                    {
                        "document_id": document_id,
                        "text": row["text"],
                        "type": entity_type.value,
                        "position": list(row["position"]),
                        "candidates": list(candidates),
                        "top_retrieved": selected_detail,
                    }
                )
        entities.append(
            Entity(
                text=row["text"],
                type=entity_type,
                candidates=candidates,
                assertions=[Assertion(value) for value in row["assertions"]],
                position=(row["position"][0], row["position"][1]),
            )
        )
    return entities


def serialize_entity(
    entity: Entity,
    *,
    omit_nonlinkable_candidates: bool,
) -> dict[str, Any]:
    return entity.output_dict(
        omit_nonlinkable_candidates=omit_nonlinkable_candidates
    )


def validate_strict_output_rows(
    rows: list[dict[str, Any]],
    *,
    require_linked_candidates: bool = True,
) -> None:
    """Validate the v2 conditional candidate-field contract."""
    linkable_types = {
        EntityType.DIAGNOSIS.value,
        EntityType.MEDICATION.value,
    }
    assertion_types = linkable_types | {EntityType.SYMPTOM.value}
    for row in rows:
        entity_type = row["type"]
        expected_keys = {"text", "type", "assertions", "position"}
        if entity_type in linkable_types:
            expected_keys.add("candidates")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) > 1:
                raise ValueError(
                    "strict linkable entities require at most one candidate"
                )
            if require_linked_candidates and len(candidates) != 1:
                raise ValueError(
                    "strict linkable entities require exactly one candidate"
                )
        elif "candidates" in row:
            raise ValueError(
                "strict non-linkable entities must omit the candidates field"
            )
        if set(row) != expected_keys:
            raise ValueError(
                f"strict output keys do not match for {entity_type}: "
                f"{sorted(row)}"
            )
        assertions = row["assertions"]
        if not isinstance(assertions, list):
            raise ValueError("strict assertions field must be an array")
        if entity_type not in assertion_types and assertions:
            raise ValueError("strict lab entities cannot carry assertions")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(
    run_id: str,
    input_dir: Path,
    ids: list[str],
    *,
    candidate_policy: CandidatePolicy,
    omit_nonlinkable_candidates: bool,
    correction_path: Path | None,
) -> dict[str, Any]:
    inputs = []
    for document_id in ids:
        path = input_dir / f"{document_id}.txt"
        payload = path.read_bytes()
        inputs.append(
            {
                "id": document_id,
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "inputs": inputs,
        "selection": {"document_ids": ids},
        "models": {"extraction": "direct", "linking": "deterministic"},
        "direct_extraction": {
            "candidate_policy": candidate_policy,
            "omit_nonlinkable_candidates": omit_nonlinkable_candidates,
            "correction_overlay": (
                {
                    "path": str(correction_path),
                    "sha256": _sha256_file(correction_path),
                }
                if correction_path is not None
                else None
            ),
        },
    }


def convert(
    *,
    raw_dir: Path,
    input_dir: Path,
    run_dir: Path,
    icd_index_path: Path,
    rxnorm_dir: Path,
    audit_path: Path,
    rxnorm_cache_path: Path | None = None,
    candidate_policy: CandidatePolicy = "conservative",
    correction_path: Path | None = None,
    omit_nonlinkable_candidates: bool = False,
) -> dict[str, Any]:
    raw_documents = load_raw_batches(raw_dir)
    corrections = load_correction_overlay(correction_path)
    document_ids = sorted(raw_documents, key=int)
    missing = [
        document_id
        for document_id in document_ids
        if not (input_dir / f"{document_id}.txt").exists()
    ]
    if missing:
        raise ValueError(f"raw batches name documents without inputs: {missing}")

    icd_index = ICDIndex.load(icd_index_path)
    rxnorm_catalog = LocalRxNormCatalog(rxnorm_dir)
    rxnorm_fallback = (
        RxNormIndex(rxnorm_cache_path, use_api=False)
        if rxnorm_cache_path is not None and rxnorm_cache_path.exists()
        else None
    )
    candidate_overrides: dict[
        tuple[str, str, int, int], CandidateOverride
    ] = {}
    for override in corrections.candidate_overrides:
        start, end = override.position
        key = (override.document_id, override.type.value, start, end)
        if key in candidate_overrides:
            raise ValueError(f"duplicate candidate override: {key}")
        candidate_overrides[key] = override

    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    link_cache: dict[tuple[str, str], dict[str, Any]] = {}
    audit: dict[str, list[dict[str, Any]]] = {
        "dropped": [],
        "forced_candidates": [],
        "retained_unlinked": [],
        "overrides": [],
    }
    per_document: dict[str, dict[str, int]] = {}
    term_types: dict[str, int] = {}
    totals = {
        "entities": 0,
        "assertions": 0,
        "coded_icd": 0,
        "coded_rxnorm": 0,
        "masked": 0,
    }

    for document_id in document_ids:
        document = Document(
            id=document_id,
            text=(input_dir / f"{document_id}.txt").read_text("utf-8"),
        )
        raw_entities, occurrence_audit = apply_occurrence_overrides(
            document_id,
            raw_documents[document_id],
            corrections.occurrence_overrides,
        )
        rows, document_failures = reconstruct(document, raw_entities)
        failures.extend(document_failures)
        for record in occurrence_audit:
            expected = record["expected_position"]
            matching = [
                row
                for row in rows
                if row["text"] == record["text"]
                and row["type"] == record["type"]
                and row["position"] == expected
            ]
            if len(matching) != 1:
                raise ValueError(
                    "occurrence override did not resolve to its expected source "
                    f"position: {record}"
                )
            record["resolved_position"] = list(matching[0]["position"])
            audit["overrides"].append(record)
        entities = link_rows(
            rows,
            icd_index=icd_index,
            rxnorm_catalog=rxnorm_catalog,
            rxnorm_fallback=rxnorm_fallback,
            cache=link_cache,
            candidate_policy=candidate_policy,
            document_id=document_id,
            candidate_overrides=candidate_overrides,
            audit=audit,
        )
        validate_entities(document, entities)
        payload = [
            serialize_entity(
                entity,
                omit_nonlinkable_candidates=omit_nonlinkable_candidates,
            )
            for entity in entities
        ]
        if omit_nonlinkable_candidates:
            validate_strict_output_rows(
                payload,
                require_linked_candidates=candidate_policy == "top-or-drop",
            )
        atomic_write_json(
            output_dir / f"{document_id}.json",
            payload,
        )
        coded_icd = sum(
            1
            for entity in entities
            if entity.type == EntityType.DIAGNOSIS and entity.candidates
        )
        coded_rxnorm = sum(
            1
            for entity in entities
            if entity.type == EntityType.MEDICATION and entity.candidates
        )
        for entity in entities:
            if entity.type != EntityType.MEDICATION:
                continue
            for identifier in entity.candidates or []:
                concept = rxnorm_catalog.concepts.get(identifier)
                if concept is None and rxnorm_fallback is not None:
                    concept = rxnorm_fallback.concepts.get(identifier)
                term_type = (concept or {}).get("tty", "?")
                term_types[term_type] = term_types.get(term_type, 0) + 1
        per_document[document_id] = {
            "entities": len(entities),
            "assertions": sum(len(entity.assertions) for entity in entities),
            "coded_icd": coded_icd,
            "coded_rxnorm": coded_rxnorm,
        }
        totals["entities"] += len(entities)
        totals["assertions"] += per_document[document_id]["assertions"]
        totals["coded_icd"] += coded_icd
        totals["coded_rxnorm"] += coded_rxnorm
        totals["masked"] += sum(
            1 for entity in entities if is_masked_span(entity.text)
        )

    applied_occurrence_overrides = [
        row for row in audit["overrides"] if row["kind"] == "occurrence"
    ]
    applied_candidate_overrides = [
        row for row in audit["overrides"] if row["kind"] == "candidate"
    ]
    if len(applied_occurrence_overrides) != len(corrections.occurrence_overrides):
        raise ValueError(
            "not every occurrence override was applied exactly once: "
            f"expected={len(corrections.occurrence_overrides)} "
            f"actual={len(applied_occurrence_overrides)}"
        )
    if len(applied_candidate_overrides) != len(corrections.candidate_overrides):
        raise ValueError(
            "not every candidate override was applied exactly once: "
            f"expected={len(corrections.candidate_overrides)} "
            f"actual={len(applied_candidate_overrides)}"
        )

    atomic_write_json(
        run_dir / "source_manifest.json",
        _source_manifest(
            run_dir.name,
            input_dir,
            document_ids,
            candidate_policy=candidate_policy,
            omit_nonlinkable_candidates=omit_nonlinkable_candidates,
            correction_path=correction_path,
        ),
    )

    raw_total = sum(len(rows) for rows in raw_documents.values())
    dropped_assertions = sum(
        row["removed_assertions"] for row in audit["dropped"]
    )
    accounted = (
        totals["entities"] + len(audit["dropped"]) + len(failures) == raw_total
    )
    audit_payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "policy": {
            "candidate_policy": candidate_policy,
            "omit_nonlinkable_candidates": omit_nonlinkable_candidates,
        },
        "sources": {
            "raw_batches": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in sorted(raw_dir.glob("*.json"))
            ],
            "correction_overlay": (
                {
                    "path": str(correction_path),
                    "bytes": correction_path.stat().st_size,
                    "sha256": _sha256_file(correction_path),
                }
                if correction_path is not None
                else None
            ),
        },
        "counts": {
            "raw_entities": raw_total,
            "written_entities": totals["entities"],
            "unresolved": len(failures),
            "dropped": len(audit["dropped"]),
            "dropped_assertions": dropped_assertions,
            "forced_candidates": len(audit["forced_candidates"]),
            "retained_unlinked": len(audit["retained_unlinked"]),
            "overrides": len(audit["overrides"]),
            "accounted": accounted,
        },
        "unresolved": failures,
        "dropped": audit["dropped"],
        "forced_candidates": audit["forced_candidates"],
        "retained_unlinked": audit["retained_unlinked"],
        "overrides": audit["overrides"],
    }
    atomic_write_json(audit_path, audit_payload)

    return {
        "run_dir": str(run_dir),
        "documents": len(document_ids),
        "raw_entities": raw_total,
        "written_entities": totals["entities"],
        "lossless": raw_total == totals["entities"],
        "accounted": accounted,
        "assertions": totals["assertions"],
        "coded_icd": totals["coded_icd"],
        "coded_rxnorm": totals["coded_rxnorm"],
        "coded_total": totals["coded_icd"] + totals["coded_rxnorm"],
        "rxnorm_term_types": dict(sorted(term_types.items())),
        "masked_spans": totals["masked"],
        "unresolved": len(failures),
        "dropped": len(audit["dropped"]),
        "dropped_assertions": dropped_assertions,
        "forced_candidates": len(audit["forced_candidates"]),
        "retained_unlinked": len(audit["retained_unlinked"]),
        "overrides": len(audit["overrides"]),
        "per_document": per_document,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinical-nlp-direct")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument(
        "--raw-dir", type=Path, default=Path("artifacts/direct-extraction/raw")
    )
    convert_parser.add_argument("--input-dir", type=Path, default=Path("input"))
    convert_parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/direct-extraction-v1")
    )
    convert_parser.add_argument(
        "--icd-index", type=Path, default=Path("artifacts/icd_index.json")
    )
    convert_parser.add_argument(
        "--rxnorm-dir",
        type=Path,
        default=Path("rxnorm"),
        help="directory holding the in/scd/sbd RxNorm crawl",
    )
    convert_parser.add_argument(
        "--rxnorm-cache",
        type=Path,
        default=Path("artifacts/rxnorm_direct_extraction_cache.json"),
        help="cached RxNorm index used only for concepts the crawl cannot reach",
    )
    convert_parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/direct-extraction/audit/unresolved.json"),
    )
    convert_parser.add_argument(
        "--candidate-policy",
        choices=("conservative", "top-or-drop", "top-or-empty"),
        default="conservative",
        help=(
            "conservative keeps only high-confidence links; top-or-drop forces "
            "the top retrieved identifier and drops linkable rows with no hit; "
            "top-or-empty forces retrievable rows and keeps no-hit rows empty"
        ),
    )
    convert_parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="optional versioned occurrence/candidate correction overlay",
    )
    convert_parser.add_argument(
        "--omit-nonlinkable-candidates",
        action="store_true",
        help="omit candidates for symptoms and lab entities",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = convert(
        raw_dir=args.raw_dir,
        input_dir=args.input_dir,
        run_dir=args.run_dir,
        icd_index_path=args.icd_index,
        rxnorm_dir=args.rxnorm_dir,
        audit_path=args.audit,
        rxnorm_cache_path=args.rxnorm_cache,
        candidate_policy=args.candidate_policy,
        correction_path=args.corrections,
        omit_nonlinkable_candidates=args.omit_nonlinkable_candidates,
    )
    summary = {key: value for key, value in result.items() if key != "per_document"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
