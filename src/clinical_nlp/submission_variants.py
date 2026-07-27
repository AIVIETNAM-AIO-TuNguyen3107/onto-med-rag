from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from clinical_nlp.atomic import atomic_write_json, atomic_write_text
from clinical_nlp.schemas import Assertion, Entity, EntityType
from clinical_nlp.validation import validate_output_directory


EntityKey = tuple[str, int, int, str]
ReviewField = Literal["candidates", "assertions"]

BASELINE_SCORE = 29.1764
BASELINE_WER = 66.9345
BASELINE_ASSERTIONS = 36.2262
BASELINE_CANDIDATES = 20.9721
SCORE_TOLERANCE = 0.0002


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    position: tuple[int, int]
    type: EntityType
    selected_value: list[str]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_interval(self) -> "ReviewDecision":
        start, end = self.position
        if start < 0 or end <= start:
            raise ValueError("decision position must be a positive interval")
        if len(self.selected_value) != len(set(self.selected_value)):
            raise ValueError("decision selected_value contains duplicates")
        return self


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wer: float = Field(ge=0)
    assertions_score: float = Field(ge=0, le=100)
    candidates_score: float = Field(ge=0, le=100)
    final_score: float | None = Field(default=None, ge=0, le=100)

    @property
    def text_score(self) -> float:
        return 100.0 - self.wer

    @property
    def computed_final_score(self) -> float:
        return (
            0.3 * self.text_score
            + 0.3 * self.assertions_score
            + 0.4 * self.candidates_score
        )

    @model_validator(mode="after")
    def final_matches_components(self) -> "ScoreBreakdown":
        if self.final_score is not None and not math.isclose(
            self.final_score,
            self.computed_final_score,
            abs_tol=SCORE_TOLERANCE,
        ):
            raise ValueError(
                "reported final_score does not match the weighted components"
            )
        return self


class LeaderboardScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: ScoreBreakdown
    candidates: ScoreBreakdown
    assertions: ScoreBreakdown
    gliner: ScoreBreakdown


def _document_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _entity_key(document_id: str, entity: dict[str, Any]) -> EntityKey:
    start, end = entity["position"]
    return (document_id, start, end, entity["type"])


def _decision_key(decision: ReviewDecision) -> EntityKey:
    start, end = decision.position
    return (decision.document_id, start, end, decision.type.value)


def _read_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [Entity.model_validate(row).output_dict() for row in payload]


def load_output_directory(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    files = sorted(
        output_dir.glob("*.json"),
        key=lambda path: _document_sort_key(path.stem),
    )
    if not files:
        raise ValueError(f"no JSON outputs found in {output_dir}")
    return {path.stem: _read_payload(path) for path in files}


def _index_outputs(
    corpus: dict[str, list[dict[str, Any]]],
) -> dict[EntityKey, dict[str, Any]]:
    indexed: dict[EntityKey, dict[str, Any]] = {}
    for document_id, entities in corpus.items():
        for entity in entities:
            key = _entity_key(document_id, entity)
            if key in indexed:
                raise ValueError(f"duplicate entity key: {key}")
            indexed[key] = entity
    return indexed


def _context(text: str, start: int, end: int, width: int) -> dict[str, Any]:
    context_start = max(0, start - width)
    context_end = min(len(text), end + width)
    return {
        "context_start": context_start,
        "context_end": context_end,
        "context": text[context_start:context_end],
        "mention_start_in_context": start - context_start,
        "mention_end_in_context": end - context_start,
    }


def _candidate_artifact_path(
    run_dir: Path,
    document_id: str,
    entity_type: str,
) -> Path:
    filename = (
        "rxnorm_candidates.json"
        if entity_type == EntityType.MEDICATION.value
        else "icd_candidates.json"
    )
    return run_dir / "documents" / document_id / filename


def _candidate_details(
    baseline_run: Path,
    experimental_run: Path,
    key: EntityKey,
) -> list[dict[str, Any]]:
    document_id, start, end, entity_type = key
    details: dict[str, dict[str, Any]] = {}
    for label, run_dir in (
        ("baseline", baseline_run),
        ("experimental", experimental_run),
    ):
        path = _candidate_artifact_path(run_dir, document_id, entity_type)
        if not path.exists():
            continue
        payload = json.loads(path.read_text("utf-8"))
        entry = next(
            (
                row
                for row in payload
                if tuple(row.get("position", ())) == (start, end)
            ),
            None,
        )
        if entry is None:
            continue
        for collection in (
            "retrieved_candidates",
            "eligible_candidates",
            "selected_candidates",
        ):
            for candidate in entry.get(collection, []):
                identifier = candidate["identifier"]
                if identifier not in details:
                    details[identifier] = {
                        "identifier": identifier,
                        "name": candidate.get("name", ""),
                        "terminology_type": candidate.get("terminology_type"),
                        "score": candidate.get("score", 0.0),
                        "retrieval_sources": candidate.get(
                            "retrieval_sources", []
                        ),
                        "observed_in": [],
                    }
                observation = f"{label}:{collection}"
                if observation not in details[identifier]["observed_in"]:
                    details[identifier]["observed_in"].append(observation)
                details[identifier]["score"] = max(
                    float(details[identifier]["score"]),
                    float(candidate.get("score", 0.0)),
                )
    return sorted(details.values(), key=lambda row: row["identifier"])


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write_text(path, content)


def generate_review_packets(
    *,
    baseline_run: Path,
    experimental_run: Path,
    input_dir: Path,
    output_dir: Path,
    context_chars: int = 240,
    expected_candidate_changes: int | None = None,
    expected_assertion_changes: int | None = None,
) -> dict[str, Any]:
    baseline = load_output_directory(baseline_run / "outputs")
    experimental = load_output_directory(experimental_run / "outputs")
    if set(baseline) != set(experimental):
        raise ValueError("baseline and experimental document sets differ")
    baseline_index = _index_outputs(baseline)
    experimental_index = _index_outputs(experimental)
    shared = sorted(
        baseline_index.keys() & experimental_index.keys(),
        key=lambda key: (_document_sort_key(key[0]), key[1], key[2], key[3]),
    )

    candidate_rows: list[dict[str, Any]] = []
    assertion_rows: list[dict[str, Any]] = []
    for key in shared:
        document_id, start, end, entity_type = key
        baseline_entity = baseline_index[key]
        experimental_entity = experimental_index[key]
        text = (input_dir / f"{document_id}.txt").read_text("utf-8")
        common = {
            "document_id": document_id,
            "position": [start, end],
            "type": entity_type,
            "text": baseline_entity["text"],
            **_context(text, start, end, context_chars),
        }
        if baseline_entity["candidates"] != experimental_entity["candidates"]:
            candidate_rows.append(
                {
                    **common,
                    "baseline_value": baseline_entity["candidates"],
                    "experimental_value": experimental_entity["candidates"],
                    "allowed_values": sorted(
                        set(baseline_entity["candidates"])
                        | set(experimental_entity["candidates"])
                    ),
                    "candidate_details": _candidate_details(
                        baseline_run,
                        experimental_run,
                        key,
                    ),
                }
            )
        if baseline_entity["assertions"] != experimental_entity["assertions"]:
            assertion_rows.append(
                {
                    **common,
                    "baseline_value": baseline_entity["assertions"],
                    "experimental_value": experimental_entity["assertions"],
                    "allowed_values": [
                        assertion.value
                        for assertion in Assertion
                        if assertion.value
                        in (
                            set(baseline_entity["assertions"])
                            | set(experimental_entity["assertions"])
                        )
                    ],
                }
            )

    if (
        expected_candidate_changes is not None
        and len(candidate_rows) != expected_candidate_changes
    ):
        raise ValueError(
            "candidate change count mismatch: "
            f"expected {expected_candidate_changes}, got {len(candidate_rows)}"
        )
    if (
        expected_assertion_changes is not None
        and len(assertion_rows) != expected_assertion_changes
    ):
        raise ValueError(
            "assertion change count mismatch: "
            f"expected {expected_assertion_changes}, got {len(assertion_rows)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_review_packet.jsonl"
    assertion_path = output_dir / "assertion_review_packet.jsonl"
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(assertion_path, assertion_rows)
    manifest = {
        "baseline_run": str(baseline_run),
        "experimental_run": str(experimental_run),
        "documents": len(baseline),
        "baseline_entities": len(baseline_index),
        "experimental_entities": len(experimental_index),
        "shared_entities": len(shared),
        "candidate_changes": len(candidate_rows),
        "assertion_changes": len(assertion_rows),
        "context_chars": context_chars,
        "packets": {
            "candidates": str(candidate_path),
            "assertions": str(assertion_path),
        },
    }
    atomic_write_json(output_dir / "packet_manifest.json", manifest)
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        rows.append(row)
    return rows


def load_decisions(path: Path) -> dict[EntityKey, ReviewDecision]:
    decisions: dict[EntityKey, ReviewDecision] = {}
    for raw in _read_jsonl(path):
        decision = ReviewDecision.model_validate(raw)
        key = _decision_key(decision)
        if key in decisions:
            raise ValueError(f"duplicate review decision: {key}")
        decisions[key] = decision
    return decisions


def _packet_index(path: Path) -> dict[EntityKey, dict[str, Any]]:
    indexed: dict[EntityKey, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = (
            str(row["document_id"]),
            int(row["position"][0]),
            int(row["position"][1]),
            str(row["type"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate review packet row: {key}")
        indexed[key] = row
    return indexed


def complete_review_decisions(
    *,
    packet_path: Path,
    overrides_path: Path,
    output_path: Path,
    field: ReviewField,
) -> dict[str, Any]:
    packet = _packet_index(packet_path)
    overrides = load_decisions(overrides_path)
    unknown = overrides.keys() - packet.keys()
    if unknown:
        raise ValueError(f"review overrides target unknown packet rows: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    changed = 0
    for key, packet_row in packet.items():
        decision = overrides.get(key)
        if decision is None:
            decision = ReviewDecision(
                document_id=key[0],
                position=(key[1], key[2]),
                type=EntityType(key[3]),
                selected_value=list(packet_row["baseline_value"]),
                rationale=(
                    "Retain the protected baseline; the experimental change "
                    "was not accepted by the conservative audit."
                ),
            )
        _validate_selected_value(field, decision, packet_row)
        if decision.selected_value != packet_row["baseline_value"]:
            changed += 1
        rows.append(decision.model_dump(mode="json"))
    _write_jsonl(output_path, rows)
    manifest = {
        "field": field,
        "packet": str(packet_path),
        "overrides": str(overrides_path),
        "output": str(output_path),
        "reviewed_rows": len(rows),
        "explicit_overrides": len(overrides),
        "changed_from_baseline": changed,
        "defaulted_to_baseline": len(rows) - len(overrides),
    }
    atomic_write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def _validate_selected_value(
    field: ReviewField,
    decision: ReviewDecision,
    packet_row: dict[str, Any],
) -> None:
    selected = decision.selected_value
    allowed = set(packet_row["allowed_values"])
    if any(value not in allowed for value in selected):
        raise ValueError(
            f"decision for {_decision_key(decision)} selects a value "
            "outside the observed baseline/experimental union"
        )
    if field == "candidates":
        if decision.type not in {EntityType.DIAGNOSIS, EntityType.MEDICATION}:
            raise ValueError("candidate decision targets a non-linkable entity")
        if decision.type == EntityType.MEDICATION and any(
            not value.isascii() or not value.isdigit() for value in selected
        ):
            raise ValueError("medication decision contains a non-numeric RxNorm ID")
    else:
        if decision.type not in {
            EntityType.DIAGNOSIS,
            EntityType.MEDICATION,
            EntityType.SYMPTOM,
        } and selected:
            raise ValueError("assertion decision targets an ineligible entity type")
        try:
            [Assertion(value) for value in selected]
        except ValueError as exc:
            raise ValueError("assertion decision contains an unsupported value") from exc


def _assert_field_isolation(
    baseline: dict[str, list[dict[str, Any]]],
    variant: dict[str, list[dict[str, Any]]],
    field: ReviewField,
) -> None:
    baseline_index = _index_outputs(baseline)
    variant_index = _index_outputs(variant)
    if baseline_index.keys() != variant_index.keys():
        raise ValueError(f"{field}-only variant changed the entity inventory")
    protected = {"text", "type", "position", "assertions", "candidates"} - {field}
    for key in baseline_index:
        for name in protected:
            if baseline_index[key][name] != variant_index[key][name]:
                raise ValueError(
                    f"{field}-only variant changed protected field {name} at {key}"
                )


def build_field_variant(
    *,
    baseline_output: Path,
    packet_path: Path,
    decisions_path: Path,
    field: ReviewField,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output dir {output_dir}")
    baseline = load_output_directory(baseline_output)
    baseline_index = _index_outputs(baseline)
    packet = _packet_index(packet_path)
    decisions = load_decisions(decisions_path)
    if decisions.keys() != packet.keys():
        missing = sorted(packet.keys() - decisions.keys())
        unknown = sorted(decisions.keys() - packet.keys())
        raise ValueError(
            "review decisions must cover the packet exactly: "
            f"missing={missing[:5]}, unknown={unknown[:5]}"
        )

    for key, decision in decisions.items():
        if key not in baseline_index:
            raise ValueError(f"review decision targets an unknown baseline row: {key}")
        _validate_selected_value(field, decision, packet[key])

    variant: dict[str, list[dict[str, Any]]] = {}
    categories = {"baseline": 0, "experimental": 0, "custom": 0}
    changed = 0
    for document_id, entities in baseline.items():
        rows: list[dict[str, Any]] = []
        for entity in entities:
            row = dict(entity)
            key = _entity_key(document_id, entity)
            decision = decisions.get(key)
            if decision is not None:
                selected = list(decision.selected_value)
                packet_row = packet[key]
                if selected == packet_row["baseline_value"]:
                    categories["baseline"] += 1
                elif selected == packet_row["experimental_value"]:
                    categories["experimental"] += 1
                else:
                    categories["custom"] += 1
                if selected != entity[field]:
                    changed += 1
                row[field] = selected
            rows.append(Entity.model_validate(row).output_dict())
        variant[document_id] = rows

    _assert_field_isolation(baseline, variant, field)
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(variant, key=_document_sort_key):
        atomic_write_json(output_dir / f"{document_id}.json", variant[document_id])
    # Isolation experiments preserve the protected baseline inventory exactly,
    # including any legacy redaction-only span. All required structural checks
    # remain enabled.
    validate_output_directory(
        output_dir,
        input_dir,
        expected_stems=set(baseline),
        allow_masked=True,
    )
    manifest = {
        "kind": f"{field}_only",
        "baseline_output": str(baseline_output),
        "packet": str(packet_path),
        "decisions": str(decisions_path),
        "documents": len(variant),
        "entities": len(baseline_index),
        "reviewed_rows": len(decisions),
        "changed_rows": changed,
        "decision_categories": categories,
        "protected_fields_verified": True,
    }
    atomic_write_json(output_dir.parent / f"{output_dir.name}_manifest.json", manifest)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_output(
    *,
    output_dir: Path,
    input_dir: Path,
    zip_path: Path,
    expected_documents: int = 100,
) -> dict[str, Any]:
    validate_output_directory(output_dir, input_dir, allow_masked=True)
    files = sorted(
        output_dir.glob("*.json"),
        key=lambda path: _document_sort_key(path.stem),
    )
    if len(files) != expected_documents:
        raise ValueError(
            f"expected {expected_documents} JSON files, found {len(files)}"
        )
    expected_names = {f"{index}.json" for index in range(1, expected_documents + 1)}
    actual_names = {path.name for path in files}
    if actual_names != expected_names:
        raise ValueError("submission files must be consecutively named 1.json..N.json")
    if zip_path.exists():
        raise ValueError(f"refusing to overwrite existing archive {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{zip_path.name}.",
        suffix=".tmp",
        dir=zip_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in files:
                info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if names != [path.name for path in files]:
                raise ValueError("archive member order or names are invalid")
            if any("/" in name for name in names):
                raise ValueError("archive contains a parent directory")
        os.replace(temporary, zip_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest = {
        "output_dir": str(output_dir),
        "zip_path": str(zip_path),
        "documents": len(files),
        "members": [path.name for path in files],
        "zip_sha256": _sha256(zip_path),
    }
    atomic_write_json(zip_path.with_suffix(".manifest.json"), manifest)
    return manifest


def _copy_metadata(
    *,
    base: dict[str, list[dict[str, Any]]],
    metadata_source: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    source_index = _index_outputs(metadata_source)
    copied: dict[str, list[dict[str, Any]]] = {}
    for document_id, entities in base.items():
        rows: list[dict[str, Any]] = []
        for entity in entities:
            row = dict(entity)
            source = source_index.get(_entity_key(document_id, entity))
            if source is not None:
                row["candidates"] = list(source["candidates"])
                row["assertions"] = list(source["assertions"])
            rows.append(Entity.model_validate(row).output_dict())
        copied[document_id] = rows
    return copied


def _apply_decisions_to_corpus(
    corpus: dict[str, list[dict[str, Any]]],
    decisions: dict[EntityKey, ReviewDecision],
    field: ReviewField,
) -> int:
    changed = 0
    for document_id, entities in corpus.items():
        for index, entity in enumerate(entities):
            decision = decisions.get(_entity_key(document_id, entity))
            if decision is None:
                continue
            if entity[field] != decision.selected_value:
                changed += 1
            row = dict(entity)
            row[field] = list(decision.selected_value)
            entities[index] = Entity.model_validate(row).output_dict()
    return changed


def _same_score(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=SCORE_TOLERANCE)


def _validate_controlled_scores(scores: LeaderboardScores) -> None:
    baseline = scores.baseline
    if not (
        _same_score(baseline.wer, BASELINE_WER)
        and _same_score(baseline.assertions_score, BASELINE_ASSERTIONS)
        and _same_score(baseline.candidates_score, BASELINE_CANDIDATES)
        and _same_score(baseline.computed_final_score, BASELINE_SCORE)
    ):
        raise ValueError("baseline leaderboard metrics do not match full100-v1")
    if not (
        _same_score(scores.candidates.wer, baseline.wer)
        and _same_score(
            scores.candidates.assertions_score,
            baseline.assertions_score,
        )
    ):
        raise ValueError("candidate-only leaderboard result is contaminated")
    if not (
        _same_score(scores.assertions.wer, baseline.wer)
        and _same_score(
            scores.assertions.candidates_score,
            baseline.candidates_score,
        )
    ):
        raise ValueError("assertion-only leaderboard result is contaminated")


def build_adaptive_variant(
    *,
    baseline_output: Path,
    gliner_output: Path,
    candidate_decisions_path: Path,
    assertion_decisions_path: Path,
    scores_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output dir {output_dir}")
    scores = LeaderboardScores.model_validate_json(scores_path.read_text("utf-8"))
    _validate_controlled_scores(scores)
    use_candidates = (
        scores.candidates.candidates_score
        > scores.baseline.candidates_score + SCORE_TOLERANCE
    )
    use_assertions = (
        scores.assertions.assertions_score
        > scores.baseline.assertions_score + SCORE_TOLERANCE
    )
    candidate_gain = (
        scores.candidates.candidates_score - scores.baseline.candidates_score
        if use_candidates
        else 0.0
    )
    assertion_gain = (
        scores.assertions.assertions_score - scores.baseline.assertions_score
        if use_assertions
        else 0.0
    )
    predicted_baseline_composite = (
        scores.baseline.computed_final_score
        + 0.4 * candidate_gain
        + 0.3 * assertion_gain
    )
    use_gliner = (
        scores.gliner.computed_final_score
        > predicted_baseline_composite + SCORE_TOLERANCE
    )

    baseline = load_output_directory(baseline_output)
    if use_gliner:
        gliner = load_output_directory(gliner_output)
        corpus = _copy_metadata(base=gliner, metadata_source=baseline)
        entity_base = "gliner"
    else:
        corpus = {
            document_id: [dict(entity) for entity in entities]
            for document_id, entities in baseline.items()
        }
        entity_base = "baseline"

    candidate_changes = 0
    assertion_changes = 0
    if use_candidates:
        candidate_changes = _apply_decisions_to_corpus(
            corpus,
            load_decisions(candidate_decisions_path),
            "candidates",
        )
    if use_assertions:
        assertion_changes = _apply_decisions_to_corpus(
            corpus,
            load_decisions(assertion_decisions_path),
            "assertions",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(corpus, key=_document_sort_key):
        atomic_write_json(output_dir / f"{document_id}.json", corpus[document_id])
    validate_output_directory(
        output_dir,
        input_dir,
        expected_stems=set(corpus),
        allow_masked=True,
    )
    manifest = {
        "kind": "adaptive_composite",
        "entity_base": entity_base,
        "use_candidate_overlay": use_candidates,
        "use_assertion_overlay": use_assertions,
        "candidate_changes_applied": candidate_changes,
        "assertion_changes_applied": assertion_changes,
        "predicted_baseline_composite": predicted_baseline_composite,
        "gliner_final_score": scores.gliner.computed_final_score,
        "documents": len(corpus),
        "entities": len(_index_outputs(corpus)),
    }
    atomic_write_json(output_dir.parent / f"{output_dir.name}_manifest.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinical-nlp-variants")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packets = subparsers.add_parser("packets")
    packets.add_argument("--baseline-run", type=Path, required=True)
    packets.add_argument("--experimental-run", type=Path, required=True)
    packets.add_argument("--input-dir", type=Path, required=True)
    packets.add_argument("--output-dir", type=Path, required=True)
    packets.add_argument("--context-chars", type=int, default=240)
    packets.add_argument("--expect-candidates", type=int)
    packets.add_argument("--expect-assertions", type=int)

    field = subparsers.add_parser("build-field")
    field.add_argument("--baseline-output", type=Path, required=True)
    field.add_argument("--packet", type=Path, required=True)
    field.add_argument("--decisions", type=Path, required=True)
    field.add_argument(
        "--field",
        choices=("candidates", "assertions"),
        required=True,
    )
    field.add_argument("--input-dir", type=Path, required=True)
    field.add_argument("--output-dir", type=Path, required=True)

    decisions = subparsers.add_parser("complete-decisions")
    decisions.add_argument("--packet", type=Path, required=True)
    decisions.add_argument("--overrides", type=Path, required=True)
    decisions.add_argument("--output", type=Path, required=True)
    decisions.add_argument(
        "--field",
        choices=("candidates", "assertions"),
        required=True,
    )

    package = subparsers.add_parser("package")
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--input-dir", type=Path, required=True)
    package.add_argument("--zip", dest="zip_path", type=Path, required=True)
    package.add_argument("--expected-documents", type=int, default=100)

    adaptive = subparsers.add_parser("adaptive")
    adaptive.add_argument("--baseline-output", type=Path, required=True)
    adaptive.add_argument("--gliner-output", type=Path, required=True)
    adaptive.add_argument("--candidate-decisions", type=Path, required=True)
    adaptive.add_argument("--assertion-decisions", type=Path, required=True)
    adaptive.add_argument("--scores", type=Path, required=True)
    adaptive.add_argument("--input-dir", type=Path, required=True)
    adaptive.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "packets":
        result = generate_review_packets(
            baseline_run=args.baseline_run,
            experimental_run=args.experimental_run,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            context_chars=args.context_chars,
            expected_candidate_changes=args.expect_candidates,
            expected_assertion_changes=args.expect_assertions,
        )
    elif args.command == "build-field":
        result = build_field_variant(
            baseline_output=args.baseline_output,
            packet_path=args.packet,
            decisions_path=args.decisions,
            field=args.field,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "complete-decisions":
        result = complete_review_decisions(
            packet_path=args.packet,
            overrides_path=args.overrides,
            output_path=args.output,
            field=args.field,
        )
    elif args.command == "package":
        result = package_output(
            output_dir=args.output_dir,
            input_dir=args.input_dir,
            zip_path=args.zip_path,
            expected_documents=args.expected_documents,
        )
    else:
        result = build_adaptive_variant(
            baseline_output=args.baseline_output,
            gliner_output=args.gliner_output,
            candidate_decisions_path=args.candidate_decisions,
            assertion_decisions_path=args.assertion_decisions,
            scores_path=args.scores,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
