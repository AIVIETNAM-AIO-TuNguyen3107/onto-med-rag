from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_nlp.atomic import atomic_write_json
from clinical_nlp.config import PipelineConfig
from clinical_nlp.pipeline import ClinicalPipeline, DocumentArtifacts
from clinical_nlp.schemas import Document, Entity
from clinical_nlp.validation.output import validate_entities, write_entities


DOCUMENT_ARTIFACT_NAMES = (
    "chunks.json",
    "rule_proposals.json",
    "ner_proposals.json",
    "llm_proposals.json",
    "llm_recovery_audit.json",
    "merged_entities.json",
    "llm_reviews.json",
    "assertions.json",
    "icd_candidates.json",
    "rxnorm_candidates.json",
    "reranked_entities.json",
    "model_metadata.json",
    "warnings.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


class RunSupervisor:
    def __init__(
        self,
        config: PipelineConfig,
        pipeline: ClinicalPipeline,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = config.paths.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _selected_files(self, document_ids: list[str] | None = None) -> list[Path]:
        if document_ids is None:
            files = list(self.config.paths.input_dir.glob("*.txt"))
        else:
            if not document_ids:
                raise ValueError("at least one document ID is required")
            if len(document_ids) != len(set(document_ids)):
                raise ValueError("document IDs must be unique")
            if any(
                not value or Path(value).name != value or Path(value).suffix
                for value in document_ids
            ):
                raise ValueError("document IDs must be plain filename stems")
            files = [
                self.config.paths.input_dir / f"{value}.txt"
                for value in document_ids
            ]
            missing = [str(path) for path in files if not path.exists()]
            if missing:
                raise FileNotFoundError(f"selected input files do not exist: {missing}")
        return sorted(
            files,
            key=lambda path: (
                (0, int(path.stem))
                if path.stem.isdigit()
                else (1, path.stem)
            ),
        )

    def preflight(
        self,
        document_ids: list[str] | None = None,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        icd_source = self.config.paths.preferred_icd_source()
        files = self._selected_files(document_ids)
        if not files:
            raise ValueError("no input text files found")
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "inputs": [
                {
                    "id": path.stem,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            ],
            "selection": {
                "document_ids": [path.stem for path in files],
                "document_count": len(files),
            },
            "models": {
                "ner": self.config.ner.model_dump(mode="json"),
                "llm": self.config.llm.model_dump(mode="json"),
                "runtime": self.pipeline.model_metadata(),
            },
            "terminology": {
                "icd_source": str(icd_source),
                "icd_source_sha256": (
                    _sha256(icd_source) if icd_source.exists() else None
                ),
                "icd_index": str(self.config.paths.icd_index),
                "icd_index_sha256": (
                    _sha256(self.config.paths.icd_index)
                    if self.config.paths.icd_index.exists()
                    else None
                ),
                "rxnorm_cache": str(self.config.paths.rxnorm_cache),
                "rxnorm_cache_sha256_before": (
                    _sha256(self.config.paths.rxnorm_cache)
                    if self.config.paths.rxnorm_cache.exists()
                    else None
                ),
            },
        }
        config_payload = self.config.model_dump(mode="json")
        source_manifest_path = self.run_dir / "source_manifest.json"
        config_path = self.run_dir / "config.json"
        if resume:
            self._validate_resume_compatibility(manifest, config_payload)
            events_path = self.run_dir / "resume_events.json"
            events = (
                json.loads(events_path.read_text("utf-8"))
                if events_path.exists()
                else []
            )
            if not isinstance(events, list):
                raise ValueError("resume_events.json must contain a JSON array")
            events.append(
                {
                    "resumed_at": datetime.now(UTC).isoformat(),
                    "document_ids": [path.stem for path in files],
                }
            )
            _write_json(events_path, events)
        else:
            if source_manifest_path.exists() or config_path.exists():
                raise FileExistsError(
                    f"run ID {self.run_id!r} is already initialized; "
                    "use --resume or a new run ID"
                )
            _write_json(source_manifest_path, manifest)
            _write_json(config_path, config_payload)
        return manifest

    def _validate_resume_compatibility(
        self,
        current_manifest: dict[str, Any],
        current_config: dict[str, Any],
    ) -> None:
        source_manifest_path = self.run_dir / "source_manifest.json"
        config_path = self.run_dir / "config.json"
        if not source_manifest_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                "resume requires existing source_manifest.json and config.json"
            )
        stored_manifest = json.loads(source_manifest_path.read_text("utf-8"))
        stored_config = json.loads(config_path.read_text("utf-8"))
        if stored_config != current_config:
            raise ValueError("resume configuration does not match the original run")
        if _resume_manifest_signature(stored_manifest) != _resume_manifest_signature(
            current_manifest
        ):
            raise ValueError(
                "resume input, model, selection, or ICD hashes do not match "
                "the original run"
            )

    def record_online_preflight(self, payload: dict[str, Any]) -> None:
        _write_json(self.run_dir / "online_preflight.json", payload)

    def run_document(self, document_id: str) -> dict[str, Any]:
        source = self.config.paths.input_dir / f"{document_id}.txt"
        if not source.exists():
            raise FileNotFoundError(source)
        started = time.monotonic()
        document = Document(
            id=document_id,
            text=source.read_text("utf-8"),
            source_path=str(source),
        )
        entities, artifacts = self.pipeline.process(document)
        doc_dir = self.run_dir / "documents" / document_id
        self._write_artifacts(doc_dir, artifacts)
        write_entities(
            self.run_dir / "outputs" / f"{document_id}.json",
            document,
            entities,
            pretty=self.config.run.pretty_json,
        )
        validation = {
            "document_id": document_id,
            "status": "ok",
            "entity_count": len(entities),
            "warning_count": len(artifacts.warnings),
            "warnings": artifacts.warnings,
            "elapsed_seconds": time.monotonic() - started,
            "stage_counts": {
                "rule_proposals": len(artifacts.rule_proposals),
                "ner_proposals": len(artifacts.ner_proposals),
                "llm_proposals": len(artifacts.llm_proposals),
                "merged_entities": len(artifacts.merged_entities),
                "reranked_entities": len(artifacts.reranked_entities),
            },
            "resumed": False,
            "type_counts": dict(Counter(entity.type.value for entity in entities)),
            "assertion_counts": dict(
                Counter(
                    assertion.value
                    for entity in entities
                    for assertion in entity.assertions
                )
            ),
            "linked_entities": sum(bool(entity.candidates) for entity in entities),
            "empty_candidate_entities": sum(
                entity.type.value in {"BỆNH_LÝ", "THUỐC"} and not entity.candidates
                for entity in entities
            ),
        }
        _write_json(doc_dir / "validation.json", validation)
        return validation

    def run_all(
        self,
        document_ids: list[str] | None = None,
        *,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        files = self._selected_files(document_ids)
        output_dir = self.run_dir / "outputs"
        existing_outputs = sorted(path.name for path in output_dir.glob("*.json"))
        expected_names = {f"{path.stem}.json" for path in files}
        unexpected_outputs = set(existing_outputs) - expected_names
        if unexpected_outputs:
            raise ValueError(
                "run contains outputs outside the selected document set: "
                f"{sorted(unexpected_outputs)}"
            )
        if existing_outputs and not resume:
            raise FileExistsError(
                "run output directory is not empty; use --resume or a new run ID: "
                f"{existing_outputs}"
            )
        results: list[dict[str, Any]] = []
        for index, path in enumerate(files, start=1):
            output_path = output_dir / f"{path.stem}.json"
            if resume and output_path.exists():
                result = self._load_completed_document(path.stem)
                result = {**result, "resumed": True}
                progress_status = "skipped_valid"
            else:
                result = self.run_document(path.stem)
                progress_status = result["status"]
            results.append(result)
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(files)}",
                        "document_id": path.stem,
                        "status": progress_status,
                        "entities": result["entity_count"],
                        "warnings": result["warning_count"],
                        "elapsed_seconds": round(result["elapsed_seconds"], 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        _write_json(
            self.run_dir / "stages" / "09_validation_summary.json",
            {
                "status": "ok",
                "documents": len(results),
                "entities": sum(row["entity_count"] for row in results),
                "warnings": sum(row["warning_count"] for row in results),
            },
        )
        type_counts: Counter[str] = Counter()
        assertion_counts: Counter[str] = Counter()
        for row in results:
            type_counts.update(row["type_counts"])
            assertion_counts.update(row["assertion_counts"])
        _write_json(
            self.run_dir / "quality_summary.json",
            {
                "documents": len(results),
                "entities": sum(row["entity_count"] for row in results),
                "types": dict(type_counts),
                "assertions": dict(assertion_counts),
                "linked_entities": sum(row["linked_entities"] for row in results),
                "empty_candidate_entities": sum(
                    row["empty_candidate_entities"] for row in results
                ),
                "per_document": [
                    {
                        "document_id": row["document_id"],
                        "entities": row["entity_count"],
                        "types": row["type_counts"],
                        "assertions": row["assertion_counts"],
                        "linked_entities": row["linked_entities"],
                        "empty_candidate_entities": row[
                            "empty_candidate_entities"
                        ],
                        "warnings": row["warning_count"],
                    }
                    for row in results
                ],
            },
        )
        for index, stage in enumerate(
            (
                "rule_proposals",
                "ner_proposals",
                "llm_proposals",
                "merged_entities",
                "reranked_entities",
            ),
            start=1,
        ):
            _write_json(
                self.run_dir / "stages" / f"{index:02d}_{stage}_summary.json",
                {
                    "stage": stage,
                    "documents": len(results),
                    "total": sum(row["stage_counts"][stage] for row in results),
                    "per_document": [
                        {
                            "document_id": row["document_id"],
                            "count": row["stage_counts"][stage],
                        }
                        for row in results
                    ],
                },
            )
        _write_json(
            self.run_dir / "run_manifest_final.json",
            {
                "run_id": self.run_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "document_ids": [path.stem for path in files],
                "models": self.pipeline.model_metadata(),
                "terminology": {
                    "icd_index_sha256": (
                        _sha256(self.config.paths.icd_index)
                        if self.config.paths.icd_index.exists()
                        else None
                    ),
                    "rxnorm_cache_sha256_after": (
                        _sha256(self.config.paths.rxnorm_cache)
                        if self.config.paths.rxnorm_cache.exists()
                        else None
                    ),
                },
                "validation": {
                    "documents": len(results),
                    "entities": sum(row["entity_count"] for row in results),
                    "warnings": sum(row["warning_count"] for row in results),
                },
            },
        )
        return results

    def _load_completed_document(self, document_id: str) -> dict[str, Any]:
        source = self.config.paths.input_dir / f"{document_id}.txt"
        output = self.run_dir / "outputs" / f"{document_id}.json"
        payload = json.loads(output.read_text("utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{output} must contain a JSON array")
        document = Document(
            id=document_id,
            text=source.read_text("utf-8"),
            source_path=str(source),
        )
        entities = [Entity.model_validate(row) for row in payload]
        validate_entities(document, entities)

        doc_dir = self.run_dir / "documents" / document_id
        missing_artifacts = [
            name
            for name in (*DOCUMENT_ARTIFACT_NAMES, "validation.json")
            if not (doc_dir / name).exists()
        ]
        if missing_artifacts:
            raise ValueError(
                f"cannot resume document {document_id}; missing audit artifacts: "
                f"{missing_artifacts}"
            )
        validation = json.loads((doc_dir / "validation.json").read_text("utf-8"))
        required_keys = {
            "document_id",
            "status",
            "entity_count",
            "warning_count",
            "warnings",
            "elapsed_seconds",
            "stage_counts",
            "type_counts",
            "assertion_counts",
            "linked_entities",
            "empty_candidate_entities",
        }
        if not isinstance(validation, dict) or not required_keys <= validation.keys():
            raise ValueError(
                f"cannot resume document {document_id}; invalid validation artifact"
            )
        if (
            validation["status"] != "ok"
            or validation["document_id"] != document_id
            or validation["entity_count"] != len(entities)
        ):
            raise ValueError(
                f"cannot resume document {document_id}; validation mismatch"
            )
        return validation

    @staticmethod
    def _write_artifacts(doc_dir: Path, artifacts: DocumentArtifacts) -> None:
        mapping = {
            "chunks.json": artifacts.chunks,
            "rule_proposals.json": artifacts.rule_proposals,
            "ner_proposals.json": artifacts.ner_proposals,
            "llm_proposals.json": artifacts.llm_proposals,
            "llm_recovery_audit.json": artifacts.llm_recovery_audit,
            "merged_entities.json": artifacts.merged_entities,
            "llm_reviews.json": artifacts.llm_reviews,
            "assertions.json": artifacts.assertions,
            "icd_candidates.json": artifacts.icd_candidates,
            "rxnorm_candidates.json": artifacts.rxnorm_candidates,
            "reranked_entities.json": artifacts.reranked_entities,
            "model_metadata.json": artifacts.model_metadata,
            "warnings.json": artifacts.warnings,
        }
        for name, payload in mapping.items():
            _write_json(doc_dir / name, payload)


def _resume_manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    models = manifest.get("models", {})
    terminology = manifest.get("terminology", {})
    return {
        "inputs": manifest.get("inputs"),
        "selection": manifest.get("selection"),
        "models": {
            "ner": models.get("ner"),
            "llm": models.get("llm"),
        },
        "terminology": {
            "icd_source": terminology.get("icd_source"),
            "icd_source_sha256": terminology.get("icd_source_sha256"),
            "icd_index": terminology.get("icd_index"),
            "icd_index_sha256": terminology.get("icd_index_sha256"),
            "rxnorm_cache": terminology.get("rxnorm_cache"),
        },
    }
