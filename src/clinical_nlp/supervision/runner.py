from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinical_nlp.config import PipelineConfig
from clinical_nlp.pipeline import ClinicalPipeline, DocumentArtifacts
from clinical_nlp.schemas import Document
from clinical_nlp.validation.output import write_entities


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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

    def preflight(self, document_ids: list[str] | None = None) -> dict[str, Any]:
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
        _write_json(self.run_dir / "source_manifest.json", manifest)
        _write_json(
            self.run_dir / "config.json",
            self.config.model_dump(mode="json"),
        )
        return manifest

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
    ) -> list[dict[str, Any]]:
        files = self._selected_files(document_ids)
        output_dir = self.run_dir / "outputs"
        existing_outputs = sorted(path.name for path in output_dir.glob("*.json"))
        if existing_outputs:
            raise FileExistsError(
                "run output directory is not empty; use a new run ID: "
                f"{existing_outputs}"
            )
        results: list[dict[str, Any]] = []
        for index, path in enumerate(files, start=1):
            result = self.run_document(path.stem)
            results.append(result)
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(files)}",
                        "document_id": path.stem,
                        "status": result["status"],
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

    @staticmethod
    def _write_artifacts(doc_dir: Path, artifacts: DocumentArtifacts) -> None:
        mapping = {
            "chunks.json": artifacts.chunks,
            "rule_proposals.json": artifacts.rule_proposals,
            "ner_proposals.json": artifacts.ner_proposals,
            "llm_proposals.json": artifacts.llm_proposals,
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
