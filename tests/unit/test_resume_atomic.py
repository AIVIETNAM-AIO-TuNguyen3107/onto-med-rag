from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from clinical_nlp.atomic import atomic_write_text
from clinical_nlp.config import PathsConfig, PipelineConfig
from clinical_nlp.pipeline import DocumentArtifacts
from clinical_nlp.schemas import Document, Entity, EntityType
from clinical_nlp.supervision import RunSupervisor


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    def model_metadata(self) -> dict:
        return {
            "ner": {"backend": "fake", "model_id": "fake-ner"},
            "llm": {"backend": "fake", "model_id": "fake-llm"},
        }

    def process(self, document: Document, *, checkpoint_dir=None):
        self.calls += 1
        entity = Entity(
            text=document.text,
            type=EntityType.SYMPTOM,
            position=(0, len(document.text)),
        )
        artifacts = DocumentArtifacts(
            chunks=[],
            rule_proposals=[],
            ner_proposals=[],
            llm_proposals=[],
            llm_recovery_audit=[],
            merged_entities=[],
            llm_reviews=[],
            assertions=[],
            icd_candidates=[],
            rxnorm_candidates=[],
            reranked_entities=[entity.output_dict()],
            model_metadata=self.model_metadata(),
            warnings=[],
        )
        return [entity], artifacts


class ConcurrentFakePipeline(FakePipeline):
    def __init__(
        self,
        *,
        barrier_ids: set[str] | None = None,
        delays: dict[str, float] | None = None,
        fail_ids: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.barrier_ids = barrier_ids or set()
        self.delays = delays or {}
        self.fail_ids = fail_ids or set()
        self.barrier = (
            threading.Barrier(len(self.barrier_ids))
            if self.barrier_ids
            else None
        )
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.started: list[str] = []

    def process(self, document: Document, *, checkpoint_dir=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started.append(document.id)
        try:
            if document.id in self.barrier_ids:
                assert self.barrier is not None
                self.barrier.wait(timeout=2)
            time.sleep(self.delays.get(document.id, 0))
            if document.id in self.fail_ids:
                raise RuntimeError(f"failed document {document.id}")
            return super().process(document, checkpoint_dir=checkpoint_dir)
        finally:
            with self.lock:
                self.active -= 1


def _config(
    tmp_path: Path,
    documents: dict[str, str] | None = None,
) -> PipelineConfig:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for document_id, text in (documents or {"1": "ho"}).items():
        (input_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
    catalog = tmp_path / "icd.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    index = tmp_path / "icd-index.json"
    index.write_text("{}\n", encoding="utf-8")
    return PipelineConfig(
        paths=PathsConfig(
            input_dir=input_dir,
            runs_dir=tmp_path / "runs",
            artifacts_dir=tmp_path / "artifacts",
            icd_source=tmp_path / "unused.xlsx",
            icd_catalog=catalog,
            icd_index=index,
            rxnorm_cache=tmp_path / "rxnorm.json",
        )
    )


def test_atomic_write_preserves_old_target_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source, destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("clinical_nlp.atomic.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(target, "new")

    assert target.read_text("utf-8") == "old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_resume_skips_independently_validated_completed_document(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    pipeline = FakePipeline()
    supervisor = RunSupervisor(config, pipeline, run_id="resume-test")
    supervisor.preflight(["1"])
    supervisor.run_all(["1"])
    assert pipeline.calls == 1

    supervisor.preflight(["1"], resume=True)
    results = supervisor.run_all(["1"], resume=True)

    assert pipeline.calls == 1
    assert results[0]["resumed"] is True


@pytest.mark.parametrize("value", [0, 9])
def test_document_concurrency_bounds(value: int) -> None:
    with pytest.raises(ValueError):
        PipelineConfig.model_validate({"run": {"document_concurrency": value}})


def test_parallel_run_preserves_selected_order_and_summary_order(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        {"1": "ho", "2": "sốt", "3": "đau"},
    )
    config.run.document_concurrency = 2
    pipeline = ConcurrentFakePipeline(
        barrier_ids={"1", "2"},
        delays={"1": 0.05},
    )
    supervisor = RunSupervisor(config, pipeline, run_id="parallel-test")
    supervisor.preflight(["1", "2", "3"])

    results = supervisor.run_all(["1", "2", "3"])

    assert pipeline.max_active == 2
    assert [row["document_id"] for row in results] == ["1", "2", "3"]
    summary = json.loads(
        (supervisor.run_dir / "quality_summary.json").read_text("utf-8")
    )
    assert [
        row["document_id"] for row in summary["per_document"]
    ] == ["1", "2", "3"]


def test_parallel_resume_skips_valid_output_and_processes_missing_output(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, {"1": "ho", "2": "sốt"})
    config.run.document_concurrency = 2
    pipeline = ConcurrentFakePipeline()
    supervisor = RunSupervisor(config, pipeline, run_id="parallel-resume")
    supervisor.preflight(["1", "2"])
    supervisor.run_document("1")

    supervisor.preflight(["1", "2"], resume=True)
    results = supervisor.run_all(["1", "2"], resume=True)

    assert pipeline.calls == 2
    assert [row["document_id"] for row in results] == ["1", "2"]
    assert [row["resumed"] for row in results] == [True, False]


def test_parallel_failure_cancels_unscheduled_documents(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {"1": "ho", "2": "sốt", "3": "đau"},
    )
    config.run.document_concurrency = 2
    pipeline = ConcurrentFakePipeline(
        barrier_ids={"1", "2"},
        delays={"2": 0.1},
        fail_ids={"1"},
    )
    supervisor = RunSupervisor(config, pipeline, run_id="parallel-failure")
    supervisor.preflight(["1", "2", "3"])

    with pytest.raises(RuntimeError, match="failed document 1"):
        supervisor.run_all(["1", "2", "3"])

    assert set(pipeline.started) == {"1", "2"}
    assert "3" not in pipeline.started


def test_resume_rejects_changed_input_hash(tmp_path: Path) -> None:
    config = _config(tmp_path)
    supervisor = RunSupervisor(config, FakePipeline(), run_id="resume-test")
    supervisor.preflight(["1"])
    (config.paths.input_dir / "1.txt").write_text("sốt", encoding="utf-8")

    with pytest.raises(ValueError, match="hashes do not match"):
        supervisor.preflight(["1"], resume=True)


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    supervisor = RunSupervisor(config, FakePipeline(), run_id="resume-test")
    supervisor.preflight(["1"])
    config.run.pretty_json = False

    with pytest.raises(ValueError, match="configuration"):
        supervisor.preflight(["1"], resume=True)


def test_resume_rejects_corrupt_existing_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    supervisor = RunSupervisor(config, FakePipeline(), run_id="resume-test")
    supervisor.preflight(["1"])
    supervisor.run_all(["1"])
    output = supervisor.run_dir / "outputs" / "1.json"
    output.write_text('[{"text":"sai"}]', encoding="utf-8")

    supervisor.preflight(["1"], resume=True)
    with pytest.raises(Exception):
        supervisor.run_all(["1"], resume=True)


def test_resume_rejects_missing_document_audit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    supervisor = RunSupervisor(config, FakePipeline(), run_id="resume-test")
    supervisor.preflight(["1"])
    supervisor.run_all(["1"])
    (
        supervisor.run_dir / "documents" / "1" / "llm_recovery_audit.json"
    ).unlink()

    supervisor.preflight(["1"], resume=True)
    with pytest.raises(ValueError, match="missing audit artifacts"):
        supervisor.run_all(["1"], resume=True)


def test_resume_rejects_output_outside_selection(tmp_path: Path) -> None:
    config = _config(tmp_path)
    supervisor = RunSupervisor(config, FakePipeline(), run_id="resume-test")
    supervisor.preflight(["1"])
    output_dir = supervisor.run_dir / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "2.json").write_text("[]", encoding="utf-8")

    supervisor.preflight(["1"], resume=True)
    with pytest.raises(ValueError, match="outside the selected"):
        supervisor.run_all(["1"], resume=True)
