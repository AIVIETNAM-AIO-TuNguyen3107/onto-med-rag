from __future__ import annotations

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


def _config(tmp_path: Path) -> PipelineConfig:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("ho", encoding="utf-8")
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
