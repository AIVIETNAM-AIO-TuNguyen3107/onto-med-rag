from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from clinical_nlp.atomic import atomic_write_json, atomic_write_text
from clinical_nlp.submission_variants import (
    ReviewDecision,
    ScoreBreakdown,
    build_adaptive_variant,
    build_field_variant,
    complete_review_decisions,
    generate_review_packets,
    package_output,
)


def _entity(
    text: str,
    entity_type: str,
    position: list[int],
    *,
    candidates: list[str] | None = None,
    assertions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "candidates": candidates or [],
        "assertions": assertions or [],
        "position": position,
    }


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    input_dir = root / "input"
    baseline_run = root / "baseline"
    experimental_run = root / "experimental"
    input_dir.mkdir(parents=True)
    (baseline_run / "outputs").mkdir(parents=True)
    (experimental_run / "outputs").mkdir(parents=True)
    (input_dir / "1.txt").write_text("ho và aspirin", encoding="utf-8")
    baseline = [
        _entity("ho", "TRIỆU_CHỨNG", [0, 2]),
        _entity("aspirin", "THUỐC", [6, 13], candidates=[]),
    ]
    experimental = [
        _entity(
            "ho",
            "TRIỆU_CHỨNG",
            [0, 2],
            assertions=["isHistorical"],
        ),
        _entity("aspirin", "THUỐC", [6, 13], candidates=["1191"]),
    ]
    atomic_write_json(baseline_run / "outputs" / "1.json", baseline)
    atomic_write_json(experimental_run / "outputs" / "1.json", experimental)
    return input_dir, baseline_run, experimental_run


def _write_decisions(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def test_generate_packets_and_build_isolated_variants(tmp_path: Path) -> None:
    input_dir, baseline_run, experimental_run = _write_fixture(tmp_path)
    packet_dir = tmp_path / "packets"
    manifest = generate_review_packets(
        baseline_run=baseline_run,
        experimental_run=experimental_run,
        input_dir=input_dir,
        output_dir=packet_dir,
        expected_candidate_changes=1,
        expected_assertion_changes=1,
    )
    assert manifest["candidate_changes"] == 1
    assert manifest["assertion_changes"] == 1

    candidate_decisions = tmp_path / "candidate_decisions.jsonl"
    _write_decisions(
        candidate_decisions,
        [
            {
                "document_id": "1",
                "position": [6, 13],
                "type": "THUỐC",
                "selected_value": ["1191"],
                "rationale": "Exact ingredient match.",
            }
        ],
    )
    candidate_output = tmp_path / "candidate_output"
    result = build_field_variant(
        baseline_output=baseline_run / "outputs",
        packet_path=packet_dir / "candidate_review_packet.jsonl",
        decisions_path=candidate_decisions,
        field="candidates",
        input_dir=input_dir,
        output_dir=candidate_output,
    )
    payload = json.loads((candidate_output / "1.json").read_text("utf-8"))
    assert result["changed_rows"] == 1
    assert payload[0]["assertions"] == []
    assert payload[1]["candidates"] == ["1191"]

    assertion_decisions = tmp_path / "assertion_decisions.jsonl"
    _write_decisions(
        assertion_decisions,
        [
            {
                "document_id": "1",
                "position": [0, 2],
                "type": "TRIỆU_CHỨNG",
                "selected_value": ["isHistorical"],
                "rationale": "Explicit history context.",
            }
        ],
    )
    assertion_output = tmp_path / "assertion_output"
    build_field_variant(
        baseline_output=baseline_run / "outputs",
        packet_path=packet_dir / "assertion_review_packet.jsonl",
        decisions_path=assertion_decisions,
        field="assertions",
        input_dir=input_dir,
        output_dir=assertion_output,
    )
    payload = json.loads((assertion_output / "1.json").read_text("utf-8"))
    assert payload[0]["assertions"] == ["isHistorical"]
    assert payload[1]["candidates"] == []


def test_unknown_or_incomplete_decisions_are_rejected(tmp_path: Path) -> None:
    input_dir, baseline_run, experimental_run = _write_fixture(tmp_path)
    packet_dir = tmp_path / "packets"
    generate_review_packets(
        baseline_run=baseline_run,
        experimental_run=experimental_run,
        input_dir=input_dir,
        output_dir=packet_dir,
    )
    decisions = tmp_path / "decisions.jsonl"
    _write_decisions(
        decisions,
        [
            {
                "document_id": "1",
                "position": [0, 2],
                "type": "TRIỆU_CHỨNG",
                "selected_value": [],
                "rationale": "Wrong packet row.",
            }
        ],
    )
    with pytest.raises(ValueError, match="cover the packet exactly"):
        build_field_variant(
            baseline_output=baseline_run / "outputs",
            packet_path=packet_dir / "candidate_review_packet.jsonl",
            decisions_path=decisions,
            field="candidates",
            input_dir=input_dir,
            output_dir=tmp_path / "out",
        )


def test_complete_decisions_defaults_unreviewed_rows_to_baseline(
    tmp_path: Path,
) -> None:
    input_dir, baseline_run, experimental_run = _write_fixture(tmp_path)
    packet_dir = tmp_path / "packets"
    generate_review_packets(
        baseline_run=baseline_run,
        experimental_run=experimental_run,
        input_dir=input_dir,
        output_dir=packet_dir,
    )
    overrides = tmp_path / "overrides.jsonl"
    atomic_write_text(overrides, "")
    decisions = tmp_path / "decisions.jsonl"
    manifest = complete_review_decisions(
        packet_path=packet_dir / "candidate_review_packet.jsonl",
        overrides_path=overrides,
        output_path=decisions,
        field="candidates",
    )
    rows = [
        json.loads(line)
        for line in decisions.read_text("utf-8").splitlines()
        if line
    ]
    assert manifest["reviewed_rows"] == 1
    assert manifest["changed_from_baseline"] == 0
    assert rows[0]["selected_value"] == []


def test_review_decision_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "document_id": "1",
                "position": [0, 2],
                "type": "TRIỆU_CHỨNG",
                "selected_value": [],
                "rationale": "No assertion.",
                "unexpected": True,
            }
        )


def test_variant_preserves_legacy_masked_baseline_span(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    baseline_run = tmp_path / "baseline"
    experimental_run = tmp_path / "experimental"
    input_dir.mkdir()
    (baseline_run / "outputs").mkdir(parents=True)
    (experimental_run / "outputs").mkdir(parents=True)
    (input_dir / "1.txt").write_text("*****", encoding="utf-8")
    atomic_write_json(
        baseline_run / "outputs" / "1.json",
        [_entity("*****", "THUỐC", [0, 5])],
    )
    atomic_write_json(
        experimental_run / "outputs" / "1.json",
        [_entity("*****", "THUỐC", [0, 5], candidates=["123"])],
    )
    packet_dir = tmp_path / "packets"
    generate_review_packets(
        baseline_run=baseline_run,
        experimental_run=experimental_run,
        input_dir=input_dir,
        output_dir=packet_dir,
    )
    packet = json.loads(
        (packet_dir / "candidate_review_packet.jsonl").read_text("utf-8")
    )
    decisions = tmp_path / "decisions.jsonl"
    _write_decisions(
        decisions,
        [
            {
                "document_id": packet["document_id"],
                "position": packet["position"],
                "type": packet["type"],
                "selected_value": ["123"],
                "rationale": "Exact numeric medication identifier.",
            }
        ],
    )

    output_dir = tmp_path / "variant"
    manifest = build_field_variant(
        baseline_output=baseline_run / "outputs",
        packet_path=packet_dir / "candidate_review_packet.jsonl",
        decisions_path=decisions,
        field="candidates",
        input_dir=input_dir,
        output_dir=output_dir,
    )

    assert manifest["changed_rows"] == 1
    assert json.loads((output_dir / "1.json").read_text("utf-8"))[0]["text"] == "*****"


def test_deterministic_zip_has_only_root_json_members(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()
    for document_id, text in (("1", "ho"), ("2", "sốt")):
        (input_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
        atomic_write_json(
            output_dir / f"{document_id}.json",
            [_entity(text, "TRIỆU_CHỨNG", [0, len(text)])],
        )
    archive = tmp_path / "submission.zip"
    manifest = package_output(
        output_dir=output_dir,
        input_dir=input_dir,
        zip_path=archive,
        expected_documents=2,
    )
    assert len(manifest["zip_sha256"]) == 64
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["1.json", "2.json"]


def test_score_formula_and_adaptive_baseline_composite(tmp_path: Path) -> None:
    baseline_score = ScoreBreakdown(
        wer=66.9345,
        assertions_score=36.2262,
        candidates_score=20.9721,
        final_score=29.1764,
    )
    assert baseline_score.computed_final_score == pytest.approx(29.17635)

    input_dir, baseline_run, experimental_run = _write_fixture(tmp_path)
    gliner_output = tmp_path / "gliner"
    gliner_output.mkdir()
    atomic_write_json(
        gliner_output / "1.json",
        [_entity("ho", "TRIỆU_CHỨNG", [0, 2])],
    )
    candidate_decisions = tmp_path / "candidate_decisions.jsonl"
    assertion_decisions = tmp_path / "assertion_decisions.jsonl"
    _write_decisions(
        candidate_decisions,
        [
            {
                "document_id": "1",
                "position": [6, 13],
                "type": "THUỐC",
                "selected_value": ["1191"],
                "rationale": "Exact ingredient.",
            }
        ],
    )
    _write_decisions(
        assertion_decisions,
        [
            {
                "document_id": "1",
                "position": [0, 2],
                "type": "TRIỆU_CHỨNG",
                "selected_value": ["isHistorical"],
                "rationale": "Explicit history.",
            }
        ],
    )
    scores = {
        "baseline": {
            "wer": 66.9345,
            "assertions_score": 36.2262,
            "candidates_score": 20.9721,
            "final_score": 29.1764,
        },
        "candidates": {
            "wer": 66.9345,
            "assertions_score": 36.2262,
            "candidates_score": 22.0,
        },
        "assertions": {
            "wer": 66.9345,
            "assertions_score": 37.0,
            "candidates_score": 20.9721,
        },
        "gliner": {
            "wer": 75.0,
            "assertions_score": 30.0,
            "candidates_score": 20.0,
        },
    }
    scores_path = tmp_path / "scores.json"
    atomic_write_json(scores_path, scores)
    output_dir = tmp_path / "adaptive"
    manifest = build_adaptive_variant(
        baseline_output=baseline_run / "outputs",
        gliner_output=gliner_output,
        candidate_decisions_path=candidate_decisions,
        assertion_decisions_path=assertion_decisions,
        scores_path=scores_path,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    payload = json.loads((output_dir / "1.json").read_text("utf-8"))
    assert manifest["entity_base"] == "baseline"
    assert manifest["use_candidate_overlay"] is True
    assert manifest["use_assertion_overlay"] is True
    assert payload[0]["assertions"] == ["isHistorical"]
    assert payload[1]["candidates"] == ["1191"]
