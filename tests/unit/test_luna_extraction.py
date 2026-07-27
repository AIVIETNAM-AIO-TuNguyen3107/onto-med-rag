from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from clinical_nlp.assertion_detection import AssertionDetector
from clinical_nlp.icd_linking.index import ICDConcept, ICDIndex
from clinical_nlp.luna_extraction import (
    AuditCandidate,
    CodexLunaRunner,
    LunaAuditDecision,
    LunaAuditResponse,
    LunaEntity,
    LunaExtractionResponse,
    LunaLeaderboardMetrics,
    SubscriptionLimitError,
    _candidate_rows,
    _inventory_change_manifest,
    _inventory_diff,
    apply_metadata,
    audit_prompt,
    build_score_adjusted_metadata_variant,
    extraction_prompt,
    merge_luna_inventory,
    reconstruct_luna_entities,
    select_submission4_strategy,
)
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.schemas import Document, EntityType


def _row(
    text: str,
    entity_type: str,
    start: int,
    *,
    candidates: list[str] | None = None,
    assertions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "candidates": candidates or [],
        "assertions": assertions or [],
        "position": [start, start + len(text)],
    }


def _audit(
    document_id: str,
    candidates: list[AuditCandidate],
    keep: dict[tuple[str, str], bool],
) -> LunaAuditResponse:
    return LunaAuditResponse(
        document_id=document_id,
        decisions=[
            LunaAuditDecision(
                id=row.id,
                keep=keep[(row.source, row.text)],
                type=row.type,
            )
            for row in candidates
        ],
    )


def _codex_events(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "test"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": json.dumps(payload, ensure_ascii=False),
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }
            ),
        ]
    )


def test_luna_schema_is_strict_and_has_only_extraction_fields() -> None:
    schema = LunaExtractionResponse.model_json_schema()
    entity_schema = schema["$defs"]["LunaEntity"]
    assert set(entity_schema["properties"]) == {"text", "occurrence", "type"}
    assert entity_schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        LunaEntity.model_validate(
            {
                "text": "ho",
                "occurrence": 1,
                "type": "TRIỆU_CHỨNG",
                "position": [0, 2],
            }
        )


def test_prompts_cover_structured_percentages_and_explicit_clinical_states() -> None:
    document = Document(
        id="2",
        text="Hiệu quả 80%. Tôi mang thai 22 tuần và lo lắng.",
    )
    extract = extraction_prompt(document, [])
    assert "standalone percentage" in extract
    assert "pregnancy with gestational age" in extract
    assert "worry or anxiety" in extract
    audit = audit_prompt(document, [])
    assert "standalone 80%" in audit
    assert "patient-described anxiety" in audit


def test_reconstructs_repeated_mentions_by_document_wide_occurrence() -> None:
    document = Document(id="7", text="đau rồi đau")
    response = LunaExtractionResponse(
        document_id="7",
        entities=[
            LunaEntity(
                text="đau",
                occurrence=2,
                type=EntityType.SYMPTOM,
            ),
            LunaEntity(
                text="đau",
                occurrence=1,
                type=EntityType.SYMPTOM,
            ),
        ],
    )
    assert reconstruct_luna_entities(document, response) == [
        {
            "text": "đau",
            "type": "TRIỆU_CHỨNG",
            "position": [0, 3],
        },
        {
            "text": "đau",
            "type": "TRIỆU_CHỨNG",
            "position": [8, 11],
        },
    ]


def test_reconstruction_collapses_an_exact_duplicate_identity() -> None:
    document = Document(id="7", text="đau")
    duplicate = LunaEntity(
        text="đau",
        occurrence=1,
        type=EntityType.SYMPTOM,
    )
    response = LunaExtractionResponse(
        document_id="7",
        entities=[duplicate, duplicate],
    )
    assert reconstruct_luna_entities(document, response) == [
        {
            "text": "đau",
            "type": "TRIỆU_CHỨNG",
            "position": [0, 3],
        }
    ]


def test_reconstruction_reports_every_nonexact_span() -> None:
    document = Document(id="8", text="mày đay và tế bào")
    response = LunaExtractionResponse(
        document_id="8",
        entities=[
            LunaEntity(
                text="mày đay không có",
                occurrence=1,
                type=EntityType.DIAGNOSIS,
            ),
            LunaEntity(
                text="tế bào vắng",
                occurrence=1,
                type=EntityType.TEST_RESULT,
            ),
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        reconstruct_luna_entities(document, response)
    message = str(exc_info.value)
    assert "mày đay không có" in message
    assert "tế bào vắng" in message
    assert "without normalization" in message


def test_reconstruction_handles_noncanonical_combining_mark_order() -> None:
    document = Document(
        id="9",
        text="te\u0301\u0323 ba\u0300o",
    )
    response = LunaExtractionResponse(
        document_id="9",
        entities=[
            LunaEntity(
                text="t\u1eb9\u0301 b\u00e0o",
                occurrence=1,
                type=EntityType.TEST_RESULT,
            )
        ],
    )
    assert reconstruct_luna_entities(document, response) == [
        {
            "text": "te\u0301\u0323 ba\u0300o",
            "type": "KẾT_QUẢ_XÉT_NGHIỆM",
            "position": [0, 9],
        }
    ]


def test_reconstruction_suggests_exact_escape_for_equal_class_mark_order() -> None:
    document = Document(id="10", text="te\u0301\u0302 bào")
    response = LunaExtractionResponse(
        document_id="10",
        entities=[
            LunaEntity(
                text="te\u0302\u0301 bào",
                occurrence=1,
                type=EntityType.TEST_RESULT,
            )
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        reconstruct_luna_entities(document, response)
    assert r"te\\u0301\\u0302 b" in str(exc_info.value)


def test_delta_merge_requires_audit_for_add_remove_and_retype() -> None:
    document = Document(id="1", text="ho và sốt cao")
    baseline = [_row("ho", "TRIỆU_CHỨNG", 0)]
    luna = [
        _row("sốt cao", "TRIỆU_CHỨNG", 6),
    ]
    candidates = _candidate_rows(document, baseline, luna)
    merged = merge_luna_inventory(
        document,
        baseline,
        luna,
        candidates,
        _audit(
            "1",
            candidates,
            {
                ("baseline", "ho"): True,
                ("luna", "sốt cao"): True,
            },
        ),
    )
    assert [(row["text"], row["type"]) for row in merged] == [
        ("ho", "TRIỆU_CHỨNG"),
        ("sốt cao", "TRIỆU_CHỨNG"),
    ]

    retype_document = Document(id="2", text="ho")
    retype_baseline = [_row("ho", "TRIỆU_CHỨNG", 0)]
    retype_luna = [_row("ho", "CHẨN_ĐOÁN", 0)]
    retype_candidates = _candidate_rows(
        retype_document,
        retype_baseline,
        retype_luna,
    )
    retyped = merge_luna_inventory(
        retype_document,
        retype_baseline,
        retype_luna,
        retype_candidates,
        _audit(
            "2",
            retype_candidates,
            {
                ("baseline", "ho"): False,
                ("luna", "ho"): True,
            },
        ),
    )
    assert retyped[0]["type"] == "CHẨN_ĐOÁN"


def test_boundary_conflict_defaults_to_baseline_and_requires_rejection() -> None:
    document = Document(id="3", text="sốt cao")
    baseline = [_row("sốt", "TRIỆU_CHỨNG", 0)]
    luna = [
        _row("sốt", "TRIỆU_CHỨNG", 0),
        _row("sốt cao", "TRIỆU_CHỨNG", 0),
    ]
    candidates = _candidate_rows(document, baseline, luna)
    assert {(row.source, row.text) for row in candidates} == {
        ("baseline", "sốt"),
        ("luna", "sốt cao"),
    }
    kept_baseline = merge_luna_inventory(
        document,
        baseline,
        luna,
        candidates,
        _audit(
            "3",
            candidates,
            {
                ("baseline", "sốt"): True,
                ("luna", "sốt cao"): False,
            },
        ),
    )
    assert [row["text"] for row in kept_baseline] == ["sốt"]

    kept_new = merge_luna_inventory(
        document,
        baseline,
        luna,
        candidates,
        _audit(
            "3",
            candidates,
            {
                ("baseline", "sốt"): False,
                ("luna", "sốt cao"): True,
            },
        ),
    )
    assert [row["text"] for row in kept_new] == ["sốt cao"]


def test_audit_cannot_invent_or_retype_candidate() -> None:
    document = Document(id="4", text="ho")
    baseline = [_row("ho", "TRIỆU_CHỨNG", 0)]
    candidates = _candidate_rows(document, baseline, [])
    with pytest.raises(ValueError, match="invented candidate"):
        merge_luna_inventory(
            document,
            baseline,
            [],
            candidates,
            LunaAuditResponse(
                document_id="4",
                decisions=[
                    LunaAuditDecision(
                        id="c9999",
                        keep=False,
                        type=EntityType.SYMPTOM,
                    )
                ],
            ),
        )
    with pytest.raises(ValueError, match="retyped candidate"):
        merge_luna_inventory(
            document,
            baseline,
            [],
            candidates,
            LunaAuditResponse(
                document_id="4",
                decisions=[
                    LunaAuditDecision(
                        id=candidates[0].id,
                        keep=False,
                        type=EntityType.DIAGNOSIS,
                    )
                ],
            ),
        )


def test_metadata_is_protected_and_new_links_are_exact_only(
    tmp_path: Path,
) -> None:
    text = "ho nystatin đái tháo đường glucose 80%"
    document = Document(id="5", text=text)
    baseline = [
        _row(
            "ho",
            "TRIỆU_CHỨNG",
            0,
            assertions=["isNegated"],
        )
    ]
    inventory = [
        *baseline,
        _row("nystatin", "THUỐC", text.index("nystatin")),
        _row(
            "đái tháo đường",
            "CHẨN_ĐOÁN",
            text.index("đái tháo đường"),
        ),
        _row(
            "glucose 80%",
            "KẾT_QUẢ_XÉT_NGHIỆM",
            text.index("glucose 80%"),
        ),
    ]
    icd = ICDIndex(
        {
            "E11": ICDConcept(
                code="E11",
                names=("đái tháo đường",),
            )
        }
    )
    rows = apply_metadata(
        document,
        inventory,
        baseline,
        icd_index=icd,
        rxnorm_index=RxNormIndex(tmp_path / "rx.json", use_api=False),
        assertion_detector=AssertionDetector(),
    )
    by_text = {row["text"]: row for row in rows}
    assert by_text["ho"]["assertions"] == ["isNegated"]
    assert by_text["nystatin"]["candidates"] == ["7597"]
    assert by_text["đái tháo đường"]["candidates"] == ["E11"]
    assert by_text["glucose 80%"]["assertions"] == []


def test_runner_retries_malformed_output_and_resumes_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    outputs = [
        _codex_events(
            {
                "document_id": "6",
                "entities": [
                    {
                        "text": "ho",
                        "occurrence": 2,
                        "type": "TRIỆU_CHỨNG",
                    }
                ],
            }
        ),
        _codex_events(
            {
                "document_id": "6",
                "entities": [
                    {
                        "text": "ho",
                        "occurrence": 1,
                        "type": "TRIỆU_CHỨNG",
                    }
                ],
            }
        ),
    ]

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, outputs.pop(0), "")

    monkeypatch.setattr("clinical_nlp.luna_extraction.subprocess.run", fake_run)
    runner = CodexLunaRunner(
        repository=tmp_path,
        run_dir=tmp_path / "run",
        codex_bin="/fake/codex",
    )
    document = Document(id="6", text="ho")

    def validate(response: LunaExtractionResponse) -> None:
        reconstruct_luna_entities(document, response)

    response = runner.run_structured(
        task="extract",
        document_id="6",
        document_text=document.text,
        prompt="extract",
        response_type=LunaExtractionResponse,
        validator=validate,
    )
    assert response.entities[0].occurrence == 1
    assert len(calls) == 2
    command = calls[0]
    assert "--ephemeral" in command
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--search" not in command

    response = runner.run_structured(
        task="extract",
        document_id="6",
        document_text=document.text,
        prompt="extract",
        response_type=LunaExtractionResponse,
        validator=validate,
    )
    assert response.entities[0].occurrence == 1
    assert len(calls) == 2
    checkpoint = json.loads(
        (
            tmp_path
            / "run"
            / "checkpoints"
            / "extract"
            / "6.json"
        ).read_text("utf-8")
    )
    assert checkpoint["model"] == "gpt-5.6-sol"
    assert checkpoint["attempts"][-1]["usage"]["input_tokens"] == 11


def test_subscription_limit_stops_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "usage limit reached")

    monkeypatch.setattr("clinical_nlp.luna_extraction.subprocess.run", fake_run)
    runner = CodexLunaRunner(
        repository=tmp_path,
        run_dir=tmp_path / "run",
        codex_bin="/fake/codex",
    )
    with pytest.raises(SubscriptionLimitError):
        runner.run_structured(
            task="extract",
            document_id="7",
            document_text="ho",
            prompt="extract",
            response_type=LunaExtractionResponse,
            validator=lambda _: None,
        )
    assert calls == 1
    with pytest.raises(ValueError, match="refusing non-Luna model"):
        CodexLunaRunner(
            repository=tmp_path,
            run_dir=tmp_path / "other",
            codex_bin="/fake/codex",
            model="qwen/qwen3.5-9b",
        )


def test_manifest_diff_reports_retype_and_protected_metadata() -> None:
    baseline = {
        "1": [
            _row("ho", "TRIỆU_CHỨNG", 0, assertions=["isHistorical"]),
        ]
    }
    variant = {
        "1": [
            _row("ho", "CHẨN_ĐOÁN", 0),
        ]
    }
    diff = _inventory_diff(baseline, variant)
    assert diff["retyped"] == 1
    assert diff["added"] == 1
    assert diff["removed"] == 1
    assert diff["metadata_changed"] == 0
    changes = _inventory_change_manifest(baseline, variant)
    assert changes["by_document"]["1"]["retyped"] == [
        {
            "text": "ho",
            "position": [0, 2],
            "from_type": "TRIỆU_CHỨNG",
            "to_type": "CHẨN_ĐOÁN",
        }
    ]
    assert changes["by_document"]["1"]["added"] == []
    assert changes["by_document"]["1"]["removed"] == []
    assert changes["by_type"]["retyped_from"]["TRIỆU_CHỨNG"] == 1
    assert changes["by_type"]["retyped_to"]["CHẨN_ĐOÁN"] == 1


def test_submission4_strategy_follows_leaderboard_gates() -> None:
    assert select_submission4_strategy(
        LunaLeaderboardMetrics(
            wer=65,
            assertions_score=38,
            candidates_score=21,
            final_score=30,
        )
    ) == "miss_recovery"
    assert select_submission4_strategy(
        LunaLeaderboardMetrics(
            wer=65,
            assertions_score=35,
            candidates_score=19,
            final_score=29,
        )
    ) == "clear_changed_metadata"
    assert select_submission4_strategy(
        LunaLeaderboardMetrics(
            wer=67,
            assertions_score=40,
            candidates_score=24,
            final_score=31,
        )
    ) == "precision_prune"


def test_score_adjustment_clears_only_changed_luna_rows(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    baseline_dir = tmp_path / "baseline"
    luna_dir = tmp_path / "luna"
    output_dir = tmp_path / "submission4" / "outputs"
    input_dir.mkdir()
    baseline_dir.mkdir()
    luna_dir.mkdir()
    (input_dir / "1.txt").write_text("ho sốt", encoding="utf-8")
    baseline = [
        _row(
            "ho",
            "TRIỆU_CHỨNG",
            0,
            assertions=["isHistorical"],
        )
    ]
    luna = [
        *baseline,
        _row(
            "sốt",
            "CHẨN_ĐOÁN",
            3,
            candidates=["R50.9"],
            assertions=["isNegated"],
        ),
    ]
    (baseline_dir / "1.json").write_text(
        json.dumps(baseline, ensure_ascii=False),
        encoding="utf-8",
    )
    (luna_dir / "1.json").write_text(
        json.dumps(luna, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = build_score_adjusted_metadata_variant(
        baseline_output=baseline_dir,
        luna_output=luna_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        metrics=LunaLeaderboardMetrics(
            wer=65,
            assertions_score=30,
            candidates_score=15,
            final_score=29,
        ),
    )
    rows = json.loads((output_dir / "1.json").read_text("utf-8"))
    assert rows[0]["assertions"] == ["isHistorical"]
    assert rows[1]["candidates"] == []
    assert rows[1]["assertions"] == []
    assert manifest["changed_rows"] == 1
