from __future__ import annotations

import re

from clinical_nlp.schemas import Assertion, EntityType, SpanProposal


NEGATION_RE = re.compile(
    r"(?i)(?:không\s+ghi\s+nhận|không\s+có|không|phủ\s+nhận|"
    r"chưa\s+thấy|chưa\s+phát\s+hiện|âm\s+tính\s+với|"
    r"without|denies|no\s+evidence\s+of)\s*$"
)
HISTORICAL_RE = re.compile(
    r"(?i)(?:tiền\s+sử|tiền\s+căn|trước\s+đây|đã\s+từng|có\s+lần|"
    r"history\s+of|previously|past\s+medical\s+history)[^.!?;:\n]{0,60}$"
)
FAMILY_SECTION_RE = re.compile(r"(?i)tiền\s+sử\s+gia\s+đình|family\s+history")
FAMILY_SUBJECT_RE = re.compile(
    r"(?i)(?:mẹ|bố|cha|anh\s+trai|chị\s+gái|em\s+trai|em\s+gái|"
    r"ông\s+(?:nội|ngoại)|bà\s+(?:nội|ngoại)|người\s+nhà|họ\s+hàng)"
    r"(?:\s+(?:của\s+)?(?:bệnh\s+nhân|tôi|em|bạn|anh|chị))?"
    r"\s+(?:bị|mắc|có|được\s+chẩn\s+đoán)[^.!?;:\n]{0,100}$"
)
CONTRAST_RE = re.compile(r"(?i)\b(?:nhưng|tuy\s+nhiên|however|but)\b")
CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;\n]")


def _section_at(text: str, position: int) -> str | None:
    prefix = text[:position]
    lines = prefix.splitlines()
    for line in reversed(lines[-12:]):
        normalized = line.strip().casefold()
        if not normalized:
            continue
        if "tiền sử bệnh hiện tại" in normalized or "bệnh sử hiện tại" in normalized:
            return "current_history"
        if "thuốc trước" in normalized or "danh sách thuốc trước" in normalized:
            return "medication_history"
        if "tiền sử gia đình" in normalized:
            return "family_history"
        if normalized.startswith("tiền sử") or "past medical history" in normalized:
            return "past_history"
        if "thuốc đang dùng" in normalized:
            return "current_medication"
        if "chẩn đoán" in normalized:
            return "diagnosis"
    return None


class AssertionDetector:
    def detect(self, text: str, proposal: SpanProposal) -> list[Assertion]:
        if proposal.type not in {
            EntityType.SYMPTOM,
            EntityType.DIAGNOSIS,
            EntityType.MEDICATION,
        }:
            return []
        assertions: set[Assertion] = set()
        section = _section_at(text, proposal.start)
        if section in {"medication_history", "past_history"}:
            assertions.add(Assertion.HISTORICAL)
        if section == "family_history":
            assertions.add(Assertion.FAMILY)

        window_start = max(0, proposal.start - 180)
        prefix = text[window_start : proposal.start]
        boundary = max(
            [match.end() for match in CLAUSE_BOUNDARY_RE.finditer(prefix)] or [0]
        )
        clause_prefix = prefix[boundary:]
        contrasts = list(CONTRAST_RE.finditer(clause_prefix))
        if contrasts:
            clause_prefix = clause_prefix[contrasts[-1].end() :]

        if NEGATION_RE.search(clause_prefix):
            assertions.add(Assertion.NEGATED)
        if HISTORICAL_RE.search(clause_prefix):
            assertions.add(Assertion.HISTORICAL)
        if FAMILY_SUBJECT_RE.search(clause_prefix):
            assertions.add(Assertion.FAMILY)

        section_prefix = text[max(0, proposal.start - 500) : proposal.start]
        if FAMILY_SECTION_RE.search(section_prefix) and section == "family_history":
            assertions.add(Assertion.FAMILY)

        order = [Assertion.NEGATED, Assertion.FAMILY, Assertion.HISTORICAL]
        return [item for item in order if item in assertions]

