"""Rule-based entity extractor — Week 2 PoC baseline.

ponytail: regex/heuristic only; swap for ViHealthBERT or ≤9B LLM in Week 3.
"""

from __future__ import annotations

import re

from src.schemas.entity import MedicalEntity

_NUMBERED_DRUG = re.compile(
    r"\d+\.\s*(.+?)(?=\s+\d+\.\s|\s*$)",
)
_DIEU_TRI_SYMPTOM = re.compile(
    r"điều trị\s+(.+?)(?=[\.\n,;]|$)",
    re.IGNORECASE,
)
_DIAGNOSIS = re.compile(
    r"(?:chẩn đoán|mắc bệnh)\s+(?:mắc\s+bệnh\s+)?(.+?)(?=[\.\n]|$)",
    re.IGNORECASE,
)
_TIEN_SU_DRUG = re.compile(
    r"tiền sử[^,\n]*?(?:sử dụng|dùng|uống)\s+(.+?)(?=[\.\n]|$)",
    re.IGNORECASE,
)
_LAB_PAIR = re.compile(
    r"([A-Za-z][A-Za-z0-9%]*(?:\s*\([^)]+\))?)\s*:\s*([\d,\.]+)",
)
_COMMA_SYMPTOM = re.compile(
    r"(?:ho[^,]*|tức ngực|đau thượng vị|ợ hơi|sốt[^,]*|đau nhức|táo bón|lo âu|mất ngủ)",
    re.IGNORECASE,
)

_HISTORICAL_MARKERS = (
    "trước nhập viện",
    "tiền sử",
    "đã dùng",
    "sử dụng",
    "home medication",
)
_NEGATION_MARKERS = ("không ", "không có ", "denies ", "negative for ")
_FAMILY_MARKERS = ("bố ", "mẹ ", "anh ", "em ", "con ", "vợ ", "chồng ", "gia đình")


def _span(text: str, start: int, end: int, **fields) -> MedicalEntity:
    return MedicalEntity(
        text=text[start:end],
        position=[start, end],
        **fields,
    )


def _window_assertions(text: str, start: int, end: int) -> list[str]:
    """Keyword assertions from a local context window around the span."""
    left = max(0, start - 80)
    right = min(len(text), end + 80)
    ctx = text[left:right].lower()
    out: list[str] = []

    if any(m in ctx for m in _HISTORICAL_MARKERS):
        out.append("isHistorical")
    if any(m in ctx for m in _NEGATION_MARKERS):
        out.append("isNegated")
    if any(m in ctx for m in _FAMILY_MARKERS):
        out.append("isFamily")
    return out


def _add_unique(entities: list[MedicalEntity], ent: MedicalEntity) -> None:
    key = (ent["position"][0], ent["position"][1], ent["type"])
    if any((e["position"][0], e["position"][1], e["type"]) == key for e in entities):
        return
    entities.append(ent)


def extract_entities(text: str) -> list[MedicalEntity]:
    entities: list[MedicalEntity] = []

    for m in _NUMBERED_DRUG.finditer(text):
        span_text = m.group(1).strip()
        start = m.start(1)
        end = start + len(span_text)
        _add_unique(
            entities,
            _span(
                text,
                start,
                end,
                type="THUỐC",
                assertions=_window_assertions(text, start, end) or ["isHistorical"],
                candidates=[],
            ),
        )

    for m in _TIEN_SU_DRUG.finditer(text):
        chunk = m.group(1)
        for part in re.split(r",\s*", chunk):
            part = part.strip()
            if not part:
                continue
            idx = text.find(part, m.start(1))
            if idx < 0:
                continue
            _add_unique(
                entities,
                _span(
                    text,
                    idx,
                    idx + len(part),
                    type="THUỐC",
                    assertions=["isHistorical"],
                    candidates=[],
                ),
            )

    for m in _DIAGNOSIS.finditer(text):
        span_text = m.group(1).strip().rstrip(".")
        start = m.start(1)
        end = start + len(span_text)
        _add_unique(
            entities,
            _span(
                text,
                start,
                end,
                type="CHẨN_ĐOÁN",
                assertions=_window_assertions(text, start, end),
                candidates=[],
            ),
        )

    for m in _DIEU_TRI_SYMPTOM.finditer(text):
        span_text = m.group(1).strip().rstrip(".")
        start = m.start(1)
        end = start + len(span_text)
        _add_unique(
            entities,
            _span(
                text,
                start,
                end,
                type="TRIỆU_CHỨNG",
                assertions=[],
                candidates=[],
            ),
        )

    for m in _COMMA_SYMPTOM.finditer(text):
        _add_unique(
            entities,
            _span(
                text,
                m.start(),
                m.end(),
                type="TRIỆU_CHỨNG",
                assertions=[],
                candidates=[],
            ),
        )

    for m in _LAB_PAIR.finditer(text):
        name, value = m.group(1).strip(), m.group(2).strip()
        ns, ne = m.start(1), m.start(1) + len(name)
        vs, ve = m.start(2), m.start(2) + len(value)
        _add_unique(
            entities,
            _span(text, ns, ne, type="TÊN_XÉT_NGHIỆM", assertions=[], candidates=[]),
        )
        _add_unique(
            entities,
            _span(text, vs, ve, type="KẾT_QUẢ_XÉT_NGHIỆM", assertions=[], candidates=[]),
        )

    entities.sort(key=lambda e: e["position"][0])
    return entities


class RuleExtractor:
    """Rule/heuristic extractor — replace with model-based Extractor in Week 3."""

    def extract(self, text: str) -> list[MedicalEntity]:
        return extract_entities(text)
