"""Validate competition output entities against schema rules."""

from __future__ import annotations

from src.schemas.entity import ASSERTION_ELIGIBLE, ASSERTION_TYPES, LINKED_TYPES

ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "TRIỆU_CHỨNG",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
        "CHẨN_ĐOÁN",
        "THUỐC",
    }
)


def validate_entities(entities: list[dict], source_text: str) -> list[str]:
    """Return human-readable validation errors (empty list = valid)."""
    errors: list[str] = []

    for i, ent in enumerate(entities):
        prefix = f"entity[{i}]"

        for key in ("text", "position", "type", "assertions", "candidates"):
            if key not in ent:
                errors.append(f"{prefix}: missing field '{key}'")

        if errors:
            continue

        text = ent["text"]
        if not isinstance(text, str) or not text:
            errors.append(f"{prefix}: text must be a non-empty string")

        pos = ent["position"]
        if (
            not isinstance(pos, list)
            or len(pos) != 2
            or not all(isinstance(x, int) for x in pos)
            or pos[0] < 0
            or pos[1] <= pos[0]
        ):
            errors.append(f"{prefix}: position must be [start, end) with start < end")
        elif isinstance(text, str) and text != source_text[pos[0] : pos[1]]:
            errors.append(
                f"{prefix}: text does not match source_text[{pos[0]}:{pos[1]}]"
            )

        etype = ent["type"]
        if etype not in ENTITY_TYPES:
            errors.append(f"{prefix}: invalid type '{etype}'")

        assertions = ent["assertions"]
        if not isinstance(assertions, list):
            errors.append(f"{prefix}: assertions must be a list")
        elif len(assertions) > 3:
            errors.append(f"{prefix}: at most 3 assertions allowed")
        elif isinstance(etype, str) and etype in ENTITY_TYPES:
            if etype not in ASSERTION_ELIGIBLE and assertions:
                errors.append(f"{prefix}: assertions not allowed for type {etype}")
            for a in assertions:
                if a not in ASSERTION_TYPES:
                    errors.append(f"{prefix}: invalid assertion '{a}'")

        candidates = ent["candidates"]
        if not isinstance(candidates, list):
            errors.append(f"{prefix}: candidates must be a list")
        elif isinstance(etype, str) and etype in ENTITY_TYPES:
            if etype not in LINKED_TYPES and candidates:
                errors.append(f"{prefix}: candidates not allowed for type {etype}")

    return errors


def assert_valid(entities: list[dict], source_text: str) -> None:
    errors = validate_entities(entities, source_text)
    if errors:
        raise ValueError("\n".join(errors))
