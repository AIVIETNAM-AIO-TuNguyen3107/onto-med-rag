from __future__ import annotations

import json
from pathlib import Path

from clinical_nlp.schemas import Document, Entity


def validate_entities(document: Document, entities: list[Entity]) -> None:
    previous_end = -1
    seen: set[tuple[int, int, str]] = set()
    for entity in entities:
        start, end = entity.position
        if start < previous_end:
            raise ValueError(f"overlap or unsorted entity at {entity.position}")
        if document.text[start:end] != entity.text:
            raise ValueError(f"substring mismatch at {entity.position}")
        key = (start, end, entity.type.value)
        if key in seen:
            raise ValueError(f"duplicate entity at {entity.position}")
        seen.add(key)
        previous_end = end


def write_entities(
    path: Path,
    document: Document,
    entities: list[Entity],
    pretty: bool = True,
) -> None:
    validate_entities(document, entities)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entity.output_dict() for entity in entities]
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def validate_output_directory(
    output_dir: Path,
    input_dir: Path,
    expected_stems: set[str] | None = None,
) -> None:
    input_stems = (
        expected_stems
        if expected_stems is not None
        else {path.stem for path in input_dir.glob("*.txt")}
    )
    missing_inputs = [
        stem for stem in input_stems if not (input_dir / f"{stem}.txt").exists()
    ]
    if missing_inputs:
        raise ValueError(f"selected input files do not exist: {sorted(missing_inputs)}")
    output_files = list(output_dir.glob("*.json"))
    output_stems = {path.stem for path in output_files}
    if output_stems != input_stems:
        missing = sorted(input_stems - output_stems)
        extra = sorted(output_stems - input_stems)
        raise ValueError(f"output stem mismatch: missing={missing}, extra={extra}")
    for path in output_files:
        payload = json.loads(path.read_text("utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array")
        document = Document(
            id=path.stem,
            text=(input_dir / f"{path.stem}.txt").read_text("utf-8"),
        )
        entities = [Entity.model_validate(row) for row in payload]
        validate_entities(document, entities)
