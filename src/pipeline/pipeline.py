"""End-to-end clinical NLP pipeline (composition over hard-wired functions)."""

from __future__ import annotations

import json
from pathlib import Path

from src.extract.rules import RuleExtractor
from src.kb.load import load_kb_root
from src.linking.fuzzy import FuzzyLinker
from src.pipeline.protocols import Extractor, Linker
from src.schemas.entity import LINKED_TYPES
from src.schemas.validate import assert_valid


class ClinicalNLPipeline:
    def __init__(self, extractor: Extractor, linker: Linker) -> None:
        self._extractor = extractor
        self._linker = linker

    def process(self, text: str) -> list[dict]:
        entities = self._extractor.extract(text)
        out: list[dict] = []
        for ent in entities:
            row = dict(ent)
            etype = row["type"]
            if etype in LINKED_TYPES:
                row["candidates"] = self._linker.link(row["text"], etype)
            else:
                row["candidates"] = []
            out.append(row)
        assert_valid(out, text)
        return out


def build_default_pipeline(kb_root: Path) -> ClinicalNLPipeline:
    """Week 2 baseline: rule extractor + fuzzy linker."""
    kb = load_kb_root(kb_root)
    return ClinicalNLPipeline(RuleExtractor(), FuzzyLinker(kb))


def write_predictions(
    pipeline: ClinicalNLPipeline,
    input_dir: Path,
    output_dir: Path,
) -> int:
    """Run *pipeline* on every ``*.txt`` in *input_dir*; write JSON alongside."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        input_dir.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        entities = pipeline.process(text)
        out_path = output_dir / f"{path.stem}.json"
        out_path.write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return len(files)
