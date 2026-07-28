"""Rewrite raw-batch entity text to the exact source slice it resolves to.

Input documents are inconsistently normalized: some store Vietnamese
precomposed, others as a base character plus combining tone mark.
``find_occurrence`` matches in NFD space and locates the correct span either
way, but emitted text must equal ``original_text[start:end]`` verbatim.

This script replaces only the emitted form at an already-resolved offset. It
never moves or guesses a span. Anything that cannot be resolved is left
untouched and reported.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from clinical_nlp.text import find_occurrence


def main(paths: list[str]) -> int:
    unresolved: list[tuple[str, str]] = []
    for path in paths:
        target = Path(path)
        batch = json.loads(target.read_text(encoding="utf-8"))
        rewritten = 0
        for document in batch:
            text = Path(f"input/{document['document_id']}.txt").read_text(
                encoding="utf-8"
            )
            for entity in document["entities"]:
                try:
                    start, end = find_occurrence(
                        text, entity["text"], entity["occurrence"]
                    )
                except ValueError:
                    unresolved.append((document["document_id"], entity["text"]))
                    continue
                if text[start:end] != entity["text"]:
                    entity["text"] = text[start:end]
                    rewritten += 1
        target.write_text(
            json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{target.name}: rewrote {rewritten} span texts")
    for item in unresolved:
        print("UNRESOLVED", item)
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
