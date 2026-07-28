"""Read-only audit of raw direct-extraction batches.

Resolve every ``(text, occurrence)`` with ``find_occurrence``, assert the exact
source-substring invariant, and report overlaps, duplicates, masked spans, and
count calibration. Report failures rather than repairing them.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

from clinical_nlp.text import find_occurrence, is_masked_span

RAW = Path("artifacts/direct-extraction/raw")
BASELINE = Path("runs/full100-v1/outputs")


def main(paths: list[str]) -> int:
    failures: list[tuple[str, str, str]] = []
    per_doc: dict[str, int] = {}
    types: collections.Counter[str] = collections.Counter()
    assertions: collections.Counter[str] = collections.Counter()

    for path in paths:
        for document in json.loads(Path(path).read_text(encoding="utf-8")):
            document_id = document["document_id"]
            text = Path(f"input/{document_id}.txt").read_text(encoding="utf-8")
            per_doc[document_id] = len(document["entities"])
            resolved: list[tuple[int, int, str]] = []
            seen: collections.Counter[tuple[str, int]] = collections.Counter()

            for entity in document["entities"]:
                types[entity["type"]] += 1
                assertions.update(entity["assertions"])
                key = (entity["text"], entity["occurrence"])
                seen[key] += 1
                if seen[key] > 1:
                    failures.append((document_id, entity["text"], "DUPLICATE"))
                if is_masked_span(entity["text"]):
                    failures.append((document_id, entity["text"], "MASKED"))
                try:
                    start, end = find_occurrence(
                        text, entity["text"], entity["occurrence"]
                    )
                except ValueError as error:
                    failures.append(
                        (
                            document_id,
                            entity["text"],
                            f"occ={entity['occurrence']} {error}",
                        )
                    )
                    continue
                if text[start:end] != entity["text"]:
                    failures.append(
                        (
                            document_id,
                            entity["text"],
                            f"ROUNDTRIP {text[start:end]!r}",
                        )
                    )
                    continue
                resolved.append((start, end, entity["text"]))

            resolved.sort()
            for earlier, later in zip(resolved, resolved[1:]):
                if later[0] < earlier[1]:
                    failures.append(
                        (document_id, later[2], f"OVERLAPS {earlier[2]!r}")
                    )

    total = sum(per_doc.values())
    baseline = sum(
        len(json.loads((BASELINE / f"{document_id}.json").read_text("utf-8")))
        for document_id in per_doc
    )
    print(
        f"documents {len(per_doc)}  entities {total}  baseline {baseline}"
        f"  ratio {total / baseline:.2f}"
    )
    print(f"assertions {sum(assertions.values())}  {dict(assertions)}")
    print(f"types {dict(types)}")
    print("per-doc", {key: per_doc[key] for key in sorted(per_doc, key=int)})
    print(f"\nFAILURES {len(failures)}")
    for failure in failures:
        print("  ", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    requested = sys.argv[1:] or [str(path) for path in sorted(RAW.glob("*.json"))]
    raise SystemExit(main(requested))
