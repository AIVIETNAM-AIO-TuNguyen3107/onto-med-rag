"""Load ICD-10 / RxNorm concept indexes from TSV files."""

from __future__ import annotations

from pathlib import Path


def load_concepts(path: Path) -> list[tuple[str, str]]:
    """Load ``code \\t name`` rows from a TSV file.

    Extra columns are ignored. Returns empty list if file missing.
    """
    if not path.is_file():
        return []

    concepts: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, name = parts[0].strip(), parts[1].strip()
        if code and name:
            concepts.append((code, name))
    return concepts


def load_kb_root(kb_root: Path) -> dict[str, list[tuple[str, str]]]:
    """Load ICD-10 and RxNorm indexes from standard layout under *kb_root*."""
    icd_path = kb_root / "icd10" / "concepts.tsv"
    rx_path = kb_root / "rxnorm" / "concepts.tsv"

    # ponytail: fall back to bundled sample KBs for dev without full downloads
    if not icd_path.is_file():
        icd_path = kb_root / "icd10" / "concepts.sample.tsv"
    if not rx_path.is_file():
        rx_path = kb_root / "rxnorm" / "concepts.sample.tsv"

    return {
        "icd10": load_concepts(icd_path),
        "rxnorm": load_concepts(rx_path),
    }
