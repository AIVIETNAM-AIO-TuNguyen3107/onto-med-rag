"""Fuzzy concept linking via RapidFuzz."""

from __future__ import annotations

from rapidfuzz import fuzz, process


def link_concept(
    name: str,
    index: list[tuple[str, str]],
    top_k: int = 3,
    score_cutoff: float = 60.0,
) -> list[str]:
    """Return top-k concept codes for *name* using token-set ratio."""
    if not name.strip() or not index:
        return []

    labels = [label for _code, label in index]
    matches = process.extract(
        name,
        labels,
        scorer=fuzz.token_set_ratio,
        limit=top_k,
        score_cutoff=score_cutoff,
    )
    label_to_code = {label: code for code, label in index}
    return [label_to_code[match[0]] for match in matches if match[0] in label_to_code]


class FuzzyLinker:
    """RapidFuzz linker — replace or wrap for embedding rerank later."""

    def __init__(
        self,
        kb: dict[str, list[tuple[str, str]]],
        top_k: int = 3,
        score_cutoff: float = 60.0,
    ) -> None:
        self._kb = kb
        self._top_k = top_k
        self._score_cutoff = score_cutoff

    def link(self, text: str, entity_type: str) -> list[str]:
        if entity_type == "CHẨN_ĐOÁN":
            index = self._kb["icd10"]
        elif entity_type == "THUỐC":
            index = self._kb["rxnorm"]
        else:
            return []
        return link_concept(text, index, self._top_k, self._score_cutoff)
