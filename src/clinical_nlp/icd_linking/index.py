from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from openpyxl import load_workbook

from clinical_nlp.normalization import normalize_search
from clinical_nlp.schemas import LinkCandidate


CODE_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$")
CURATED_ALIASES = {
    "thiếu men g6pd": {"D55.0"},
    "thiếu máu": {"D64.9"},
}


@dataclass(frozen=True)
class ICDConcept:
    code: str
    names: tuple[str, ...]


class ICDIndex:
    def __init__(
        self,
        concepts: dict[str, ICDConcept],
        source_sha256: str | None = None,
    ) -> None:
        self.concepts = concepts
        self.source_sha256 = source_sha256
        aliases: dict[str, set[str]] = defaultdict(set)
        for code, concept in concepts.items():
            for name in concept.names:
                key = normalize_search(name)
                if key:
                    aliases[key].add(code)
        for alias, codes in CURATED_ALIASES.items():
            aliases[alias].update(code for code in codes if code in concepts)
        self.alias_to_codes = dict(aliases)
        token_aliases: dict[str, set[str]] = defaultdict(set)
        for alias in self.alias_to_codes:
            for token in set(alias.split()):
                token_aliases[token].add(alias)
        self.token_aliases = dict(token_aliases)

    @classmethod
    def from_workbook(cls, path: Path) -> "ICDIndex":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        names_by_code: dict[str, set[str]] = defaultdict(set)
        for row in sheet.iter_rows(min_row=6, values_only=True):
            raw_code = "" if row[1] is None else str(row[1]).strip()
            name = "" if row[2] is None else str(row[2]).strip()
            code = raw_code.rstrip("*†")
            if not CODE_RE.fullmatch(code) or not name:
                continue
            names_by_code[code].add(name)
        concepts = {
            code: ICDConcept(code=code, names=tuple(sorted(names)))
            for code, names in names_by_code.items()
        }
        return cls(concepts=concepts, source_sha256=digest)

    @classmethod
    def load(cls, path: Path) -> "ICDIndex":
        raw = json.loads(path.read_text("utf-8"))
        concepts = {
            row["code"]: ICDConcept(
                code=row["code"],
                names=tuple(row["names"]),
            )
            for row in raw["concepts"]
        }
        return cls(concepts=concepts, source_sha256=raw.get("source_sha256"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_sha256": self.source_sha256,
            "concepts": [
                {"code": concept.code, "names": list(concept.names)}
                for concept in sorted(self.concepts.values(), key=lambda row: row.code)
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def contains(self, code: str) -> bool:
        return code in self.concepts

    def retrieve(self, mention: str, limit: int = 10) -> list[LinkCandidate]:
        query = normalize_search(mention)
        if not query:
            return []
        exact = self.alias_to_codes.get(query, set())
        results: dict[str, LinkCandidate] = {}
        for code in exact:
            if code.startswith("Z"):
                continue
            concept = self.concepts[code]
            results[code] = LinkCandidate(
                identifier=code,
                name=concept.names[0],
                terminology_type="ICD10",
                score=1.0,
                retrieval_sources=["exact"],
                component_scores={"lexical": 1.0},
            )
        if len(results) >= limit:
            return sorted(results.values(), key=lambda row: (-row.score, row.identifier))[
                :limit
            ]

        query_tokens = set(query.split())
        candidate_aliases: set[str] = set()
        for token in query_tokens:
            candidate_aliases.update(self.token_aliases.get(token, ()))
        scored_aliases: list[tuple[float, str, set[str]]] = []
        for alias in candidate_aliases:
            codes = self.alias_to_codes[alias]
            alias_tokens = set(alias.split())
            token_score = (
                len(query_tokens & alias_tokens) / len(query_tokens | alias_tokens)
                if query_tokens | alias_tokens
                else 0.0
            )
            if token_score < 0.25 and query not in alias and alias not in query:
                continue
            char_score = SequenceMatcher(None, query, alias).ratio()
            score = 0.55 * char_score + 0.45 * token_score
            if score >= 0.48:
                scored_aliases.append((score, alias, codes))
        scored_aliases.sort(key=lambda row: (-row[0], len(row[1])))
        for score, alias, codes in scored_aliases:
            for code in codes:
                if code.startswith("Z"):
                    continue
                existing = results.get(code)
                if existing is None or score > existing.score:
                    concept = self.concepts[code]
                    results[code] = LinkCandidate(
                        identifier=code,
                        name=concept.names[0],
                        terminology_type="ICD10",
                        score=score,
                        retrieval_sources=["fuzzy"],
                        component_scores={"lexical": score},
                    )
            if len(results) >= limit * 3:
                break
        return sorted(results.values(), key=lambda row: (-row.score, row.identifier))[
            :limit
        ]

    def all_aliases(self, min_length: int = 4) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for concept in self.concepts.values():
            for name in concept.names:
                if len(name) >= min_length:
                    rows.append((name, concept.code))
        return rows
