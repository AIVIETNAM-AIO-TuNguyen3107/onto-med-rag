from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from clinical_nlp.normalization import normalize_search
from clinical_nlp.schemas import LinkCandidate


SEED_CONCEPTS: dict[str, dict[str, str]] = {
    "308135": {"name": "amlodipine 10 MG Oral Tablet", "tty": "SCD"},
    "243670": {"name": "aspirin 81 MG Oral Tablet", "tty": "SCD"},
    "866436": {
        "name": "24 HR metoprolol succinate 50 MG Extended Release Oral Tablet",
        "tty": "SCD",
    },
    "392085": {"name": "guaifenesin 800 MG Oral Tablet", "tty": "SCD"},
    "7597": {"name": "nystatin", "tty": "IN"},
    "313782": {"name": "acetaminophen 325 MG Oral Tablet", "tty": "SCD"},
    "904475": {"name": "pravastatin sodium 40 MG Oral Tablet", "tty": "SCD"},
    "1099279": {"name": "docusate sodium 100 MG Oral Tablet", "tty": "SCD"},
    "312935": {"name": "sennosides, USP 8.6 MG Oral Tablet", "tty": "SCD"},
    "197527": {"name": "clonazepam 0.5 MG Oral Tablet", "tty": "SCD"},
    "197528": {"name": "clonazepam 1 MG Oral Tablet", "tty": "SCD"},
}


class RxNormIndex:
    def __init__(self, cache_path: Path, use_api: bool = True) -> None:
        self.cache_path = cache_path
        self.use_api = use_api
        self.concepts: dict[str, dict[str, str]] = dict(SEED_CONCEPTS)
        self.queries: dict[str, list[str]] = {}
        if cache_path.exists():
            raw = json.loads(cache_path.read_text("utf-8"))
            self.concepts.update(raw.get("concepts", {}))
            self.queries.update(raw.get("queries", {}))

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {"concepts": self.concepts, "queries": self.queries},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def contains(self, rxcui: str) -> bool:
        return rxcui in self.concepts

    def _api_json(self, url: str) -> dict[str, Any]:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _fetch_properties(self, rxcui: str) -> None:
        if rxcui in self.concepts:
            return
        raw = self._api_json(
            f"https://rxnav.nlm.nih.gov/REST/rxcui/{quote(rxcui)}/properties.json"
        )
        properties = raw.get("properties")
        if properties:
            self.concepts[rxcui] = {
                "name": properties.get("name", ""),
                "tty": properties.get("tty", ""),
            }

    def retrieve(self, mention: str, limit: int = 10) -> list[LinkCandidate]:
        key = normalize_search(mention)
        rxcuis = list(self.queries.get(key, []))
        local_scores: list[tuple[float, str]] = []
        query_tokens = set(key.split())
        for rxcui, concept in self.concepts.items():
            name_key = normalize_search(concept.get("name", ""))
            name_tokens = set(name_key.split())
            char_score = SequenceMatcher(None, key, name_key).ratio()
            token_score = (
                len(query_tokens & name_tokens) / len(query_tokens | name_tokens)
                if query_tokens | name_tokens
                else 0.0
            )
            score = 0.55 * char_score + 0.45 * token_score
            if score >= 0.34:
                local_scores.append((score, rxcui))
        local_scores.sort(reverse=True)
        rxcuis = list(
            dict.fromkeys(rxcuis + [rxcui for _, rxcui in local_scores[: limit * 2]])
        )
        if not rxcuis and self.use_api:
            url = (
                "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
                f"?term={quote(mention)}&maxEntries={max(limit * 3, 20)}"
            )
            raw = self._api_json(url)
            candidates = (
                raw.get("approximateGroup", {})
                .get("candidate", [])
            )
            rxcuis = [str(row["rxcui"]) for row in candidates if row.get("rxcui")]
            self.queries[key] = list(dict.fromkeys(rxcuis))
            for rxcui in self.queries[key][: max(limit * 2, 10)]:
                self._fetch_properties(rxcui)
            self.save()

        query = normalize_search(mention)
        results: list[LinkCandidate] = []
        for rxcui in rxcuis:
            concept = self.concepts.get(rxcui)
            if not concept:
                continue
            name = concept.get("name", "")
            name_key = normalize_search(name)
            char_score = SequenceMatcher(None, query, name_key).ratio()
            left_tokens = set(query.split())
            right_tokens = set(name_key.split())
            token_score = (
                len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
                if left_tokens | right_tokens
                else 0.0
            )
            lexical = 0.55 * char_score + 0.45 * token_score
            results.append(
                LinkCandidate(
                    identifier=rxcui,
                    name=name,
                    terminology_type=concept.get("tty"),
                    score=lexical,
                    retrieval_sources=["rxnav_approximate"],
                    component_scores={"lexical": lexical},
                )
            )
        results.sort(key=lambda row: (-row.score, row.identifier))
        return results[:limit]


DRUG_CORE_RE = re.compile(
    r"(?i)\b(?:po|oral|iv|im|sc|sq|sl|topical|inh|inhaled|"
    r"daily|bid|tid|qid|qhs|qam|q\d+h(?::?prn)?|prn|stat|xl|xr|er)\b"
)


def query_variants(mention: str) -> list[str]:
    variants = [mention]
    reduced = re.sub(
        r"(?i)\b(?:po|oral|iv|im|sc|sq|sl|topical|inh|inhaled|"
        r"daily|bid|tid|qid|qhs|qam|q\d+h(?::?prn)?|prn|stat)\b",
        " ",
        mention,
    )
    reduced = re.sub(r"\s+", " ", reduced).strip()
    if reduced and reduced != mention:
        variants.append(reduced)
    return list(dict.fromkeys(variants))
