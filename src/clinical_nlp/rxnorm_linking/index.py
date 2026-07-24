from __future__ import annotations

import json
import re
import time
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
    def __init__(
        self,
        cache_path: Path,
        use_api: bool = True,
        session: requests.Session | None = None,
        base_url: str = "https://rxnav.nlm.nih.gov/REST",
        max_retries: int = 3,
    ) -> None:
        self.cache_path = cache_path
        self.use_api = use_api
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.concepts: dict[str, dict[str, str]] = dict(SEED_CONCEPTS)
        self.queries: dict[str, list[dict[str, Any]]] = {}
        if cache_path.exists():
            raw = json.loads(cache_path.read_text("utf-8"))
            self.concepts.update(raw.get("concepts", {}))
            for key, rows in raw.get("queries", {}).items():
                if rows and isinstance(rows[0], str):
                    self.queries[key] = [
                        {
                            "rxcui": value,
                            "source": "rxnorm_legacy_cache",
                            "rank": index,
                        }
                        for index, value in enumerate(rows, start=1)
                    ]
                else:
                    self.queries[key] = rows

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 2,
                    "concepts": self.concepts,
                    "queries": self.queries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    def contains(self, rxcui: str) -> bool:
        return rxcui in self.concepts

    def _api_json(self, resource: str, params: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    f"{self.base_url}/{resource.lstrip('/')}",
                    params=params,
                    timeout=30,
                    headers={"User-Agent": "clinical-nlp-rxnorm/0.1"},
                )
                if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    raise _RetryableRxNavError(response)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                raise RuntimeError(
                    f"RxNav request failed with non-retryable HTTP status {status}"
                ) from exc
            except (
                requests.RequestException,
                _RetryableRxNavError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(_retry_delay(exc, attempt))
        raise RuntimeError(
            f"RxNav request failed after {self.max_retries + 1} attempts"
        ) from last_error

    def _fetch_properties(self, rxcui: str) -> None:
        if rxcui in self.concepts:
            return
        raw = self._api_json(
            f"rxcui/{quote(rxcui)}/properties.json",
            {},
        )
        properties = raw.get("properties")
        if properties:
            self.concepts[rxcui] = {
                "name": properties.get("name", ""),
                "tty": properties.get("tty", ""),
            }

    def _lookup_online(self, mention: str, limit: int) -> list[dict[str, Any]]:
        key = normalize_search(mention)
        cached = self.queries.get(key)
        if cached is not None:
            for row in cached:
                self._fetch_properties(row["rxcui"])
            return cached
        exact = self._api_json(
            "rxcui.json",
            {"name": mention, "search": "2", "allsrc": "0"},
        )
        identifiers = exact.get("idGroup", {}).get("rxnormId", []) or []
        rows = [
            {
                "rxcui": str(identifier),
                "source": "rxnav_exact_or_normalized",
                "rank": rank,
                "raw_score": 1.0,
            }
            for rank, identifier in enumerate(identifiers, start=1)
        ]
        if not rows:
            approximate = self._api_json(
                "approximateTerm.json",
                {
                    "term": mention,
                    "maxEntries": str(min(max(limit, 20), 100)),
                    "option": "1",
                },
            )
            candidates = (
                approximate.get("approximateGroup", {}).get("candidate", []) or []
            )
            rows = [
                {
                    "rxcui": str(row["rxcui"]),
                    "source": "rxnav_approximate",
                    "rank": int(row.get("rank", fallback_rank)),
                    "raw_score": float(row.get("score", 0.0)),
                }
                for fallback_rank, row in enumerate(candidates, start=1)
                if row.get("rxcui")
            ]
        deduplicated: dict[str, dict[str, Any]] = {}
        for row in rows:
            old = deduplicated.get(row["rxcui"])
            if old is None or row["rank"] < old["rank"]:
                deduplicated[row["rxcui"]] = row
        selected = sorted(
            deduplicated.values(),
            key=lambda row: (row["rank"], row["rxcui"]),
        )[:limit]
        for row in selected:
            self._fetch_properties(row["rxcui"])
        self.queries[key] = selected
        self.save()
        return selected

    def _local_candidates(self, key: str, limit: int) -> list[dict[str, Any]]:
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
            if score >= 0.55:
                local_scores.append((score, rxcui))
        local_scores.sort(reverse=True)
        return [
            {
                "rxcui": rxcui,
                "source": "rxnorm_local_lexical",
                "rank": rank,
                "raw_score": score,
            }
            for rank, (score, rxcui) in enumerate(local_scores[:limit], start=1)
        ]

    def retrieve(self, mention: str, limit: int = 20) -> list[LinkCandidate]:
        key = normalize_search(mention)
        if not key:
            return []
        rows: list[dict[str, Any]] = []
        if self.use_api:
            rows.extend(self._lookup_online(mention, limit))
        rows.extend(self._local_candidates(key, limit))
        by_identifier: dict[str, dict[str, Any]] = {}
        for row in rows:
            old = by_identifier.get(row["rxcui"])
            if old is None or row["rank"] < old["rank"]:
                by_identifier[row["rxcui"]] = row
        query = normalize_search(mention)
        results: list[LinkCandidate] = []
        for rxcui, retrieval in by_identifier.items():
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
            tty = concept.get("tty", "")
            specificity_bonus = {
                "SCD": 0.08,
                "IN": 0.03,
                "SCDC": 0.02,
                "SCDF": 0.01,
            }.get(tty, 0.0)
            score = min(1.0, lexical + specificity_bonus)
            results.append(
                LinkCandidate(
                    identifier=rxcui,
                    name=name,
                    terminology_type=tty,
                    score=score,
                    retrieval_sources=[retrieval["source"]],
                    component_scores={
                        "lexical": lexical,
                        "specificity_bonus": specificity_bonus,
                        "api_raw_score": float(retrieval.get("raw_score", 0.0)),
                    },
                    metadata={"api_rank": retrieval["rank"]},
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
    clinical = re.sub(r"(?i)\bpo\b", "oral", mention)
    clinical = re.sub(r"(?i)\b(?:xl|xr|er)\b", "extended release", clinical)
    clinical = re.sub(
        r"(?i)\b(?:daily|bid|tid|qid|qhs|qam|q\d+h(?::?prn)?|prn|stat)\b",
        " ",
        clinical,
    )
    clinical = re.sub(r"\s+", " ", clinical).strip(" :")
    if clinical and clinical != mention:
        variants.append(clinical)
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


class _RetryableRxNavError(Exception):
    def __init__(self, response: requests.Response) -> None:
        super().__init__(f"retryable RxNav HTTP status {response.status_code}")
        self.response = response


def _retry_delay(exc: Exception, attempt: int) -> float:
    if isinstance(exc, _RetryableRxNavError):
        value = exc.response.headers.get("Retry-After")
        if value:
            try:
                return min(float(value), 30.0)
            except ValueError:
                pass
    return min(2.0**attempt, 10.0)
