from __future__ import annotations

from typing import Any, Iterator

import httpx

from src.crawler.core import CrawlError, CrawlRecord


class HttpJsonFetcher:
    """Fetches records from a single-response JSON endpoint.

    This adapter only supports sources that return all records in one
    HTTP call (e.g. RxNorm's ``allconcepts`` endpoint). It does not
    implement pagination; a paginated source (e.g. an ICD-10 API) needs
    this adapter extended first.
    """

    def __init__(
        self,
        base_url: str,
        params: dict[str, Any],
        records_path: str,
        id_field: str,
        next_page_field: str | None,
        source_name: str,
    ) -> None:
        if next_page_field is not None:
            raise NotImplementedError(
                "HttpJsonFetcher does not implement pagination yet; "
                "next_page_field must be None (see docs/superpowers/specs/2026-07-26-crawler-module-design.md)"
            )
        self.base_url = base_url
        self.params = params
        self.records_path = records_path
        self.id_field = id_field
        self.next_page_field = next_page_field
        self.source_name = source_name

    def fetch(self) -> Iterator[CrawlRecord]:
        response = httpx.get(self.base_url, params=self.params, timeout=30.0)
        if response.status_code != 200:
            raise CrawlError(
                f"{self.base_url} returned HTTP {response.status_code}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise CrawlError(f"{self.base_url} returned invalid JSON") from exc

        records = self._resolve_path(body, self.records_path)
        if not isinstance(records, list):
            raise CrawlError(
                f"records_path {self.records_path!r} did not resolve to a list "
                f"in the response from {self.base_url}"
            )

        for record in records:
            if self.id_field not in record:
                raise CrawlError(
                    f"id_field {self.id_field!r} missing from record {record!r}"
                )
            yield CrawlRecord(
                id=str(record[self.id_field]),
                data=record,
                source=self.source_name,
            )

    @staticmethod
    def _resolve_path(body: Any, path: str) -> Any:
        if not path:
            return body
        value = body
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                raise CrawlError(f"path {path!r} not found in response body")
            value = value[key]
        return value
