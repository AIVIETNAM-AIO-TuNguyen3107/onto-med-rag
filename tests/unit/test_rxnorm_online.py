from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from clinical_nlp.rxnorm_linking import RxNormIndex


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_exact_lookup_fetches_properties_and_reuses_cache(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"idGroup": {"rxnormId": ["999"]}}),
            FakeResponse(
                200,
                {
                    "properties": {
                        "name": "example 10 MG Oral Tablet",
                        "tty": "SCD",
                    }
                },
            ),
        ]
    )
    cache = tmp_path / "rxnorm.json"
    index = RxNormIndex(cache, use_api=True, session=session)

    rows = index.retrieve("example 10 mg")
    assert rows[0].identifier == "999"
    assert rows[0].terminology_type == "SCD"
    assert session.calls[0]["url"].endswith("/rxcui.json")
    assert cache.exists()

    offline = RxNormIndex(cache, use_api=False, session=FakeSession([]))
    assert offline.retrieve("example 10 mg")[0].identifier == "999"


def test_approximate_fallback_deduplicates_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "clinical_nlp.rxnorm_linking.index.time.sleep",
        lambda _: None,
    )
    session = FakeSession(
        [
            requests.Timeout("temporary"),
            FakeResponse(200, {"idGroup": {}}),
            FakeResponse(
                200,
                {
                    "approximateGroup": {
                        "candidate": [
                            {"rxcui": "998", "rank": "1", "score": "9.0"},
                            {"rxcui": "998", "rank": "2", "score": "8.0"},
                        ]
                    }
                },
            ),
            FakeResponse(
                200,
                {"properties": {"name": "fallback tablet", "tty": "SCD"}},
            ),
        ]
    )
    index = RxNormIndex(
        tmp_path / "rxnorm.json",
        use_api=True,
        session=session,
    )

    rows = index.retrieve("misspelled fallback")
    assert [row.identifier for row in rows] == ["998"]
    assert rows[0].retrieval_sources == ["rxnav_approximate"]
    assert any(
        call["url"].endswith("/approximateTerm.json")
        for call in session.calls
    )


def test_query_variants_are_generated_for_route_and_release() -> None:
    from clinical_nlp.rxnorm_linking.index import query_variants

    variants = query_variants("metoprolol succinate xl 50 mg po daily")
    assert any("extended release" in value for value in variants)
    assert any("oral" in value for value in variants)


def test_rxnav_does_not_retry_non_retryable_4xx(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(404, {})])
    index = RxNormIndex(
        tmp_path / "rxnorm.json",
        use_api=True,
        session=session,
    )

    with pytest.raises(RuntimeError, match="status 404"):
        index.retrieve("missing")

    assert len(session.calls) == 1
