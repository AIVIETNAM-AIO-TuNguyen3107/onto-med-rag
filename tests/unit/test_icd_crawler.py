from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import requests

from clinical_nlp import cli
from clinical_nlp.config import PathsConfig
from clinical_nlp.icd_linking import ICDCatalogCrawler, ICDCrawlOutput, ICDIndex
from clinical_nlp.icd_linking.crawler import (
    ICDCrawlDataError,
    ICDCrawlHTTPError,
    ICDCrawlSchemaError,
    crawl_to_files,
)


API_BASE = "https://example.test/api/ICD10_TT06"


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(
        self,
        routes: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            list[FakeResponse | requests.RequestException],
        ],
    ) -> None:
        self.routes = {key: list(values) for key, values in routes.items()}
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
        timeout: tuple[float, float],
    ) -> FakeResponse:
        del headers, timeout
        key = (url, tuple(sorted(params.items())))
        self.calls.append(key)
        if key not in self.routes or not self.routes[key]:
            raise AssertionError(f"unexpected HTTP request: {key}")
        result = self.routes[key].pop(0)
        if isinstance(result, requests.RequestException):
            raise result
        return result


def _key(path: str, **params: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    return f"{API_BASE}/{path}", tuple(sorted(params.items()))


def _item(
    model: str,
    node_id: str,
    code: str,
    name: str,
    *,
    is_leaf: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "id": node_id,
        "is_leaf": is_leaf,
        "data": {
            "id": node_id,
            "code": code,
            "name": name,
            "html": None,
        },
    }


def _success(*items: dict[str, Any]) -> FakeResponse:
    return FakeResponse(200, {"status": "success", "data": list(items)})


def _miniature_routes() -> dict[
    tuple[str, tuple[tuple[str, str], ...]],
    list[FakeResponse | requests.RequestException],
]:
    return {
        _key("root", lang="vi"): [
            _success(
                _item(
                    "chapter",
                    "I",
                    "I",
                    "Bệnh truyền nhiễm và ký sinh trùng",
                    is_leaf=False,
                )
            )
        ],
        _key("childs/chapter", id="I", lang="vi"): [
            _success(
                _item(
                    "section",
                    "A00",
                    "A00",
                    "Bệnh truyền nhiễm đường ruột",
                    is_leaf=False,
                )
            )
        ],
        _key("childs/section", id="A00", lang="vi"): [
            _success(
                _item("type", "A00", "A00", "Bệnh tả", is_leaf=False)
            )
        ],
        _key("childs/type", id="A00", lang="vi"): [
            _success(
                _item(
                    "disease",
                    "A000",
                    "A00.0",
                    "Bệnh tả cổ điển",
                    is_leaf=True,
                ),
                _item(
                    "disease",
                    "A001",
                    "A00.1",
                    "Bệnh tả típ sinh học eltor",
                    is_leaf=True,
                ),
            )
        ],
    }


def _crawler(
    session: FakeSession,
    *,
    max_retries: int = 0,
    sleeper: Any = lambda _: None,
) -> ICDCatalogCrawler:
    return ICDCatalogCrawler(
        api_base_url=API_BASE,
        language="vi",
        request_delay_seconds=0,
        max_retries=max_retries,
        session=session,
        sleeper=sleeper,
    )


def test_crawl_exports_tree_and_builds_index(tmp_path: Path) -> None:
    session = FakeSession(_miniature_routes())
    crawler = _crawler(session)
    jsonl_path = tmp_path / "icd.jsonl"
    csv_path = tmp_path / "icd.csv"
    manifest_path = tmp_path / "icd.manifest.json"

    output = crawl_to_files(
        crawler,
        source_page_url="https://example.test/icd",
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
    )

    rows = [
        json.loads(line)
        for line in jsonl_path.read_text("utf-8").splitlines()
    ]
    assert [row["model"] for row in rows] == [
        "chapter",
        "section",
        "type",
        "disease",
        "disease",
    ]
    assert rows[0]["name_vi"] == "Bệnh truyền nhiễm và ký sinh trùng"
    assert rows[3]["parent_model"] == "type"
    assert rows[3]["parent_id"] == "A00"
    assert rows[3]["depth"] == 3
    assert rows[4]["sibling_order"] == 1
    assert output.record_count == 5
    assert output.request_count == 4
    assert len(session.calls) == 4

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert csv_rows[3]["code"] == "A00.0"
    assert csv_rows[3]["name_vi"] == rows[3]["name_vi"]

    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["counts_by_model"] == {
        "chapter": 1,
        "disease": 2,
        "section": 1,
        "type": 1,
    }
    assert manifest["files"]["jsonl"]["sha256"] == hashlib.sha256(
        jsonl_path.read_bytes()
    ).hexdigest()
    assert manifest["files"]["csv"]["sha256"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert not (tmp_path / "icd.jsonl.checkpoint.jsonl").exists()

    index = ICDIndex.from_source(jsonl_path)
    assert set(index.concepts) == {"A00", "A00.0", "A00.1"}
    assert index.retrieve("Bệnh tả cổ điển")[0].identifier == "A00.0"
    assert "Bệnh truyền nhiễm đường ruột" not in index.concepts["A00"].names


def test_resume_skips_completed_requests(tmp_path: Path) -> None:
    root = _success(
        _item("chapter", "I", "I", "Chương một", is_leaf=False),
        _item("chapter", "II", "II", "Chương hai", is_leaf=False),
    )
    first_session = FakeSession(
        {
            _key("root", lang="vi"): [root],
            _key("childs/chapter", id="I", lang="vi"): [
                _success(
                    _item(
                        "section",
                        "A00",
                        "A00",
                        "Nhóm thứ nhất",
                        is_leaf=True,
                    )
                )
            ],
            _key("childs/chapter", id="II", lang="vi"): [
                FakeResponse(500, {"status": "error"})
            ],
        }
    )
    checkpoint = tmp_path / "crawl.checkpoint.jsonl"
    with pytest.raises(ICDCrawlHTTPError):
        _crawler(first_session).crawl(checkpoint)

    second_session = FakeSession(
        {
            _key("childs/chapter", id="II", lang="vi"): [
                _success(
                    _item(
                        "section",
                        "B00",
                        "B00",
                        "Nhóm thứ hai",
                        is_leaf=True,
                    )
                )
            ]
        }
    )
    snapshot = _crawler(second_session).crawl(checkpoint, resume=True)

    assert len(second_session.calls) == 1
    assert second_session.calls[0][0].endswith("/childs/chapter")
    assert [row.id for row in snapshot.rows] == ["I", "II", "A00", "B00"]
    assert snapshot.request_count == 3


def test_shared_node_occurrences_keep_paths_and_reuse_children(
    tmp_path: Path,
) -> None:
    session = FakeSession(
        {
            _key("root", lang="vi"): [
                _success(
                    _item("chapter", "I", "I", "Chương một", is_leaf=False),
                    _item("chapter", "II", "II", "Chương hai", is_leaf=False),
                )
            ],
            _key("childs/chapter", id="I", lang="vi"): [
                _success(
                    _item("type", "A00", "A00", "Bệnh tả", is_leaf=False)
                )
            ],
            _key("childs/chapter", id="II", lang="vi"): [
                _success(
                    _item("type", "A00", "A00", "Bệnh tả", is_leaf=False)
                )
            ],
            _key("childs/type", id="A00", lang="vi"): [
                _success(
                    _item(
                        "disease",
                        "A000",
                        "A00.0",
                        "Bệnh tả cổ điển",
                        is_leaf=True,
                    )
                )
            ],
        }
    )

    snapshot = _crawler(session).crawl(tmp_path / "state.jsonl")

    type_rows = [row for row in snapshot.rows if row.model == "type"]
    disease_rows = [row for row in snapshot.rows if row.model == "disease"]
    assert len(type_rows) == 2
    assert len({row.path for row in type_rows}) == 2
    assert len(disease_rows) == 2
    assert {row.parent_path for row in disease_rows} == {
        row.path for row in type_rows
    }
    assert len(session.calls) == 4
    assert sum(call[0].endswith("/childs/type") for call in session.calls) == 1


def test_retry_honors_retry_after(tmp_path: Path) -> None:
    sleeps: list[float] = []
    session = FakeSession(
        {
            _key("root", lang="vi"): [
                FakeResponse(
                    429,
                    {"status": "error"},
                    headers={"Retry-After": "2"},
                ),
                _success(
                    _item(
                        "disease",
                        "A000",
                        "A00.0",
                        "Bệnh tả cổ điển",
                        is_leaf=True,
                    )
                ),
            ]
        }
    )
    crawler = _crawler(session, max_retries=1, sleeper=sleeps.append)

    snapshot = crawler.crawl(tmp_path / "state.jsonl")

    assert snapshot.request_count == 2
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "failure", "data": []},
        {"status": "success", "data": {}},
        {"status": "success", "data": [{"model": "chapter"}]},
    ],
)
def test_malformed_api_payload_is_rejected(
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    session = FakeSession(
        {_key("root", lang="vi"): [FakeResponse(200, payload)]}
    )
    with pytest.raises(ICDCrawlSchemaError):
        _crawler(session).crawl(tmp_path / "state.jsonl")


def test_duplicate_and_empty_non_leaf_are_rejected(tmp_path: Path) -> None:
    duplicate = _item(
        "chapter",
        "I",
        "I",
        "Chương một",
        is_leaf=False,
    )
    duplicate_session = FakeSession(
        {_key("root", lang="vi"): [_success(duplicate, duplicate)]}
    )
    with pytest.raises(ICDCrawlDataError, match="duplicate"):
        _crawler(duplicate_session).crawl(tmp_path / "duplicate.jsonl")

    empty_session = FakeSession(
        {
            _key("root", lang="vi"): [_success(duplicate)],
            _key("childs/chapter", id="I", lang="vi"): [_success()],
        }
    )
    with pytest.raises(ICDCrawlDataError, match="no children"):
        _crawler(empty_session).crawl(tmp_path / "empty.jsonl")


def test_force_keeps_existing_outputs_when_crawl_fails(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "icd.jsonl"
    csv_path = tmp_path / "icd.csv"
    manifest_path = tmp_path / "icd.manifest.json"
    for path in (jsonl_path, csv_path, manifest_path):
        path.write_text("existing\n", encoding="utf-8")
    session = FakeSession(
        {
            _key("root", lang="vi"): [
                _success(
                    _item(
                        "chapter",
                        "I",
                        "I",
                        "Chương một",
                        is_leaf=False,
                    )
                )
            ],
            _key("childs/chapter", id="I", lang="vi"): [
                FakeResponse(503, {"status": "error"})
            ],
        }
    )

    with pytest.raises(ICDCrawlHTTPError):
        crawl_to_files(
            _crawler(session),
            source_page_url="https://example.test/icd",
            jsonl_path=jsonl_path,
            csv_path=csv_path,
            manifest_path=manifest_path,
            force=True,
        )

    assert all(
        path.read_text("utf-8") == "existing\n"
        for path in (jsonl_path, csv_path, manifest_path)
    )


def test_catalog_loader_merges_names_and_ignores_hierarchy(tmp_path: Path) -> None:
    rows = [
        {
            "model": "section",
            "code": "A00",
            "name_vi": "Tên nhóm không phải tên mã",
        },
        {"model": "type", "code": "A00", "name_vi": "Bệnh tả"},
        {"model": "disease", "code": "A00.0", "name_vi": "Bệnh tả cổ điển"},
        {
            "model": "disease",
            "code": "A00.0",
            "name_vi": "Tả do Vibrio cholerae",
        },
    ]
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    index = ICDIndex.from_catalog(catalog)

    assert index.concepts["A00"].names == ("Bệnh tả",)
    assert index.concepts["A00.0"].names == (
        "Bệnh tả cổ điển",
        "Tả do Vibrio cholerae",
    )


def test_paths_prefer_catalog_when_present(tmp_path: Path) -> None:
    workbook = tmp_path / "source.xlsx"
    catalog = tmp_path / "catalog.jsonl"
    paths = PathsConfig(icd_source=workbook, icd_catalog=catalog)

    assert paths.preferred_icd_source() == workbook
    catalog.write_text("", encoding="utf-8")
    assert paths.preferred_icd_source() == catalog


def test_crawl_cli_writes_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    csv_path = tmp_path / "catalog.csv"
    manifest = tmp_path / "manifest.json"
    index_path = tmp_path / "index.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  icd_catalog: {catalog}",
                f"  icd_catalog_csv: {csv_path}",
                f"  icd_catalog_manifest: {manifest}",
                f"  icd_index: {index_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class StubCrawler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def close(self) -> None:
            pass

    crawl_arguments: dict[str, Any] = {}

    def fake_crawl_to_files(
        crawler: StubCrawler,
        **kwargs: Any,
    ) -> ICDCrawlOutput:
        del crawler
        crawl_arguments.update(kwargs)
        catalog.write_text(
            json.dumps(
                {
                    "model": "disease",
                    "code": "A00.0",
                    "name_vi": "Bệnh tả cổ điển",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        csv_path.write_text("code,name_vi\nA00.0,Bệnh tả cổ điển\n", "utf-8")
        manifest.write_text("{}\n", "utf-8")
        return ICDCrawlOutput(
            jsonl_path=catalog,
            csv_path=csv_path,
            manifest_path=manifest,
            record_count=1,
            request_count=1,
            jsonl_sha256="abc",
            csv_sha256="def",
        )

    monkeypatch.setattr(cli, "ICDCatalogCrawler", StubCrawler)
    monkeypatch.setattr(cli, "crawl_to_files", fake_crawl_to_files)

    cli.main(["--config", str(config_path), "crawl-icd", "--resume"])

    saved_index = ICDIndex.load(index_path)
    assert saved_index.contains("A00.0")
    assert crawl_arguments["resume"] is True
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["status"] == "ok"
    assert result["concepts"] == 1
