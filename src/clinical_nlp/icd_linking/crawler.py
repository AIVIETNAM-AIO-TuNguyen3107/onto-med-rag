from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import requests


CATALOG_FIELDS = (
    "model",
    "id",
    "code",
    "name_vi",
    "parent_model",
    "parent_id",
    "parent_path",
    "path",
    "depth",
    "sibling_order",
    "is_leaf",
)
CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class ICDCrawlError(RuntimeError):
    """Base error for the ICD catalog crawler."""


class ICDCrawlHTTPError(ICDCrawlError):
    """Raised when the source API cannot be read successfully."""


class ICDCrawlSchemaError(ICDCrawlError):
    """Raised when the source API or checkpoint has an unexpected shape."""


class ICDCrawlDataError(ICDCrawlError):
    """Raised when the source hierarchy is internally inconsistent."""


@dataclass(frozen=True)
class ICDCatalogRow:
    model: str
    id: str
    code: str
    name_vi: str
    parent_model: str | None
    parent_id: str | None
    parent_path: str | None
    path: str
    depth: int
    sibling_order: int
    is_leaf: bool

    @property
    def node_key(self) -> tuple[str, str]:
        return self.model, self.id

    @property
    def key(self) -> str:
        return self.path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "ICDCatalogRow":
        if not isinstance(raw, dict):
            raise ICDCrawlSchemaError("catalog row must be an object")
        missing = [field for field in CATALOG_FIELDS if field not in raw]
        if missing:
            raise ICDCrawlSchemaError(
                f"catalog row is missing fields: {', '.join(missing)}"
            )
        values = {field: raw[field] for field in CATALOG_FIELDS}
        if not all(isinstance(values[field], str) for field in ("model", "id", "code", "name_vi")):
            raise ICDCrawlSchemaError("catalog model, id, code, and name_vi must be strings")
        for field in ("parent_model", "parent_id", "parent_path"):
            if values[field] is not None and not isinstance(values[field], str):
                raise ICDCrawlSchemaError(f"catalog {field} must be a string or null")
        if not isinstance(values["path"], str) or not values["path"]:
            raise ICDCrawlSchemaError("catalog path must be a non-empty string")
        if type(values["depth"]) is not int or values["depth"] < 0:
            raise ICDCrawlSchemaError("catalog depth must be a non-negative integer")
        if type(values["sibling_order"]) is not int or values["sibling_order"] < 0:
            raise ICDCrawlSchemaError(
                "catalog sibling_order must be a non-negative integer"
            )
        if type(values["is_leaf"]) is not bool:
            raise ICDCrawlSchemaError("catalog is_leaf must be a boolean")
        return cls(**values)


@dataclass(frozen=True)
class _ICDChildTemplate:
    model: str
    id: str
    code: str
    name_vi: str
    sibling_order: int
    is_leaf: bool

    @classmethod
    def from_row(cls, row: ICDCatalogRow) -> "_ICDChildTemplate":
        return cls(
            model=row.model,
            id=row.id,
            code=row.code,
            name_vi=row.name_vi,
            sibling_order=row.sibling_order,
            is_leaf=row.is_leaf,
        )

    def under(self, parent: ICDCatalogRow) -> ICDCatalogRow:
        return ICDCatalogRow(
            model=self.model,
            id=self.id,
            code=self.code,
            name_vi=self.name_vi,
            parent_model=parent.model,
            parent_id=parent.id,
            parent_path=parent.path,
            path=_child_path(parent.path, self.model, self.id),
            depth=parent.depth + 1,
            sibling_order=self.sibling_order,
            is_leaf=self.is_leaf,
        )


@dataclass(frozen=True)
class ICDCrawlSnapshot:
    rows: tuple[ICDCatalogRow, ...]
    request_count: int


@dataclass(frozen=True)
class ICDCrawlOutput:
    jsonl_path: Path
    csv_path: Path
    manifest_path: Path
    record_count: int
    request_count: int
    jsonl_sha256: str
    csv_sha256: str


ProgressCallback = Callable[[dict[str, int]], None]


class ICDCatalogCrawler:
    def __init__(
        self,
        *,
        api_base_url: str,
        language: str = "vi",
        request_delay_seconds: float = 0.5,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 30.0,
        max_retries: int = 5,
        session: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if request_delay_seconds < 0:
            raise ValueError("request_delay_seconds must be non-negative")
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0:
            raise ValueError("HTTP timeouts must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.api_base_url = api_base_url.rstrip("/")
        self.language = language
        self.request_delay_seconds = request_delay_seconds
        self.timeout = (connect_timeout_seconds, read_timeout_seconds)
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._owns_session = session is None
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.last_request_at: float | None = None
        self.request_count = 0
        self.headers = {
            "Accept": "application/json",
            "User-Agent": (
                "ViettelAIrace-clinical-nlp/0.1 "
                "(polite ICD-10 TT06 terminology snapshot)"
            ),
        }

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def crawl(
        self,
        checkpoint_path: Path,
        *,
        resume: bool = False,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> ICDCrawlSnapshot:
        if resume and force:
            raise ValueError("--resume and --force cannot be used together")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if force:
            checkpoint_path.unlink(missing_ok=True)
        elif checkpoint_path.exists() and not resume:
            raise FileExistsError(
                f"checkpoint already exists; use --resume or --force: {checkpoint_path}"
            )
        elif resume and not checkpoint_path.exists():
            raise FileNotFoundError(
                f"cannot resume because checkpoint does not exist: {checkpoint_path}"
            )

        if checkpoint_path.exists():
            rows, seen, definitions, expanded, child_cache = self._load_checkpoint(
                checkpoint_path
            )
        else:
            self._initialize_checkpoint(checkpoint_path)
            rows = []
            seen: dict[str, ICDCatalogRow] = {}
            definitions: dict[tuple[str, str], ICDCatalogRow] = {}
            expanded: set[str] = set()
            child_cache: dict[
                tuple[str, str], tuple[_ICDChildTemplate, ...]
            ] = {}

        if not rows:
            payload = self._request_json(
                f"{self.api_base_url}/root",
                params={"lang": self.language},
            )
            root_rows = self._parse_api_rows(payload, parent=None)
            if not root_rows:
                raise ICDCrawlDataError("root endpoint returned no hierarchy nodes")
            self._add_new_rows(rows, seen, definitions, root_rows)
            self._append_event(
                checkpoint_path,
                {
                    "type": "root",
                    "request_count": self.request_count,
                    "rows": [row.to_dict() for row in root_rows],
                },
            )

        queue = deque(
            row for row in rows if not row.is_leaf and row.key not in expanded
        )
        expanded_count = len(expanded)
        while queue:
            parent = queue.popleft()
            cached_templates = child_cache.get(parent.node_key)
            if cached_templates is None:
                payload = self._request_json(
                    f"{self.api_base_url}/childs/{quote(parent.model, safe='')}",
                    params={"id": parent.id, "lang": self.language},
                )
                child_rows = self._parse_api_rows(payload, parent=parent)
                child_cache[parent.node_key] = tuple(
                    _ICDChildTemplate.from_row(row) for row in child_rows
                )
                from_cache = False
            else:
                child_rows = [
                    template.under(parent) for template in cached_templates
                ]
                from_cache = True
            if not child_rows:
                raise ICDCrawlDataError(
                    f"non-leaf node {parent.model}:{parent.id} returned no children"
                )
            self._add_new_rows(rows, seen, definitions, child_rows)
            self._append_event(
                checkpoint_path,
                {
                    "type": "children",
                    "parent_model": parent.model,
                    "parent_id": parent.id,
                    "parent_path": parent.path,
                    "from_cache": from_cache,
                    "request_count": self.request_count,
                    "rows": [row.to_dict() for row in child_rows],
                },
            )
            expanded.add(parent.key)
            expanded_count += 1
            queue.extend(row for row in child_rows if not row.is_leaf)
            if progress is not None:
                progress(
                    {
                        "expanded": expanded_count,
                        "queued": len(queue),
                        "records": len(rows),
                        "requests": self.request_count,
                    }
                )

        return ICDCrawlSnapshot(rows=tuple(rows), request_count=self.request_count)

    def _request_json(self, url: str, *, params: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._respect_request_interval()
            try:
                self.request_count += 1
                response = self.session.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                self.last_request_at = self.monotonic()
            except requests.RequestException as exc:
                self.last_request_at = self.monotonic()
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self.sleeper(2**attempt)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ICDCrawlSchemaError(
                        f"API returned invalid JSON for {url}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ICDCrawlSchemaError(
                        f"API response must be an object for {url}"
                    )
                return payload

            error = ICDCrawlHTTPError(
                f"API request failed with HTTP {response.status_code}: {url}"
            )
            last_error = error
            if (
                response.status_code not in TRANSIENT_STATUS_CODES
                or attempt >= self.max_retries
            ):
                break
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            self.sleeper(retry_after if retry_after is not None else 2**attempt)

        if isinstance(last_error, ICDCrawlError):
            raise last_error
        raise ICDCrawlHTTPError(
            f"API request failed after {self.max_retries + 1} attempts: {url}"
        ) from last_error

    def _respect_request_interval(self) -> None:
        if self.last_request_at is None or self.request_delay_seconds == 0:
            return
        elapsed = self.monotonic() - self.last_request_at
        remaining = self.request_delay_seconds - elapsed
        if remaining > 0:
            self.sleeper(remaining)

    def _parse_api_rows(
        self,
        payload: dict[str, Any],
        *,
        parent: ICDCatalogRow | None,
    ) -> list[ICDCatalogRow]:
        if payload.get("status") != "success":
            raise ICDCrawlSchemaError(
                f"API response status is not success: {payload.get('status')!r}"
            )
        items = payload.get("data")
        if not isinstance(items, list):
            raise ICDCrawlSchemaError("API response data must be a list")
        rows: list[ICDCatalogRow] = []
        for sibling_order, item in enumerate(items):
            if not isinstance(item, dict):
                raise ICDCrawlSchemaError("API hierarchy item must be an object")
            data = item.get("data")
            if not isinstance(data, dict):
                raise ICDCrawlSchemaError("API hierarchy item data must be an object")
            model = item.get("model")
            node_id = item.get("id")
            code = data.get("code")
            name = data.get("name")
            is_leaf = item.get("is_leaf")
            if not all(isinstance(value, str) and value for value in (model, node_id, code, name)):
                raise ICDCrawlSchemaError(
                    "API hierarchy item requires non-empty model, id, code, and name"
                )
            if "id" in data and str(data["id"]) != node_id:
                raise ICDCrawlSchemaError(
                    f"API hierarchy item id mismatch for {model}:{node_id}"
                )
            if type(is_leaf) is not bool:
                raise ICDCrawlSchemaError(
                    f"API hierarchy item is_leaf must be boolean for {model}:{node_id}"
                )
            rows.append(
                ICDCatalogRow(
                    model=model,
                    id=node_id,
                    code=code,
                    name_vi=name,
                    parent_model=parent.model if parent else None,
                    parent_id=parent.id if parent else None,
                    parent_path=parent.path if parent else None,
                    path=(
                        _child_path(parent.path, model, node_id)
                        if parent
                        else _root_path(model, node_id)
                    ),
                    depth=parent.depth + 1 if parent else 0,
                    sibling_order=sibling_order,
                    is_leaf=is_leaf,
                )
            )
        return rows

    @staticmethod
    def _add_new_rows(
        rows: list[ICDCatalogRow],
        seen: dict[str, ICDCatalogRow],
        definitions: dict[tuple[str, str], ICDCatalogRow],
        new_rows: Iterable[ICDCatalogRow],
    ) -> None:
        for row in new_rows:
            existing = seen.get(row.key)
            if existing is not None:
                if (
                    existing.model == row.model
                    and existing.id == row.id
                    and existing.code == row.code
                    and existing.name_vi == row.name_vi
                    and existing.parent_model == row.parent_model
                    and existing.parent_id == row.parent_id
                    and existing.parent_path == row.parent_path
                    and existing.depth == row.depth
                    and existing.is_leaf == row.is_leaf
                ):
                    raise ICDCrawlDataError(
                        f"duplicate hierarchy node: {row.model}:{row.id}"
                    )
                raise ICDCrawlDataError(
                    f"conflicting hierarchy occurrence: {row.path}"
                )
            definition = definitions.get(row.node_key)
            if definition is not None and (
                definition.code != row.code
                or definition.name_vi != row.name_vi
                or definition.is_leaf != row.is_leaf
            ):
                raise ICDCrawlDataError(
                    f"conflicting hierarchy node: {row.model}:{row.id}"
                )
            seen[row.key] = row
            definitions.setdefault(row.node_key, row)
            rows.append(row)

    def _initialize_checkpoint(self, path: Path) -> None:
        event = {
            "type": "metadata",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "api_base_url": self.api_base_url,
            "language": self.language,
        }
        _atomic_write_text(
            path,
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n",
        )

    def _load_checkpoint(
        self,
        path: Path,
    ) -> tuple[
        list[ICDCatalogRow],
        dict[str, ICDCatalogRow],
        dict[tuple[str, str], ICDCatalogRow],
        set[str],
        dict[tuple[str, str], tuple[_ICDChildTemplate, ...]],
    ]:
        events = _read_checkpoint_events(path)
        if not events or events[0].get("type") != "metadata":
            raise ICDCrawlSchemaError("checkpoint metadata event is missing")
        metadata = events[0]
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ICDCrawlSchemaError("unsupported checkpoint schema version")
        if metadata.get("api_base_url") != self.api_base_url:
            raise ICDCrawlSchemaError("checkpoint API base URL does not match")
        if metadata.get("language") != self.language:
            raise ICDCrawlSchemaError("checkpoint language does not match")

        rows: list[ICDCatalogRow] = []
        seen: dict[str, ICDCatalogRow] = {}
        definitions: dict[tuple[str, str], ICDCatalogRow] = {}
        expanded: set[str] = set()
        child_cache: dict[
            tuple[str, str], tuple[_ICDChildTemplate, ...]
        ] = {}
        root_seen = False
        for event in events[1:]:
            event_type = event.get("type")
            raw_rows = event.get("rows")
            if not isinstance(raw_rows, list):
                raise ICDCrawlSchemaError("checkpoint event rows must be a list")
            if event_type == "root":
                if root_seen:
                    raise ICDCrawlSchemaError("checkpoint contains multiple root events")
                root_seen = True
                parent = None
            elif event_type == "children":
                parent_model = event.get("parent_model")
                parent_id = event.get("parent_id")
                if not isinstance(parent_model, str) or not isinstance(parent_id, str):
                    raise ICDCrawlSchemaError(
                        "checkpoint children event has an invalid parent"
                    )
                parent_path = event.get("parent_path")
                if parent_path is not None and not isinstance(parent_path, str):
                    raise ICDCrawlSchemaError(
                        "checkpoint parent_path must be a string or null"
                    )
                if parent_path is not None:
                    parent = seen.get(parent_path)
                else:
                    candidates = [
                        row
                        for row in rows
                        if row.node_key == (parent_model, parent_id)
                    ]
                    parent = candidates[0] if len(candidates) == 1 else None
                if parent is None or parent.is_leaf:
                    raise ICDCrawlSchemaError(
                        f"checkpoint parent is missing or leaf: {parent_model}:{parent_id}"
                    )
                if parent.node_key != (parent_model, parent_id):
                    raise ICDCrawlSchemaError(
                        f"checkpoint parent identity mismatch: {parent_model}:{parent_id}"
                    )
                if parent.path in expanded:
                    raise ICDCrawlSchemaError(
                        f"checkpoint expands a parent twice: {parent.path}"
                    )
            else:
                raise ICDCrawlSchemaError(
                    f"unsupported checkpoint event type: {event_type!r}"
                )
            parsed_rows = [
                self._checkpoint_row(raw_row, parent=parent)
                for raw_row in raw_rows
            ]
            self._add_new_rows(rows, seen, definitions, parsed_rows)
            if parent is not None:
                templates = tuple(
                    _ICDChildTemplate.from_row(row) for row in parsed_rows
                )
                cached = child_cache.get(parent.node_key)
                if cached is not None and cached != templates:
                    raise ICDCrawlSchemaError(
                        f"checkpoint has conflicting child lists for "
                        f"{parent.model}:{parent.id}"
                    )
                child_cache.setdefault(parent.node_key, templates)
                expanded.add(parent.path)
            request_count = event.get("request_count", 0)
            if type(request_count) is not int or request_count < 0:
                raise ICDCrawlSchemaError(
                    "checkpoint request_count must be a non-negative integer"
                )
            self.request_count = max(self.request_count, request_count)
        return rows, seen, definitions, expanded, child_cache

    @staticmethod
    def _checkpoint_row(
        raw: Any,
        *,
        parent: ICDCatalogRow | None,
    ) -> ICDCatalogRow:
        if not isinstance(raw, dict):
            raise ICDCrawlSchemaError("checkpoint catalog row must be an object")
        migrated = dict(raw)
        model = migrated.get("model")
        node_id = migrated.get("id")
        if not isinstance(model, str) or not isinstance(node_id, str):
            raise ICDCrawlSchemaError(
                "checkpoint catalog row has an invalid model or id"
            )
        expected_parent_model = parent.model if parent else None
        expected_parent_id = parent.id if parent else None
        if migrated.get("parent_model") != expected_parent_model:
            raise ICDCrawlSchemaError(
                f"checkpoint parent model mismatch for {model}:{node_id}"
            )
        if migrated.get("parent_id") != expected_parent_id:
            raise ICDCrawlSchemaError(
                f"checkpoint parent id mismatch for {model}:{node_id}"
            )
        expected_parent_path = parent.path if parent else None
        expected_path = (
            _child_path(parent.path, model, node_id)
            if parent
            else _root_path(model, node_id)
        )
        migrated.setdefault("parent_path", expected_parent_path)
        migrated.setdefault("path", expected_path)
        if migrated["parent_path"] != expected_parent_path:
            raise ICDCrawlSchemaError(
                f"checkpoint parent path mismatch for {model}:{node_id}"
            )
        if migrated["path"] != expected_path:
            raise ICDCrawlSchemaError(
                f"checkpoint path mismatch for {model}:{node_id}"
            )
        return ICDCatalogRow.from_dict(migrated)

    @staticmethod
    def _append_event(path: Path, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def crawl_to_files(
    crawler: ICDCatalogCrawler,
    *,
    source_page_url: str,
    jsonl_path: Path,
    csv_path: Path,
    manifest_path: Path,
    resume: bool = False,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> ICDCrawlOutput:
    final_paths = (jsonl_path, csv_path, manifest_path)
    existing = [path for path in final_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "crawl output already exists; use --force to replace it: "
            + ", ".join(str(path) for path in existing)
        )
    checkpoint_path = jsonl_path.with_suffix(
        jsonl_path.suffix + ".checkpoint.jsonl"
    )
    snapshot = crawler.crawl(
        checkpoint_path,
        resume=resume,
        force=force,
        progress=progress,
    )
    output = write_catalog_exports(
        snapshot,
        source_page_url=source_page_url,
        api_base_url=crawler.api_base_url,
        language=crawler.language,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
    )
    checkpoint_path.unlink(missing_ok=True)
    return output


def write_catalog_exports(
    snapshot: ICDCrawlSnapshot,
    *,
    source_page_url: str,
    api_base_url: str,
    language: str,
    jsonl_path: Path,
    csv_path: Path,
    manifest_path: Path,
) -> ICDCrawlOutput:
    if not snapshot.rows:
        raise ICDCrawlDataError("cannot export an empty ICD catalog")
    for path in (jsonl_path, csv_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary_paths: list[Path] = []
    try:
        jsonl_temp = _temporary_path(jsonl_path)
        temporary_paths.append(jsonl_temp)
        with jsonl_temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in snapshot.rows:
                handle.write(
                    json.dumps(
                        row.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        jsonl_sha256 = _sha256(jsonl_temp)

        csv_temp = _temporary_path(csv_path)
        temporary_paths.append(csv_temp)
        with csv_temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
            writer.writeheader()
            writer.writerows(row.to_dict() for row in snapshot.rows)
        csv_sha256 = _sha256(csv_temp)

        counts_by_model = dict(
            sorted(Counter(row.model for row in snapshot.rows).items())
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset": "ICD-10 06/2026/TT-BYT",
            "language": language,
            "source_page_url": source_page_url,
            "api_base_url": api_base_url,
            "completed_at": datetime.now(UTC).isoformat(),
            "record_count": len(snapshot.rows),
            "unique_node_count": len({row.node_key for row in snapshot.rows}),
            "reused_node_occurrences": (
                len(snapshot.rows) - len({row.node_key for row in snapshot.rows})
            ),
            "counts_by_model": counts_by_model,
            "request_count": snapshot.request_count,
            "files": {
                "jsonl": {
                    "path": str(jsonl_path),
                    "sha256": jsonl_sha256,
                },
                "csv": {
                    "path": str(csv_path),
                    "sha256": csv_sha256,
                },
            },
        }
        manifest_temp = _temporary_path(manifest_path)
        temporary_paths.append(manifest_temp)
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        os.replace(jsonl_temp, jsonl_path)
        temporary_paths.remove(jsonl_temp)
        os.replace(csv_temp, csv_path)
        temporary_paths.remove(csv_temp)
        os.replace(manifest_temp, manifest_path)
        temporary_paths.remove(manifest_temp)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    return ICDCrawlOutput(
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
        record_count=len(snapshot.rows),
        request_count=snapshot.request_count,
        jsonl_sha256=jsonl_sha256,
        csv_sha256=csv_sha256,
    )


def _read_checkpoint_events(path: Path) -> list[dict[str, Any]]:
    raw_lines = path.read_text("utf-8").splitlines()
    events: list[dict[str, Any]] = []
    valid_lines: list[str] = []
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            if index != len(raw_lines) - 1:
                raise ICDCrawlSchemaError(
                    f"checkpoint is corrupt at line {index + 1}"
                ) from exc
            _atomic_write_text(
                path,
                ("\n".join(valid_lines) + "\n") if valid_lines else "",
            )
            break
        if not isinstance(event, dict):
            raise ICDCrawlSchemaError(
                f"checkpoint event at line {index + 1} must be an object"
            )
        events.append(event)
        valid_lines.append(line)
    return events


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _root_path(model: str, node_id: str) -> str:
    return f"{quote(model, safe='')}:{quote(node_id, safe='')}"


def _child_path(parent_path: str, model: str, node_id: str) -> str:
    return f"{parent_path}/{_root_path(model, node_id)}"


def _temporary_path(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
