from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol


@dataclass
class CrawlRecord:
    id: str
    data: dict | str
    source: str


class CrawlError(Exception):
    """Raised by a Fetcher adapter on unrecoverable failure."""


class Fetcher(Protocol):
    def fetch(self) -> Iterator[CrawlRecord]:
        ...


class RecordWriter(Protocol):
    def write_all(self, records: Iterable[CrawlRecord], output_path: Path) -> int:
        ...
