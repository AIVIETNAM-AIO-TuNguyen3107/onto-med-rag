from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.crawler.adapters.http_json import HttpJsonFetcher
from src.crawler.adapters.writers import JsonFileWriter
from src.crawler.core import Fetcher, RecordWriter

RXNORM_ALLCONCEPTS_URL = "https://rxnav.nlm.nih.gov/REST/allconcepts.json"


@dataclass
class SourceConfig:
    fetcher_factory: Callable[[], Fetcher]
    writer: RecordWriter
    output_path: Path


def _rxnorm_fetcher(tty: str, source_name: str) -> HttpJsonFetcher:
    return HttpJsonFetcher(
        base_url=RXNORM_ALLCONCEPTS_URL,
        params={"tty": tty},
        records_path="minConceptGroup.minConcept",
        id_field="rxcui",
        next_page_field=None,
        source_name=source_name,
    )


SOURCES: dict[str, SourceConfig] = {
    "rxnorm-in": SourceConfig(
        fetcher_factory=lambda: _rxnorm_fetcher("IN", "rxnorm-in"),
        writer=JsonFileWriter(),
        output_path=Path("data/kb/rxnorm/in.json"),
    ),
    "rxnorm-scd": SourceConfig(
        fetcher_factory=lambda: _rxnorm_fetcher("SCD", "rxnorm-scd"),
        writer=JsonFileWriter(),
        output_path=Path("data/kb/rxnorm/scd.json"),
    ),
    "rxnorm-sbd": SourceConfig(
        fetcher_factory=lambda: _rxnorm_fetcher("SBD", "rxnorm-sbd"),
        writer=JsonFileWriter(),
        output_path=Path("data/kb/rxnorm/sbd.json"),
    ),
}
