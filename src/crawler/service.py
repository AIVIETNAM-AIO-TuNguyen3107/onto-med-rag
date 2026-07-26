from __future__ import annotations

from pathlib import Path

from src.crawler.core import Fetcher, RecordWriter


def run_crawl(fetcher: Fetcher, writer: RecordWriter, output_path: Path) -> int:
    """Fetch all records and write them; returns the count written."""
    records = list(fetcher.fetch())
    return writer.write_all(records, output_path)
