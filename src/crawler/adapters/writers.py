from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.crawler.core import CrawlRecord


class JsonFileWriter:
    def write_all(self, records: Iterable[CrawlRecord], output_path: Path) -> int:
        payload = [record.data for record in records]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return len(payload)


class TxtFileWriter:
    def write_all(self, records: Iterable[CrawlRecord], output_path: Path) -> int:
        lines = [str(record.data) for record in records]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return len(lines)
