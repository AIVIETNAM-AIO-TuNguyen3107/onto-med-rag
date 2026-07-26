from __future__ import annotations

import argparse
import sys

from src.crawler.registry import SOURCES
from src.crawler.service import run_crawl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.crawler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a crawl for one registered source"
    )
    run_parser.add_argument("--source", required=True, help="Registered source name")

    args = parser.parse_args(argv)

    if args.command == "run":
        config = SOURCES.get(args.source)
        if config is None:
            available = ", ".join(sorted(SOURCES))
            print(
                f"Unknown source {args.source!r}. Available: {available}",
                file=sys.stderr,
            )
            return 1

        fetcher = config.fetcher_factory()
        count = run_crawl(fetcher, config.writer, config.output_path)
        print(f"Wrote {count} records to {config.output_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
