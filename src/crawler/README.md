# Crawler

Downloads RxNorm's drug vocabulary into `data/kb/rxnorm/` for offline use by
the linking module. See
[`docs/superpowers/specs/2026-07-26-crawler-module-design.md`](../../docs/superpowers/specs/2026-07-26-crawler-module-design.md)
for the design rationale.

## Run it

From the repo root:

```bash
uv sync
uv run python -m src.crawler run --source rxnorm-in
uv run python -m src.crawler run --source rxnorm-scd
uv run python -m src.crawler run --source rxnorm-sbd
```

Or via the Makefile at the repo root:

```bash
make crawl-rxnorm
```

Each run hits the public RxNav API (no auth needed, just internet access)
and writes one file:

| Source | Output | ~Records |
|---|---|---|
| `rxnorm-in` | `data/kb/rxnorm/in.json` | 14,600 |
| `rxnorm-scd` | `data/kb/rxnorm/scd.json` | 17,500 |
| `rxnorm-sbd` | `data/kb/rxnorm/sbd.json` | 9,700 |

`data/kb/` is gitignored — every fresh clone needs to run this once to get
its own local copy; nothing here runs at inference time.

## Add a new source

Add an entry to `SOURCES` in [`registry.py`](registry.py). If it's a
single-response JSON API, reuse `HttpJsonFetcher` as-is (see the three
`rxnorm-*` entries for the pattern). Paginated APIs aren't supported yet —
`HttpJsonFetcher` raises `NotImplementedError` if you pass a
`next_page_field`.
