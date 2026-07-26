.PHONY: sync crawl-rxnorm crawl-rxnorm-in crawl-rxnorm-scd crawl-rxnorm-sbd

sync:
	uv sync

crawl-rxnorm-in:
	uv run python -m src.crawler run --source rxnorm-in

crawl-rxnorm-scd:
	uv run python -m src.crawler run --source rxnorm-scd

crawl-rxnorm-sbd:
	uv run python -m src.crawler run --source rxnorm-sbd

crawl-rxnorm: crawl-rxnorm-in crawl-rxnorm-scd crawl-rxnorm-sbd
