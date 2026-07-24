from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from clinical_nlp.config import load_config
from clinical_nlp.icd_linking import (
    ICDCatalogCrawler,
    ICDCrawlOutput,
    ICDIndex,
    crawl_to_files,
)
from clinical_nlp.llm import create_llm_backend
from clinical_nlp.ner import create_ner_backend
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.supervision import RunSupervisor
from clinical_nlp.validation import validate_output_directory


def _build_pipeline(config_path: str) -> tuple[ClinicalPipeline, object]:
    config = load_config(config_path)
    source = config.paths.preferred_icd_source()
    source_sha256 = _sha256(source)
    rebuild = not config.paths.icd_index.exists()
    if not rebuild:
        try:
            index = ICDIndex.load(config.paths.icd_index)
            rebuild = index.source_sha256 != source_sha256
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            rebuild = True
    if rebuild:
        index = ICDIndex.from_source(source)
        index.save(config.paths.icd_index)
    rxnorm = RxNormIndex(
        config.paths.rxnorm_cache,
        use_api=config.linking.use_rxnav_api,
    )
    try:
        ner = create_ner_backend(config.ner)
    except Exception:
        if config.run.fail_on_model_unavailable:
            raise
        config.ner.backend = "noop"
        ner = create_ner_backend(config.ner)
    try:
        llm = create_llm_backend(config.llm)
    except Exception:
        if config.run.fail_on_model_unavailable:
            raise
        config.llm.backend = "noop"
        llm = create_llm_backend(config.llm)
    return ClinicalPipeline(config, index, rxnorm, ner, llm), config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clinical-nlp")
    parser.add_argument("--config", default="configs/base.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-icd")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--run-id")
    preflight.add_argument("--documents", nargs="+")
    crawl_icd = subparsers.add_parser("crawl-icd")
    crawl_icd.add_argument("--resume", action="store_true")
    crawl_icd.add_argument("--force", action="store_true")
    run_document = subparsers.add_parser("run-document")
    run_document.add_argument("--document", required=True)
    run_document.add_argument("--run-id")
    run_all = subparsers.add_parser("infer")
    run_all.add_argument("--run-id")
    run_all.add_argument("--documents", nargs="+")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--documents", nargs="+")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "crawl-icd":
        if args.resume and args.force:
            parser.error("crawl-icd: --resume and --force cannot be used together")
        crawler = ICDCatalogCrawler(
            api_base_url=config.icd_crawl.api_base_url,
            language=config.icd_crawl.language,
            request_delay_seconds=config.icd_crawl.request_delay_seconds,
            connect_timeout_seconds=config.icd_crawl.connect_timeout_seconds,
            read_timeout_seconds=config.icd_crawl.read_timeout_seconds,
            max_retries=config.icd_crawl.max_retries,
        )
        last_reported_requests = 0

        def report_progress(progress: dict[str, int]) -> None:
            nonlocal last_reported_requests
            if progress["requests"] - last_reported_requests < 25:
                return
            last_reported_requests = progress["requests"]
            print(json.dumps({"crawl_progress": progress}), flush=True)

        try:
            output: ICDCrawlOutput = crawl_to_files(
                crawler,
                source_page_url=config.icd_crawl.source_page_url,
                jsonl_path=config.paths.icd_catalog,
                csv_path=config.paths.icd_catalog_csv,
                manifest_path=config.paths.icd_catalog_manifest,
                resume=args.resume,
                force=args.force,
                progress=report_progress,
            )
        finally:
            crawler.close()
        index = ICDIndex.from_catalog(output.jsonl_path)
        index.save(config.paths.icd_index)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "catalog_jsonl": str(output.jsonl_path),
                    "catalog_csv": str(output.csv_path),
                    "manifest": str(output.manifest_path),
                    "records": output.record_count,
                    "requests": output.request_count,
                    "index": str(config.paths.icd_index),
                    "concepts": len(index.concepts),
                    "source_sha256": index.source_sha256,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "build-icd":
        source = config.paths.preferred_icd_source()
        index = ICDIndex.from_source(source)
        index.save(config.paths.icd_index)
        print(
            json.dumps(
                {
                    "index": str(config.paths.icd_index),
                    "source": str(source),
                    "concepts": len(index.concepts),
                    "source_sha256": index.source_sha256,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "validate":
        output_dir = config.paths.runs_dir / args.run_id / "outputs"
        expected = set(args.documents) if args.documents else _manifest_ids(
            config.paths.runs_dir / args.run_id / "source_manifest.json"
        )
        validate_output_directory(
            output_dir,
            config.paths.input_dir,
            expected_stems=expected,
        )
        print(json.dumps({"status": "ok", "output_dir": str(output_dir)}))
        return

    pipeline, config = _build_pipeline(args.config)
    if args.command == "preflight":
        supervisor = RunSupervisor(config, pipeline, run_id=args.run_id)
        manifest = supervisor.preflight(args.documents)
        online = pipeline.online_preflight()
        supervisor.record_online_preflight(online)
        print(
            json.dumps(
                {
                    "run_id": supervisor.run_id,
                    "documents": manifest["selection"]["document_count"],
                    **online,
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "run-document":
        supervisor = RunSupervisor(config, pipeline, run_id=args.run_id)
        supervisor.preflight([args.document])
        result = supervisor.run_document(args.document)
        print(json.dumps({"run_id": supervisor.run_id, **result}, ensure_ascii=False))
    elif args.command == "infer":
        supervisor = RunSupervisor(config, pipeline, run_id=args.run_id)
        supervisor.preflight(args.documents)
        online = pipeline.online_preflight()
        supervisor.record_online_preflight(online)
        results = supervisor.run_all(args.documents)
        expected = (
            set(args.documents)
            if args.documents
            else {path.stem for path in config.paths.input_dir.glob("*.txt")}
        )
        validate_output_directory(
            supervisor.run_dir / "outputs",
            config.paths.input_dir,
            expected_stems=expected,
        )
        print(
            json.dumps(
                {
                    "run_id": supervisor.run_id,
                    "documents": len(results),
                    "entities": sum(row["entity_count"] for row in results),
                },
                ensure_ascii=False,
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_ids(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text("utf-8"))
    values = raw.get("selection", {}).get("document_ids")
    if not isinstance(values, list) or not all(isinstance(row, str) for row in values):
        return None
    return set(values)


if __name__ == "__main__":
    main()
