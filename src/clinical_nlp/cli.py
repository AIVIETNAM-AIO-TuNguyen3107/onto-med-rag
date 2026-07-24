from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_nlp.config import load_config
from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.llm import create_llm_backend
from clinical_nlp.ner import create_ner_backend
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.rxnorm_linking import RxNormIndex
from clinical_nlp.supervision import RunSupervisor
from clinical_nlp.validation import validate_output_directory


def _build_pipeline(config_path: str) -> tuple[ClinicalPipeline, object]:
    config = load_config(config_path)
    if not config.paths.icd_index.exists():
        index = ICDIndex.from_workbook(config.paths.icd_source)
        index.save(config.paths.icd_index)
    else:
        index = ICDIndex.load(config.paths.icd_index)
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
    run_document = subparsers.add_parser("run-document")
    run_document.add_argument("--document", required=True)
    run_document.add_argument("--run-id")
    run_all = subparsers.add_parser("infer")
    run_all.add_argument("--run-id")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-id", required=True)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "build-icd":
        index = ICDIndex.from_workbook(config.paths.icd_source)
        index.save(config.paths.icd_index)
        print(
            json.dumps(
                {
                    "index": str(config.paths.icd_index),
                    "concepts": len(index.concepts),
                    "source_sha256": index.source_sha256,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "validate":
        output_dir = config.paths.runs_dir / args.run_id / "outputs"
        validate_output_directory(output_dir, config.paths.input_dir)
        print(json.dumps({"status": "ok", "output_dir": str(output_dir)}))
        return

    pipeline, config = _build_pipeline(args.config)
    if args.command == "run-document":
        supervisor = RunSupervisor(config, pipeline, run_id=args.run_id)
        supervisor.preflight()
        result = supervisor.run_document(args.document)
        print(json.dumps({"run_id": supervisor.run_id, **result}, ensure_ascii=False))
    elif args.command == "infer":
        supervisor = RunSupervisor(config, pipeline, run_id=args.run_id)
        supervisor.preflight()
        results = supervisor.run_all()
        validate_output_directory(
            supervisor.run_dir / "outputs",
            config.paths.input_dir,
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


if __name__ == "__main__":
    main()
