from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    input_dir: Path = Path("input")
    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("artifacts")
    icd_source: Path = Path("DM ICD10-19_8_BYT.xlsx")
    icd_catalog: Path = Path("artifacts/icd10_tt06_vi.jsonl")
    icd_catalog_csv: Path = Path("artifacts/icd10_tt06_vi.csv")
    icd_catalog_manifest: Path = Path("artifacts/icd10_tt06_vi.manifest.json")
    icd_index: Path = Path("artifacts/icd_index.json")
    rxnorm_cache: Path = Path("artifacts/rxnorm_cache.json")

    def preferred_icd_source(self) -> Path:
        return self.icd_catalog if self.icd_catalog.exists() else self.icd_source


class ICDCrawlConfig(BaseModel):
    api_base_url: str = "https://ccs.whiteneuron.com/api/ICD10_TT06"
    source_page_url: str = "https://icd.kcb.vn/icd-10-tt06/icd10-tt06"
    language: str = "vi"
    request_delay_seconds: float = Field(default=0.5, ge=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    read_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=5, ge=0)


class EntityFindingConfig(BaseModel):
    chunk_chars: int = 1800
    chunk_overlap_chars: int = 160
    ner_threshold: float = 0.35


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    backend: str = "noop"
    model_id: str
    local_files_only: bool = False
    reasoning_effort: str = "high"
    thinking: bool = True
    max_new_tokens: int = 4096
    max_retries: int = 2
    endpoint: str | None = None
    api_key_env: str | None = None
    request_timeout_seconds: float = Field(default=600.0, gt=0)
    # None | "bnb_4bit" — post-training load quantization for local Transformers.
    quantization: str | None = None


class LinkingConfig(BaseModel):
    icd_max_candidates: int = 3
    rxnorm_max_candidates: int = 3
    retrieval_candidates: int = Field(default=20, ge=1, le=100)
    use_rxnav_api: bool = True


class RunConfig(BaseModel):
    fail_on_model_unavailable: bool = False
    pretty_json: bool = True
    llm_full_review: bool = False


class PipelineConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    icd_crawl: ICDCrawlConfig = Field(default_factory=ICDCrawlConfig)
    entity_finding: EntityFindingConfig = Field(default_factory=EntityFindingConfig)
    ner: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model_id="Ihor/gliner-biomed-large-v1.0"
        )
    )
    llm: ModelConfig = Field(
        default_factory=lambda: ModelConfig(model_id="Qwen/Qwen3.5-9B")
    )
    linking: LinkingConfig = Field(default_factory=LinkingConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text("utf-8")) or {}
    return PipelineConfig.model_validate(raw)
