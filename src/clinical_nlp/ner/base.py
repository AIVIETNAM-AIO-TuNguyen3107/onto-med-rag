from __future__ import annotations

from typing import Protocol

from clinical_nlp.config import ModelConfig
from clinical_nlp.schemas import Chunk, Document, SpanProposal


LABEL_MAP = {
    "patient symptom or clinical finding": "TRIỆU_CHỨNG",
    "disease or medical diagnosis": "CHẨN_ĐOÁN",
    "medication including dose route and frequency": "THUỐC",
    "laboratory test name": "TÊN_XÉT_NGHIỆM",
    "laboratory test result value and unit": "KẾT_QUẢ_XÉT_NGHIỆM",
}
LABELS = list(LABEL_MAP)


class NERBackend(Protocol):
    name: str

    def predict(
        self,
        document: Document,
        chunks: list[Chunk],
        threshold: float,
    ) -> list[SpanProposal]: ...


class NoopNERBackend:
    name = "noop"

    def predict(
        self,
        document: Document,
        chunks: list[Chunk],
        threshold: float,
    ) -> list[SpanProposal]:
        return []


def create_ner_backend(config: ModelConfig) -> NERBackend:
    if config.backend == "noop":
        return NoopNERBackend()
    if config.backend == "gliner":
        from .gliner import GLiNERBackend

        return GLiNERBackend(
            model_id=config.model_id,
            local_files_only=config.local_files_only,
        )
    if config.backend == "hf_token_classifier":
        from .hf_token_classifier import HFTokenClassifierBackend

        return HFTokenClassifierBackend(
            model_id=config.model_id,
            local_files_only=config.local_files_only,
        )
    raise ValueError(f"unsupported NER backend: {config.backend}")

