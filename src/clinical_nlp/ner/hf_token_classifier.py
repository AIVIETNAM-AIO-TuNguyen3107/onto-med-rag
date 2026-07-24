from __future__ import annotations

from clinical_nlp.schemas import Chunk, Document, EntityType, SpanProposal
from clinical_nlp.text import validate_chunk


DEFAULT_LABEL_MAP = {
    "ten_benh": EntityType.DIAGNOSIS,
    "trieu_chung_benh": EntityType.SYMPTOM,
    "bien_phap_dieu_tri": EntityType.MEDICATION,
    "bien_phap_chan_doan": EntityType.TEST_NAME,
}


class HFTokenClassifierBackend:
    name = "hf_token_classifier"

    def __init__(self, model_id: str, local_files_only: bool = False) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "HF token-classifier backend requires 'transformers'"
            ) from exc
        self.model_id = model_id
        self.pipe = pipeline(
            "token-classification",
            model=model_id,
            aggregation_strategy="simple",
            local_files_only=local_files_only,
        )

    def predict(
        self,
        document: Document,
        chunks: list[Chunk],
        threshold: float,
    ) -> list[SpanProposal]:
        proposals: list[SpanProposal] = []
        seen: set[tuple[int, int, str]] = set()
        for chunk in chunks:
            validate_chunk(document, chunk)
            for row in self.pipe(chunk.text):
                label = str(row.get("entity_group", row.get("entity", "")))
                label = label.removeprefix("B-").removeprefix("I-")
                entity_type = DEFAULT_LABEL_MAP.get(label)
                if entity_type is None or float(row.get("score", 0)) < threshold:
                    continue
                start = chunk.start + int(row["start"])
                end = chunk.start + int(row["end"])
                text = document.text[start:end]
                key = (start, end, entity_type.value)
                if key in seen or not text:
                    continue
                seen.add(key)
                proposals.append(
                    SpanProposal(
                        start=start,
                        end=end,
                        text=text,
                        type=entity_type,
                        source="hf_token_classifier",
                        score=float(row["score"]),
                        evidence={"model_id": self.model_id, "label": label},
                    )
                )
        return proposals

