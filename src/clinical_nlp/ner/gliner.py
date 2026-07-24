from __future__ import annotations

from clinical_nlp.ner.base import LABEL_MAP, LABELS
from clinical_nlp.schemas import Chunk, Document, EntityType, SpanProposal
from clinical_nlp.text import validate_chunk


class GLiNERBackend:
    name = "gliner"

    def __init__(self, model_id: str, local_files_only: bool = False) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError("GLiNER backend requires the 'gliner' package") from exc
        self.model_id = model_id
        self.model = GLiNER.from_pretrained(
            model_id,
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
            rows = self.model.predict_entities(
                chunk.text,
                labels=LABELS,
                threshold=threshold,
            )
            for row in rows:
                mapped = LABEL_MAP.get(row["label"])
                if mapped is None:
                    continue
                start = chunk.start + int(row["start"])
                end = chunk.start + int(row["end"])
                entity_text = document.text[start:end]
                if entity_text != row["text"]:
                    continue
                key = (start, end, mapped)
                if key in seen:
                    continue
                seen.add(key)
                proposals.append(
                    SpanProposal(
                        start=start,
                        end=end,
                        text=entity_text,
                        type=EntityType(mapped),
                        source="gliner",
                        score=float(row.get("score", 0.5)),
                        evidence={"model_id": self.model_id},
                    )
                )
        return proposals

