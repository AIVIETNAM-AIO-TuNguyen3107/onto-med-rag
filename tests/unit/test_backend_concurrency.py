from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from clinical_nlp.ner.gliner import GLiNERBackend
from clinical_nlp.schemas import Chunk, Document


class TrackingModel:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def predict_entities(self, text, *, labels, threshold):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return []
        finally:
            with self.lock:
                self.active -= 1


def test_gliner_prediction_calls_are_serialized() -> None:
    backend = GLiNERBackend.__new__(GLiNERBackend)
    backend.model_id = "fake"
    backend.model = TrackingModel()
    backend._prediction_lock = threading.Lock()

    def predict(document_id: str) -> None:
        document = Document(id=document_id, text="ho")
        chunk = Chunk(
            document_id=document_id,
            index=0,
            start=0,
            end=2,
            text="ho",
        )
        backend.predict(document, [chunk], threshold=0.5)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(predict, ["1", "2", "3", "4"]))

    assert backend.model.max_active == 1
