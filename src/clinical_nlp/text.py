from __future__ import annotations

import re

from .schemas import Chunk, Document


BOUNDARY_RE = re.compile(r"(?:\n\s*\n|\n|(?<=[.!?])\s+)")


def chunk_document(
    document: Document,
    max_chars: int = 1800,
    overlap_chars: int = 160,
) -> list[Chunk]:
    if not document.text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars")

    text = document.text
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        target_end = min(start + max_chars, len(text))
        end = target_end
        if target_end < len(text):
            candidates = [
                match.end()
                for match in BOUNDARY_RE.finditer(text, start, target_end)
                if match.end() > start + max_chars // 2
            ]
            if candidates:
                end = candidates[-1]
        if end <= start:
            end = target_end
        chunks.append(
            Chunk(
                document_id=document.id,
                index=index,
                start=start,
                end=end,
                text=text[start:end],
            )
        )
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
        index += 1
    return chunks


def validate_chunk(document: Document, chunk: Chunk) -> None:
    if chunk.document_id != document.id:
        raise ValueError("chunk belongs to another document")
    if document.text[chunk.start : chunk.end] != chunk.text:
        raise ValueError("chunk is not an exact original-text view")


def find_occurrence(text: str, substring: str, occurrence: int = 1) -> tuple[int, int]:
    if occurrence < 1:
        raise ValueError("occurrence is one-based")
    start = -1
    cursor = 0
    for _ in range(occurrence):
        start = text.find(substring, cursor)
        if start < 0:
            raise ValueError("substring occurrence not found")
        cursor = start + len(substring)
    return start, start + len(substring)

