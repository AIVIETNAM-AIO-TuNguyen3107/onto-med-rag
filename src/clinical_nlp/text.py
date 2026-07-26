from __future__ import annotations

import re
import unicodedata

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
    if not substring:
        raise ValueError("substring must not be empty")
    normalized_text_parts: list[str] = []
    original_index_by_normalized_index: list[int] = []
    for original_index, character in enumerate(text):
        normalized_character = unicodedata.normalize("NFD", character)
        normalized_text_parts.append(normalized_character)
        original_index_by_normalized_index.extend(
            [original_index] * len(normalized_character)
        )
    normalized_text = "".join(normalized_text_parts)
    normalized_substring = unicodedata.normalize("NFD", substring)
    cursor = 0
    valid_occurrences = 0
    while True:
        start = normalized_text.find(normalized_substring, cursor)
        if start < 0:
            raise ValueError("substring occurrence not found")
        end = start + len(normalized_substring)
        original_start = original_index_by_normalized_index[start]
        original_end = original_index_by_normalized_index[end - 1] + 1
        if (
            unicodedata.normalize("NFD", text[original_start:original_end])
            == normalized_substring
        ):
            valid_occurrences += 1
            if valid_occurrences == occurrence:
                return original_start, original_end
            cursor = end
        else:
            cursor = start + 1
