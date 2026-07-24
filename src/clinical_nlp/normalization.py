from __future__ import annotations

import re
import unicodedata


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[\-‐‑‒–—_/.,:;()[\]{}]+")


def normalize_search(value: str) -> str:
    """Normalize a search key, never a span-bearing source string."""
    value = unicodedata.normalize("NFC", value).casefold()
    value = PUNCT_RE.sub(" ", value)
    return SPACE_RE.sub(" ", value).strip()

