# Direct-extraction maintenance scripts

- `verify_batches.py` is read-only. It resolves every raw `(text, occurrence)`
  pair, checks exact source slices, duplicates, overlaps, masked spans, and
  reports calibration totals.
- `canonicalize_batches.py` rewrites raw batch `text` values to the exact source
  slice at their already-resolved offsets. It never moves or guesses a span.

Run the verifier before and after canonicalization:

```bash
uv run python scripts/direct_extraction/verify_batches.py
uv run python scripts/direct_extraction/canonicalize_batches.py \
  artifacts/direct-extraction/raw/batch-01.json
uv run python scripts/direct_extraction/verify_batches.py
```

Raw batches are generated artifacts and remain outside Git. The Drive
reproducibility archive preserves the exact batch files used for V1/V2/V3.
