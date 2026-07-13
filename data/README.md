# Data

## Test input (competition)

Download the 100-record test set from Google Drive:

**Folder:** [1GEARAJjBU3726Et4kZnPjvKGN1O7ghO3](https://drive.google.com/drive/folders/1GEARAJjBU3726Et4kZnPjvKGN1O7ghO3?usp=drive_link)

```bash
uv sync

uv run gdown --folder "https://drive.google.com/drive/folders/1GEARAJjBU3726Et4kZnPjvKGN1O7ghO3" \
  -O test --remaining-ok
```

Run from this directory (`data/`), or use `-O data/test` from the repo root.

Expected result:

```
test/input/1.txt … test/input/100.txt
```

Predictions go in `test/output/` (not committed).

## Knowledge bases

Place downloaded files in:

- `kb/icd10/` — ICD-10
- `kb/rxnorm/` — RxNorm

## Training data

Self-created data goes in `raw/` and `processed/` (gitignored).
