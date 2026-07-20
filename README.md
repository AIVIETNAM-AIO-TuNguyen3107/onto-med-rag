# Medical Knowledge Retrieval

**Competition:** Bài 2 — Ontological Reasoning in Medical Knowledge Retrieval

Build a pipeline that extracts medical concepts from Vietnamese clinical free-text and maps:
- Diagnoses → **ICD-10**
- Drugs → **RxNorm**

With assertion detection (negation, family history, medical history).

## Phases

| Phase | Dates | Format |
|-------|-------|--------|
| Phase 1 (current) | 02/07 → 30/07/2026 | ZIP (GPU) |
| Phase 2 | 17/08 → 19/08/2026 | API endpoint |
| Phase 3 | 09/09 → 10/09/2026 | API endpoint |

## Sprint plan (Phase 1)

| Week | Goal |
|------|------|
| 1 | Knowledge gap — research docs, KBs, shared understanding |
| 2 | Baselines + PoC — literature review, select models, benchmark on test set |
| 3 | Fix & submit — iterate on weak spots, deliver best ZIP by 30/07 |

## Docs

- [Problem definition](docs/problem_definition.md) — input/output spec, entity types
- [Architecture](docs/architecture.md) — pipeline and module layout
- [Timeline](docs/timeline.md) — competition dates + sprint plan
- [Baseline decisions](docs/baseline_decisions.md) — model choices + Week 2 benchmark

## Repository layout

```
├── pyproject.toml         # Project config + dependencies (uv)
├── docs/                  # Problem spec, architecture, timeline, meetings
├── research/              # Team research notes (ICD-10, RxNorm, NLP, ontology)
├── data/
│   ├── kb/                # ICD-10, RxNorm knowledge bases
│   ├── raw/               # Self-created training data
│   ├── processed/         # Annotated / tokenized splits
│   ├── test/
│   │   ├── input/         # 1.txt … 100.txt (from Google Drive)
│   │   └── output/        # 1.json … 100.json (predictions)
│   └── examples/          # Sample I/O for development
├── src/
│   ├── ner/               # Span detection
│   ├── classification/    # Entity type (5 labels)
│   ├── assertion/         # isNegated, isFamily, isHistorical
│   ├── linking/
│   │   ├── icd10/         # Diagnosis → ICD-10
│   │   └── rxnorm/        # Drug → RxNorm
│   ├── ranking/           # Candidate ranking
│   ├── pipeline/          # Orchestration (ClinicalNLPipeline)
│   └── schemas/           # Output JSON types
├── notebooks/             # example_run.ipynb — dev & submission runs
└── experiments/           # Run configs and logs
```

## Quick start (Phase 1)

### 0. Install dependencies (uv)

```bash
uv sync
```

### 1. Download test input (gdown)

```bash
uv run gdown --folder "https://drive.google.com/drive/folders/1GEARAJjBU3726Et4kZnPjvKGN1O7ghO3" \
  -O data/test --remaining-ok
```

Expected layout after download:

```
data/test/input/
├── 1.txt
├── 2.txt
...
└── 100.txt
```

Drive folder: [competition test input](https://drive.google.com/drive/folders/1GEARAJjBU3726Et4kZnPjvKGN1O7ghO3?usp=drive_link)

### 2. Run PoC pipeline

Open [`notebooks/example_run.ipynb`](notebooks/example_run.ipynb) and run all cells.

Or from a Python session:

```python
from pathlib import Path
from src.pipeline.pipeline import build_default_pipeline, write_predictions

ROOT = Path(".")
pipeline = build_default_pipeline(ROOT / "data" / "kb")
text = (ROOT / "data/examples/sample_input.txt").read_text(encoding="utf-8")
entities = pipeline.process(text)

# batch (after gdown)
write_predictions(pipeline, ROOT / "data/test/input", ROOT / "data/test/output")
```

Run tests:

```bash
uv run pytest -v
```

### 3. Remaining steps

1. Download ICD-10 + RxNorm → `data/kb/` (replace sample TSVs)
2. Iterate on `src/extract/rules.py` or swap for fine-tuned model (Week 3)
3. Package submission ZIP

See `data/examples/sample_output.json` for the expected JSON format.
