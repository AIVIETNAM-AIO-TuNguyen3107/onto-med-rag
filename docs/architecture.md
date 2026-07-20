# System Architecture (Phase 1)

## Pipeline

```
Clinical Text (.txt)
        │
        ▼
┌───────────────────┐
│  Medical NER      │  Extract entity spans + positions
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Entity Classifier │  TRIỆU_CHỨNG | TÊN_XÉT_NGHIỆM | KẾT_QUẢ_XÉT_NGHIỆM
│                   │  CHẨN_ĐOÁN | THUỐC
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Assertion Detect  │  isNegated | isFamily | isHistorical
│ (CHẨN_ĐOÁN,       │  (only CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG)
│  THUỐC, TRIỆU_   │
│  CHỨNG)           │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Entity Linking    │  CHẨN_ĐOÁN → ICD-10
│                   │  THUỐC     → RxNorm
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Candidate Ranking │  Order / filter mapping candidates
└─────────┬─────────┘
          ▼
    output/{id}.json
```

## Module map (`src/`)

| Module | Responsibility |
|--------|----------------|
| `pipeline/protocols.py` | `Extractor`, `Linker` Protocols (swap points) |
| `pipeline/pipeline.py` | `ClinicalNLPipeline` — composes stages |
| `extract/rules.py` | `RuleExtractor` (Week 2 baseline) |
| `linking/fuzzy.py` | `FuzzyLinker` (ICD-10 / RxNorm retrieval) |
| `pipeline/pipeline.py` | `ClinicalNLPipeline`, `write_predictions` |
| `notebooks/example_run.ipynb` | Example runs (single + batch + ZIP) |
| `schemas/` | Output JSON validation (functions) |
| `eval/` | Local metrics (functions) |
| `kb/` | KB loading (functions) |

Week 3: add `extract/model.py` with `ViHealthExtractor`; inject via `ClinicalNLPipeline(extractor, linker)`.

## Knowledge bases (`data/kb/`)

| Source | Used for |
|--------|----------|
| ICD-10 | `CHẨN_ĐOÁN` candidate generation |
| RxNorm | `THUỐC` candidate generation |

## Data flow

```
data/
├── kb/              # ICD-10, RxNorm (downloaded, not committed if large)
├── raw/             # Self-created training data
├── processed/       # Tokenized / annotated splits
├── test/
│   ├── input/       # 1.txt … 100.txt (gdown from Google Drive)
│   └── output/      # 1.json … 100.json (our predictions)
└── examples/        # Sample I/O for dev
```

## Phase 1 scope (explicitly out of scope)

- Building a new ontology
- OWL reasoner / Description Logic rules
- Full ontological reasoning (Phase 2+)

Ontology is a **structured knowledge source** for now, not the main deliverable.

## Development sprints

| Sprint | Focus |
|--------|-------|
| Week 1 | Domain research, KB setup, pipeline understanding |
| Week 2 | Literature review → baseline selection → PoC → test-set benchmark |
| Week 3 | Fix weak modules, iterate, submit best result |

Baseline choices are documented in [`baseline_decisions.md`](baseline_decisions.md).

## Candidate approaches (evaluate in Week 2)

- Fine-tuned Vietnamese clinical NER (e.g. PhoBERT, ViHealthBERT)
- Rule-based / dictionary baselines for linking
- LLM + agents for assertion and linking fallback
- Hybrid: NER model + retrieval over ICD/RxNorm
