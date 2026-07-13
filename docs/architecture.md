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
| `ner/` | Span detection, character offsets |
| `classification/` | Assign one of 5 entity types |
| `assertion/` | Context flags per entity |
| `linking/icd10/` | Diagnosis → ICD-10 candidates |
| `linking/rxnorm/` | Drug → RxNorm candidates |
| `ranking/` | Rank and trim candidate lists |
| `pipeline/` | Orchestrate end-to-end inference |
| `schemas/` | Output JSON validation |

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
