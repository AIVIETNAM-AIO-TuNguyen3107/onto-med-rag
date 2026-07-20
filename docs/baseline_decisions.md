# Baseline Decisions

> Week 2 PoC locked 2026-07-20. Update after benchmark/error analysis.

## Comparison summary

| Task | Candidates reviewed | Selected baseline | Rationale |
|------|---------------------|-------------------|-----------|
| Medical NER | ViHealthBERT, PhoBERT, Qwen2.5-7B, rules | **Rules (`src/extract/rules.py`)** | Fastest reproducible PoC; no GPU/training needed for Week 2 |
| Entity classification | Same as NER (joint) | **Rules (pattern → type)** | Wrong type double-penalizes score; rules OK for structured sections |
| Assertion detection | Keyword rules, negation models | **Keyword window rules** | 3 labels only; high precision on obvious cues |
| ICD-10 linking | BM25, RapidFuzz, embedding retrieval | **RapidFuzz token_set_ratio** | Zero setup, works on small KB samples immediately |
| RxNorm linking | Same | **RapidFuzz token_set_ratio** | Same |
| Candidate ranking | Cross-encoder rerank | **Top-k fuzzy (no rerank yet)** | YAGNI until full KB indexed |

## Week 3 upgrade path

| Task | Target |
|------|--------|
| NER + classification | Fine-tuned **ViHealthBERT** token classifier OR self-hosted **Qwen2.5-7B-Instruct** JSON extraction |
| Linking | Full ICD-10/RxNorm index + optional embedding rerank |
| Assertions | Keep keyword rules; add model if error analysis shows gaps |

## Initial benchmark (Week 2 PoC)

| Metric / observation | Result | Notes |
|----------------------|--------|-------|
| Test set coverage | pending | Run after `gdown` download |
| NER quality | pending | Spot-check `data/examples/` |
| Linking quality | pending | Sample KB only until full RxNorm/ICD loaded |
| Assertion quality | pending | Rules tuned for `isHistorical` drug lists |
| Known failure modes | TBD | Unstructured narrative text, ambiguous spans |

## Week 3 improvement plan

1. Download full ICD-10 + RxNorm → `data/kb/`
2. Error analysis on 20 hardest test files
3. Replace rule extractor with fine-tuned model or ≤9B LLM; keep pipeline + linking
