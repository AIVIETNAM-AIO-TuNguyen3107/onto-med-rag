# Vietnamese clinical NLP pipeline

Extract Vietnamese clinical entities and assertions, then link diagnoses to
ICD-10 (TT06) and medications to RxNorm. Each input `*.txt` becomes one JSON
array of entities with Unicode offsets and end-exclusive spans.

**Stack (default GPU path):** GLiNER biomed NER → Qwen3.5-9B for recovery /
review / rerank → TT06 ICD catalog → RxNorm (cache and/or RxNav).

Private inputs, model weights, `configs/*.local.yaml`, and `runs/` stay out of
Git.

## Install

Python 3.11+. Use a CUDA PyTorch build that matches the GPU host, then:

```bash
uv sync --extra models --extra retrieval --extra test
# or: pip install -e ".[models,retrieval,test]"
```

`bitsandbytes` is included under `[models]` for optional 4-bit local Qwen loads.

## Configuration

Copy an example config to a **gitignored** local file and edit paths / models:

| Template | Use when |
| --- | --- |
| `configs/gpu_qwen.example.yaml` | Local Transformers Qwen on GPU |
| `configs/online_first5.example.yaml` | Hosted Qwen via Hugging Face router |
| `configs/base.yaml` | Crawl / light local tooling |

```bash
cp configs/gpu_qwen.example.yaml configs/gpu_qwen.local.yaml
# edit paths, model_id, quantization, thinking, RxNav, …
```

Put UTF-8 notes in `input_dir` as `1.txt`, `2.txt`, … Do not normalize source text.

### Local GPU notes (~16GB)

- Set `llm.quantization: bnb_4bit` so the 9B model stays on GPU (full BF16 often CPU-offloads and crawls).
- Prefer `thinking: false` for reliable JSON; long thinking burns `max_new_tokens` and triggers retries.
- Assertion review over many entities needs a large `max_new_tokens` (e.g. 4096).
- `infer` refuses a non-empty `runs/<run-id>/outputs/`; use a new `--run-id` or clear that folder.

## ICD-10 TT06 catalog

Crawl the public TT06 JSON API into ignored artifacts (or fall back to the
configured workbook):

```bash
clinical-nlp --config configs/base.yaml crawl-icd
clinical-nlp --config configs/base.yaml crawl-icd --resume   # interrupted crawl
clinical-nlp --config configs/base.yaml crawl-icd --force    # replace snapshot
```

Writes `artifacts/icd10_tt06_vi.{jsonl,csv}`, a manifest, and rebuilds
`artifacts/icd_index.json`. Tune `icd_crawl` in the config (delay, retries, URLs).

Build / refresh the ICD index alone:

```bash
clinical-nlp --config configs/gpu_qwen.local.yaml build-icd
```

## Local Transformers run

```bash
pytest -q

clinical-nlp --config configs/gpu_qwen.local.yaml run-document \
  --document 1 --run-id qwen35-smoke

clinical-nlp --config configs/gpu_qwen.local.yaml infer \
  --documents 1 2 3 4 5 \
  --run-id local-qwen-rxnav-icd10-first5

clinical-nlp --config configs/gpu_qwen.local.yaml validate \
  --documents 1 2 3 4 5 \
  --run-id local-qwen-rxnav-icd10-first5
```

Submission arrays: `runs/<run-id>/outputs/*.json`. Per-document audits live under
`runs/<run-id>/documents/<id>/` (proposals, candidates, warnings, non-secret
model metadata). Thinking traces and credentials are never stored.

## Hosted Qwen first-five (HF router)

GLiNER stays local; Qwen runs as `Qwen/Qwen3.5-9B:fastest` through the HF
router. TT06 JSONL + RxNav with a persistent cache.

```bash
cp configs/online_first5.example.yaml configs/online_first5.local.yaml
export HF_TOKEN="…"   # never put the token in YAML, logs, or commits
```

```bash
clinical-nlp --config configs/online_first5.local.yaml preflight \
  --documents 1 2 3 4 5 \
  --run-id online-qwen-rxnav-icd10-first5-preflight

clinical-nlp --config configs/online_first5.local.yaml infer \
  --documents 1 2 3 4 5 \
  --run-id online-qwen-rxnav-icd10-first5

clinical-nlp --config configs/online_first5.local.yaml validate \
  --documents 1 2 3 4 5 \
  --run-id online-qwen-rxnav-icd10-first5
```

Preflight checks GLiNER, a live Qwen JSON response / model id, RxNav, and
terminology / input hashes.

## Entity types

Only these types are kept in outputs:

`TRIỆU_CHỨNG` · `TÊN_XÉT_NGHIỆM` · `KẾT_QUẢ_XÉT_NGHIỆM` · `CHẨN_ĐOÁN` · `THUỐC`

LLM recovery rows with other labels (e.g. procedures) are dropped at parse time.

## More detail

Hardware, privacy, and packaging expectations:
[GPU_GITHUB_HANDOFF_PLAN.md](GPU_GITHUB_HANDOFF_PLAN.md).
