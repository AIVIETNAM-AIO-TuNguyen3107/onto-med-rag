# Vietnamese clinical NLP inference pipeline

This repository extracts Vietnamese clinical entities, assertions, ICD-10
candidates, and RxNorm candidates. It writes one JSON array per input text file
using Python Unicode offsets and end-exclusive spans.

The intended GPU configuration uses:

- `Ihor/gliner-biomed-large-v1.0` for NER;
- `Qwen/Qwen3.5-9B` with thinking enabled for recovery and reranking;
- a crawled Vietnamese ICD-10 TT06 catalog or local workbook/index;
- a local RxNorm cache by default.

Competition input, terminology workbooks, model weights, local configurations,
and generated runs are deliberately excluded from Git.

## Crawl the Vietnamese ICD-10 TT06 catalog

The website is a JavaScript application, but its classification tree is exposed
through a public JSON API. Crawl it sequentially into ignored local artifacts:

```bash
clinical-nlp --config configs/base.yaml crawl-icd
```

The command writes:

```text
artifacts/icd10_tt06_vi.jsonl
artifacts/icd10_tt06_vi.csv
artifacts/icd10_tt06_vi.manifest.json
artifacts/icd_index.json
```

The JSONL catalog is canonical. Each row records the node model, source ID,
code, Vietnamese name, parent, ancestry path, depth, zero-based sibling order,
and leaf flag. Paths preserve reused nodes that appear under multiple parents;
their children are fetched once and replayed under each path. The CSV contains
the same fields. The manifest records source URLs, retrieval time, occurrence
and unique-node counts, model counts, request count, and file hashes.

If a crawl is interrupted, rerun it with `--resume`. A completed snapshot is
never replaced implicitly; use `--force` to deliberately start a fresh crawl
and atomically replace it:

```bash
clinical-nlp --config configs/base.yaml crawl-icd --resume
clinical-nlp --config configs/base.yaml crawl-icd --force
```

The crawler uses one worker, waits 0.5 seconds between requests, retries
temporary failures, and honors `Retry-After`. These settings and the API/source
URLs are configurable under `icd_crawl`. When the catalog exists it is preferred
for ICD index construction; otherwise the configured `paths.icd_source`
workbook remains the fallback.

## GPU quick start

Use Python 3.11 and install a CUDA-enabled PyTorch build appropriate for the GPU
host before installing the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e ".[models,retrieval,test]"
```

Copy the GPU configuration and edit all paths:

```bash
cp configs/gpu_qwen.example.yaml configs/gpu_qwen.local.yaml
```

Place UTF-8 inputs directly in the configured `input_dir`, for example
`1.txt`, `2.txt`, and so on. Do not normalize or edit the source text.

Run the preflight, a one-document smoke test, then the full run:

```bash
pytest -q
clinical-nlp --config configs/gpu_qwen.local.yaml build-icd
clinical-nlp --config configs/gpu_qwen.local.yaml run-document \
  --document 1 --run-id qwen35-smoke
clinical-nlp --config configs/gpu_qwen.local.yaml infer \
  --run-id qwen35-9b-full
clinical-nlp --config configs/gpu_qwen.local.yaml validate \
  --run-id qwen35-9b-full
```

Final result files are under the configured:

```text
runs_dir/qwen35-9b-full/outputs/
```

Before running private data, read
[GPU_GITHUB_HANDOFF_PLAN.md](GPU_GITHUB_HANDOFF_PLAN.md) for hardware,
privacy, model-verification, validation, and packaging requirements.

## Strict hosted-Qwen first-five run

The first-five configuration keeps GLiNER local, calls
`Qwen/Qwen3.5-9B:fastest` through the Hugging Face router, uses the canonical
TT06 JSONL ICD catalog, and queries RxNav with a persistent local cache. Copy
the tracked template so machine-specific settings stay uncommitted:

```bash
cp configs/online_first5.example.yaml configs/online_first5.local.yaml
export HF_TOKEN="a-newly-rotated-token"
```

Never put the token in YAML, shell history, logs, commits, or run artifacts.
The strict preflight loads the cached GLiNER model, verifies a real Qwen JSON
response and returned model identity, queries RxNav, and records terminology
and input hashes:

```bash
clinical-nlp --config configs/online_first5.local.yaml preflight \
  --documents 1 2 3 4 5 \
  --run-id online-qwen-rxnav-icd10-first5-preflight
```

Run exactly the selected documents:

```bash
clinical-nlp --config configs/online_first5.local.yaml infer \
  --documents 1 2 3 4 5 \
  --run-id online-qwen-rxnav-icd10-first5

clinical-nlp --config configs/online_first5.local.yaml validate \
  --documents 1 2 3 4 5 \
  --run-id online-qwen-rxnav-icd10-first5
```

The run refuses to reuse a non-empty output directory. Final submission files
are the five JSON arrays under `runs/online-qwen-rxnav-icd10-first5/outputs/`.
`quality_summary.json` reports type/assertion/link coverage and empty candidate
lists; per-document audit directories preserve proposals, constrained Qwen
decisions, retrieved and selected terminology candidates, warnings, and
non-secret model metadata. Reasoning content and credentials are never stored.
