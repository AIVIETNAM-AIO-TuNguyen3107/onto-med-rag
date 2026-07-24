# Vietnamese clinical NLP inference pipeline

This repository extracts Vietnamese clinical entities, assertions, ICD-10
candidates, and RxNorm candidates. It writes one JSON array per input text file
using Python Unicode offsets and end-exclusive spans.

The intended GPU configuration uses:

- `Ihor/gliner-biomed-large-v1.0` for NER;
- `Qwen/Qwen3.5-9B` with thinking enabled for recovery and reranking;
- a local ICD-10 workbook/index;
- a local RxNorm cache by default.

Competition input, terminology workbooks, model weights, local configurations,
and generated runs are deliberately excluded from Git.

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
