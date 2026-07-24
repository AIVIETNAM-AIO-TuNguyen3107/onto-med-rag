# GPU GitHub handoff and inference plan

## 1. Goal

Prepare a private, reproducible GitHub repository that a GPU operator can clone,
configure with a new dataset path, and use to generate exactly one JSON output
file per input text file.

The intended model configuration is:

- NER: `Ihor/gliner-biomed-large-v1.0`
- LLM: `Qwen/Qwen3.5-9B`
- local Qwen thinking enabled
- only final structured JSON is parsed; thinking text is discarded
- ICD-10: local Ministry of Health workbook/index
- RxNorm: local cache by default; no clinical text is sent to RxNav

The current `baseline-gliner-v1` run is a useful comparison baseline, but it did
not run Qwen and is not the GPU result.

## 2. Important hardware caveat

The model weights being approximately 19 GB does not mean that a 19 GB GPU is
sufficient. Inference also needs memory for model-loading overhead, activations,
attention/KV cache, generated tokens, and possibly the NER model.

Recommended operating target:

- NVIDIA GPU with at least 24 GB VRAM for an initial attempt;
- 32 GB or more is safer for full-precision/BF16 inference;
- at least 50–70 GB free disk for model cache, Python environment, artifacts,
  and outputs;
- enough system RAM for model loading or CPU offload.

If 24 GB produces an out-of-memory error, reduce the generation budget, run
GLiNER on CPU, use an approved quantized Qwen configuration, or use CPU offload.
Do not quietly switch to another LLM and label the output as Qwen.

## 3. Repository privacy policy

Use a **private GitHub repository** unless the competition data and terminology
files are explicitly licensed for public redistribution.

Never commit:

- `input/` or any replacement test dataset;
- generated `runs/`;
- Hugging Face model weights or cache;
- API tokens, `.env`, SSH keys, or notebook credentials;
- local RxNorm caches derived from private input mentions;
- `.DS_Store`, Python caches, or virtual environments;
- the ICD workbook unless redistribution is explicitly permitted.

Before the first commit, add this `.gitignore`:

```gitignore
.DS_Store
.env
.env.*
!.env.example
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/

input/
data/
runs/
artifacts/
*.log
*.zip
*.tar.gz

.cache/
huggingface/
models/

configs/*.local.yaml
```

Use a tracked empty marker such as `input.example/README.md` only if an example
directory is useful. Do not place real clinical text in it.

## 4. Mandatory pre-push engineering gates

These should be completed and tested before asking the GPU operator to run all
documents.

### 4.1 Add a tracked GPU configuration template

Add `configs/gpu_qwen.example.yaml` using the template in section 8. The operator
copies it to `configs/gpu_qwen.local.yaml`; the local file remains ignored.

The GPU configuration must use:

```yaml
run:
  fail_on_model_unavailable: true
```

This prevents an unavailable Qwen or GLiNER model from silently turning into a
no-model run.

### 4.2 Make Qwen loading explicit and configurable

The Qwen adapter should expose and record:

- model revision or commit hash;
- requested dtype, initially `auto` or BF16 on compatible hardware;
- device map;
- optional CPU-offload directory;
- optional quantization mode;
- actual device and dtype after loading.

Add `torch_dtype="auto"` or the confirmed model-specific equivalent to avoid an
accidental FP32 load. Record all choices in `source_manifest.json`.

### 4.3 Confirm the Transformers API

The environment must successfully import:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
```

The current package declaration says `transformers>=4.57`, but the exact
working Transformers version and model revision must be pinned after the GPU
smoke test. A broad minimum version alone is not a reproducible lock.

### 4.4 Implement bounded Qwen retries

`max_retries` exists in configuration but is not currently used by the local
Qwen adapter. Add bounded retries for:

- unfinished thinking blocks;
- invalid JSON;
- output failing its Pydantic schema;
- CUDA out-of-memory only when a documented lower-memory fallback exists.

Never store raw chain-of-thought in artifacts or ordinary logs.

### 4.5 Add real resume support

The implementation plan mentions `resume-run`, but the current CLI does not
implement it. Before a long GPU run, either:

1. implement `resume-run --run-id RUN_ID`, skipping already validated documents;
   or
2. formally accept that an interruption requires restarting under a new run ID.

Option 1 is strongly preferred because Qwen inference may take hours.

Each existing output must be validated against its source hash before it is
skipped.

### 4.6 Make validation model-free

The current `validate` command builds the entire pipeline before validating.
Move validation ahead of model construction so validation does not reload Qwen.

### 4.7 Add progress and timing records

Print and save, per document:

- document number and total;
- elapsed time;
- NER and LLM call counts;
- GPU peak memory if available;
- warning/failure status.

Do not print input text or Qwen thinking.

### 4.8 Add an end-to-end Qwen smoke test

The test should use synthetic text, not competition input, and verify:

- Qwen is really loaded;
- thinking is enabled;
- thinking is absent from saved output and logs;
- final JSON parses;
- `text[start:end] == entity_text`;
- output remains flat and non-overlapping.

## 5. Files to commit

Use an explicit allowlist for the first commit:

- `src/clinical_nlp/`
- `tests/`
- `configs/base.yaml`
- `configs/baseline_gliner.yaml`
- `configs/gpu_qwen.example.yaml` after it is added
- `pyproject.toml`
- `.gitignore`
- `README.md` or this handoff document
- design and implementation-plan documents, if desired

Handle these separately:

- `DM ICD10-19_8_BYT.xlsx`: commit only if redistribution is permitted;
- `how-to-query-the-rxnorm-data.ipynb`: clear all outputs and credentials before
  committing;
- `artifacts/icd_index.json`: preferably regenerate locally from the workbook;
- `artifacts/rxnorm_cache.json`: do not commit if it contains input-derived
  lookup history.

Inspect the staged commit:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

Also search filenames and staged text for secrets before pushing.

## 6. Suggested GitHub sequence

Create an empty **private** repository on GitHub without an auto-generated
README, then run locally:

```bash
cd /Users/questionminded/coding/ViettelAIrace
git init
git branch -M main
git add .gitignore pyproject.toml src tests configs \
  GPU_GITHUB_HANDOFF_PLAN.md stage1_implementation_plan.md
git status --short
git commit -m "Add modular clinical NLP inference pipeline"
git remote add origin git@github.com:OWNER/REPOSITORY.git
git push -u origin main
```

Do not use `git add .` until `.gitignore` is present and the staged file list has
been reviewed.

If the ICD workbook is not committed, send it to the GPU operator through an
approved private channel and have them configure its absolute path.

## 7. GPU operator: clone and create the environment

```bash
git clone git@github.com:OWNER/REPOSITORY.git
cd REPOSITORY

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

Install a CUDA-enabled PyTorch build compatible with the host's NVIDIA driver.
Use the official PyTorch installation selector for that machine, then install
the project:

```bash
python -m pip install -e ".[models,retrieval,test]"
```

Record the environment after the successful smoke test:

```bash
python -m pip freeze > requirements-lock-gpu.txt
```

The lock should be reviewed before committing because CUDA-specific PyTorch
wheels are not portable to every GPU host.

### 7.1 Hardware preflight

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"
python -c "from transformers import AutoProcessor, AutoModelForMultimodalLM; print('Qwen Transformers API: OK')"
```

Stop if CUDA is unavailable or the Qwen import fails.

### 7.2 Hugging Face access

If the model requires authentication, log in using the operator's Hugging Face
account or export `HF_TOKEN` in the shell. Never write the token into YAML,
source code, shell history shared with others, or Git.

Put the Hugging Face cache on a disk with sufficient capacity:

```bash
export HF_HOME=/ABSOLUTE/PATH/WITH/ENOUGH/SPACE/huggingface
mkdir -p "$HF_HOME"
```

## 8. GPU operator: place the new dataset and configure paths

Keep the dataset outside the Git clone. Recommended layout:

```text
/data/viettel_stage1/
├── input/
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
├── terminology/
│   ├── DM ICD10-19_8_BYT.xlsx
│   └── rxnorm_cache.json
├── artifacts/
└── runs/
```

Requirements:

- every input is a UTF-8 `.txt` file;
- the filename stem is the document ID;
- place the `.txt` files directly inside the configured `input_dir`;
- do not normalize Unicode, trim, rewrite line endings, or otherwise edit input;
- each final output will use the same stem, for example `17.txt` → `17.json`.

Copy the tracked template:

```bash
cp configs/gpu_qwen.example.yaml configs/gpu_qwen.local.yaml
```

Set absolute paths in the local file:

```yaml
paths:
  input_dir: /data/viettel_stage1/input
  runs_dir: /data/viettel_stage1/runs
  artifacts_dir: /data/viettel_stage1/artifacts
  icd_source: /data/viettel_stage1/terminology/DM ICD10-19_8_BYT.xlsx
  icd_index: /data/viettel_stage1/artifacts/icd_index.json
  rxnorm_cache: /data/viettel_stage1/terminology/rxnorm_cache.json

entity_finding:
  chunk_chars: 1400
  chunk_overlap_chars: 160
  ner_threshold: 0.35

ner:
  backend: gliner
  model_id: Ihor/gliner-biomed-large-v1.0
  local_files_only: false

llm:
  backend: qwen_transformers
  model_id: Qwen/Qwen3.5-9B
  local_files_only: false
  reasoning_effort: high
  thinking: true
  max_new_tokens: 4096
  max_retries: 2

linking:
  icd_max_candidates: 3
  rxnorm_max_candidates: 3
  use_rxnav_api: false

run:
  fail_on_model_unavailable: true
  pretty_json: true
```

Notes:

- `thinking: true` is active in the local adapter.
- `reasoning_effort: high` is currently metadata for local Qwen; it is not a
  separate Transformers generation parameter. Thinking mode and token budget
  control the local reasoning behavior.
- `use_rxnav_api: false` prevents external RxNorm requests. Without a populated
  local RxNorm cache, many medication candidate arrays will be empty.
- Use `local_files_only: false` for the first model download. After both models
  are cached, optionally change it to `true` for a fully offline run.

## 9. Build terminology and run tests

```bash
source .venv/bin/activate

clinical-nlp --config configs/gpu_qwen.local.yaml build-icd
pytest -q
```

Confirm that the ICD command reports a nonzero concept count and writes the
configured `icd_index`.

## 10. Run one GPU smoke document

Choose a representative document ID that exists in the new input directory:

```bash
nvidia-smi
clinical-nlp \
  --config configs/gpu_qwen.local.yaml \
  run-document \
  --document 1 \
  --run-id qwen35-smoke
```

While it runs, monitor:

```bash
watch -n 2 nvidia-smi
```

The smoke test is accepted only if:

- the command exits successfully;
- `runs/qwen35-smoke/outputs/1.json` exists at the configured runs path;
- validation status is `ok`;
- the source manifest says NER is `gliner`;
- the source manifest says LLM is `qwen_transformers`, not `noop`;
- warnings do not say that NER or LLM was unavailable;
- no `<think>` content or reasoning text appears in output or artifacts;
- manual inspection confirms sensible spans, candidates, and assertions.

Do not proceed to all inputs if Qwen silently fell back, output JSON is invalid,
or GPU memory is unstable.

## 11. Estimate runtime before the full run

Record smoke-test elapsed time and LLM-call count. The full pipeline may call
Qwen for entity recovery and ambiguous terminology reranking, so total time is
not simply `number_of_documents × one generation`.

Run two or three varied documents before estimating completion time. Do not
assume the GLiNER-only M1 baseline timing predicts Qwen GPU timing.

## 12. Generate all result files

Use a unique immutable run ID:

```bash
clinical-nlp \
  --config configs/gpu_qwen.local.yaml \
  infer \
  --run-id qwen35-9b-full-YYYYMMDD
```

Expected location:

```text
/data/viettel_stage1/runs/qwen35-9b-full-YYYYMMDD/outputs/
```

The directory must contain one JSON file per input file with matching stems.

If resume support has been implemented and the run is interrupted:

```bash
clinical-nlp \
  --config configs/gpu_qwen.local.yaml \
  resume-run \
  --run-id qwen35-9b-full-YYYYMMDD
```

Do not use that command until it actually exists and is tested.

## 13. Validate the completed results

After the model-free validation change is implemented:

```bash
clinical-nlp \
  --config configs/gpu_qwen.local.yaml \
  validate \
  --run-id qwen35-9b-full-YYYYMMDD
```

Acceptance criteria:

- input and output filename stems match exactly;
- every output is a JSON array;
- every entity conforms to the allowed schema;
- `text[start:end] == entity["text"]` on the original input;
- positions use Python Unicode offsets and end-exclusive ends;
- entities are sorted, flat, and non-overlapping;
- only allowed entity types and assertions appear;
- only diagnosis and medication entities contain `candidates`;
- no model reasoning appears anywhere;
- `source_manifest.json` records the intended model backends and input hashes;
- validation summary contains no failed document.

Review aggregate counts and sample documents. A technically valid run can still
contain NER or linking false positives.

## 14. Package results for return

Package the final output directory separately from internal audit artifacts:

```bash
cd /data/viettel_stage1/runs/qwen35-9b-full-YYYYMMDD
tar -czf qwen35-9b-results-YYYYMMDD.tar.gz outputs
shasum -a 256 qwen35-9b-results-YYYYMMDD.tar.gz
```

Return through an approved private channel:

- the result archive;
- its SHA-256 checksum;
- `source_manifest.json`;
- `config.json`;
- validation summary;
- warnings and timing summary.

Do not push result files or input-derived artifacts to GitHub.

## 15. Final responsibility split

Repository owner:

- completes the mandatory pre-push gates;
- creates and reviews the private repository;
- provides the ICD/RxNorm files through an approved channel;
- provides the expected input count and naming convention.

GPU operator:

- installs a verified CUDA/PyTorch environment;
- places the new dataset outside the repository;
- sets absolute paths in `gpu_qwen.local.yaml`;
- runs tests and a one-document Qwen smoke test;
- verifies the manifest shows the real models;
- runs, validates, and packages the complete output.

Repository owner after return:

- verifies the archive checksum;
- reruns model-free validation against the exact original inputs;
- reviews quality distributions before using a limited competition submission.
