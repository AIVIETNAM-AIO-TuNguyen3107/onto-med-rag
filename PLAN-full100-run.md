# Finish the 100-document competition run

## Handoff notes for the implementing agent

Repo: `/Users/questionminded/coding/ViettelAIrace`

**Read this before touching any file.** Every file path and line number below refers to branch **`review/selective-fast-first5`**, which is *not* what is currently checked out. The repo is on `review/api-first5-hardening`, where `src/clinical_nlp/pipeline.py` is a different, much shorter file — line 66 there is not `NUMERIC_RESULT_RE`, and `_needs_review`, `_valid_laboratory_result_span`, and `llm/cache.py` do not exist at all.

**Do Step 1 first and completely.** Only after the branch switch will the referenced code exist. Then re-locate each symbol by name (`NUMERIC_RESULT_RE`, `_needs_review`, `run_all`) rather than trusting the line numbers, which may have shifted.

The investigation behind this plan is already done — all findings below were measured against the real corpus and run artifacts. You do not need to re-derive them. What you do need to verify is that each edit lands in the right place on the right branch.

## Context

The Viettel "Bài 2 — Ontological Reasoning in Medical Knowledge Retrieval" Phase-1 test set is 100 Vietnamese clinical `.txt` files in [input/](input/). The pipeline must emit one JSON array per file with `text`, `position` (0-indexed Python Unicode offsets, end-exclusive), `type`, `assertions`, `candidates`.

Inference stalled on 2026-07-25 at 22:21 inside `runs/openrouter-selective-first5-v3-20260725`, partway through document 3. Documents 1 and 2 have outputs; document 3 has only checkpoints. 98 documents remain unprocessed.

Investigating the stall surfaced four separate problems. Only one of them is the stall itself — the other three are accuracy/throughput defects that would have degraded the submission even if the run had finished.

### What I found

**1. The good pipeline is not the checked-out one.** The runs from 2026-07-25 were produced by branch `review/selective-fast-first5` (commit `bb56eb0`), which lives in a worktree under `/private/tmp/onto-med-rag-selective-first5`. The main repo sits on `review/api-first5-hardening` (`9b3e33f`), which lacks all of it. The selective branch is 6 commits and +2513/−329 lines ahead, adding: selective review mode, [src/clinical_nlp/llm/cache.py](src/clinical_nlp/llm/cache.py) (sqlite response cache), in-document concurrency, terminology auto-accept thresholds, per-call checkpoints, and [tests/unit/test_selective_pipeline.py](tests/unit/test_selective_pipeline.py). `/private/tmp` is purgeable by macOS; the commits are backed up on `team/review/selective-fast-first5`, but the working setup is fragile.

**2. Lab-result recall is near zero because of a regex bug.** In [src/clinical_nlp/pipeline.py:66](src/clinical_nlp/pipeline.py#L66):

```python
NUMERIC_RESULT_RE = re.compile(
    r"(?ix)^[<>]=?\s*[+-]?\d+(?:[.,]\d+)?"   # <-- [<>] is NOT optional
    ...
)
```

The comparison-operator prefix is mandatory, so `6.3`, `80%`, and `38.3°C` all fail `fullmatch`. Measured against the 100-document GLiNER baseline: **the current pattern auto-accepts 1 of 340 `KẾT_QUẢ_XÉT_NGHIỆM` spans.** Everything else trips `laboratory_span_policy`, goes to LLM review, and gets pruned. The golden example in [data/examples/sample_output.json](data/examples/sample_output.json) scores bare numerics (`"14,43"`, `"76,4"`) as their own entities, so this is a whole scored entity type sitting at zero.

**3. No document-level parallelism.** [src/clinical_nlp/supervision/runner.py:275](src/clinical_nlp/supervision/runner.py#L275) loops documents strictly sequentially; `llm.max_concurrency` only parallelizes batches *within* a document. Document 2 took 139s wall / 191s LLM latency. 100 documents ≈ 2.5–3 hours.

**4. The stalled run cannot be resumed into a full run.** Its `source_manifest.json` records `selection.document_ids: ["1","2","3","4","5"]`, and `_resume_manifest_signature` ([runner.py:460](src/clinical_nlp/supervision/runner.py#L460)) includes `selection`. A 100-document `--resume` would fail the compatibility check. A new run ID is required.

**Not a problem:** the selective LLM review is doing genuinely good work. On document 2 it dropped 14 baseline spans and every one was noise — `'bệnh'` ("disease") with candidates `A00/A54/A82`, `'Điều trị'` ("treatment") typed as `THUỐC`, `'thuốc lá thụ động'` ("secondhand smoke") typed as `THUỐC`, `'động mạch vành'` ("coronary artery") typed as `CHẨN_ĐOÁN`. Keep this behaviour; the fix in step 2 is scoped so it does not weaken it.

### Decisions taken

Model stays `qwen/qwen3.5-9b` (your call). Cost is ~$0.003/document, so the full set runs at roughly $0.30.

---

## Step 1 — Consolidate onto the selective branch

1. Commit the ~200 uncommitted lines on `review/api-first5-hardening` so nothing is lost. They are a parallel attempt at per-call `reasoning_enabled` plumbing that `review/selective-fast-first5` already solves more thoroughly ([llm/base.py:29](src/clinical_nlp/llm/base.py#L29)); the commit is a safety net, not something to merge.
2. `git checkout review/selective-fast-first5` in `/Users/questionminded/coding/ViettelAIrace`. This requires removing the `/private/tmp/onto-med-rag-selective-first5` worktree first (`git worktree remove`), since a branch cannot be checked out twice. Its only uncommitted change is `uv.lock`.
3. Run `pytest -q` to confirm the suite is green in the primary worktree before changing anything.

Steps 2 and 3 are then committed on top of `review/selective-fast-first5`.

## Step 2 — Fix lab-value auto-accept

In [src/clinical_nlp/pipeline.py:66](src/clinical_nlp/pipeline.py#L66), make the comparison prefix optional and extend the unit alternation to the forms that actually occur in this corpus:

```python
_RESULT_UNIT = (
    r"%|°\s*[CF]|g/L|mg/L|mg/dL|mmol/L|µmol/L|umol/L|U/L|IU/L|"
    r"10\^?\d+/L|x10\^?\d+/L|mmHg|bpm|lần/phút|nhịp/phút|kg|cm|mm|ml|cc"
)
NUMERIC_RESULT_RE = re.compile(
    r"(?ix)^(?:[<>]=?\s*)?[+-]?\d+(?:[.,]\d+)?"
    r"(?:\s*[-–/]\s*[+-]?\d+(?:[.,]\d+)?)?"
    rf"(?:\s*(?:{_RESULT_UNIT}))?$"
)
```

Two deliberate choices, both measured against the 340 baseline spans:

- The `/` in the range group admits blood pressure (`139/68 mmHg`, `160/70 mmHg`).
- Bare mass units (`mg`, `g`, `mcg`) are **excluded**. They match drug doses (`5mg`, `2,5g`, `2mcg`), and auto-accepting those as `KẾT_QUẢ_XÉT_NGHIỆM` would cost precision. They keep going to LLM review.

Measured effect: auto-accept goes from **1 → 160 of 340** spans. What still routes to review is correctly the ambiguous residue — `'1 viên'` ("1 tablet"), `'m'`, `'ct'`, `'ure'`, `'Nhiệt độ'` ("temperature", a test *name*), `'không có gì đáng chú ý'` ("nothing notable").

Also suppress the `low_confidence_gliner_only` review reason for `TEST_RESULT` proposals that satisfy `_valid_laboratory_result_span` (in `_needs_review`, [pipeline.py:1195-1222](src/clinical_nlp/pipeline.py#L1195-L1222)). Without this a valid numeric with a sub-0.65 GLiNER score still gets routed to review and pruned — that is exactly how `'80%'` was lost on document 2 despite being a well-formed value. This is a gating change in code, not a prompt change, so it does not invalidate cached responses.

Add cases to [tests/unit/test_selective_pipeline.py](tests/unit/test_selective_pipeline.py) covering: bare numeric accepted, `38.3°C` accepted, `139/68 mmHg` accepted, `5mg` still routed to review, `'1 viên'` still routed to review.

To confirm your edited regex reproduces the 1 → 160 result before running any inference, check it against the 100-document baseline that is already on disk:

```bash
python3 -c "
import json,glob,collections
from clinical_nlp.pipeline import NUMERIC_RESULT_RE, QUALITATIVE_RESULT_RE
v=collections.Counter()
for f in glob.glob('runs/baseline-gliner-v1/outputs/*.json'):
    for e in json.load(open(f)):
        if e['type']=='KẾT_QUẢ_XÉT_NGHIỆM': v[e['text'].strip()]+=1
ok=sum(c for t,c in v.items() if NUMERIC_RESULT_RE.fullmatch(t) or QUALITATIVE_RESULT_RE.fullmatch(t))
print(f'auto-accepted {ok} of {sum(v.values())}')
print('still to review:', [t for t in v if not (NUMERIC_RESULT_RE.fullmatch(t) or QUALITATIVE_RESULT_RE.fullmatch(t))][:15])
"
```

Expect ~160. If you get 1, the `[<>]` prefix is still mandatory. If you get well above 200, the unit list is too permissive and is swallowing drug doses — check that `mg`, `g`, and `mcg` are absent as standalone units.

## Step 3 — Document-level parallelism

Add to `RunConfig` in [src/clinical_nlp/config.py](src/clinical_nlp/config.py):

```python
document_concurrency: int = Field(default=1, ge=1, le=8)
```

Default 1 keeps existing behaviour and existing tests unchanged.

In `run_all` ([runner.py:275](src/clinical_nlp/supervision/runner.py#L275)), when `document_concurrency > 1`, dispatch documents through a `ThreadPoolExecutor` instead of the serial loop. Keep the pre-loop output-directory guards exactly as they are. Collect results into a dict keyed by document id and re-sort to the original `files` order before writing summaries, so `quality_summary.json` and the stage summaries stay byte-identical to a serial run. Print progress under a lock as each document completes.

Three shared-state fixes are required — I verified which are already safe and which are not:

| Component | Status | Action |
|---|---|---|
| `LLMResponseCache` | safe — `threading.Lock` + WAL sqlite | none |
| `OpenAICompatibleBackend._call_audits` | safe — `_state_lock`, filtered per document by `call_id` prefix | none |
| `pipeline._retrieval_cache` | safe — `_retrieval_lock` | none |
| Per-document artifact + output writes | safe — isolated dirs, `os.replace` + fsync | none |
| **GLiNER `self.model.predict_entities`** | **unsafe** — one shared torch model | add a `threading.Lock` in [src/clinical_nlp/ner/gliner.py](src/clinical_nlp/ner/gliner.py) around the predict call |
| **`RxNormIndex.save()`** | **unsafe** — full-file rewrite on every cache miss ([rxnorm_linking/index.py:190](src/clinical_nlp/rxnorm_linking/index.py#L190)) | add a `threading.Lock` around the mutate-then-save section |

Serializing GLiNER costs nothing: it is fast local inference and the bottleneck is OpenRouter latency.

## Step 4 — Config for the full run

Copy `configs/openrouter_selective_first5.example.yaml` → `configs/openrouter_full100.local.yaml` (gitignored by the `configs/*.local.yaml` rule). Change from the example:

```yaml
paths:
  rxnorm_cache: artifacts/rxnorm_full100_cache.json   # fresh, avoids contending with the first5 cache
  llm_cache: artifacts/llm_response_cache_openrouter.sqlite3   # keep — reuses the 49 cached rows

llm:
  max_concurrency: 2      # unchanged from the validated first5 runs

run:
  document_concurrency: 4 # new
```

`llm.max_concurrency` stays at 2 on purpose: that is the value that produced the validated documents 1–2, so the only variable changing is the outer loop. Total in-flight OpenRouter requests ≈ 8. Everything else — `reasoning_max_tokens: 1024`, `structured_outputs: true`, `llm_review_mode: selective`, `gliner_review_threshold: 0.65` — carries over unchanged.

## Step 5 — Execute

```bash
export OPENROUTER_API_KEY="..."   # not in YAML, not in shell history

clinical-nlp --config configs/openrouter_full100.local.yaml preflight --run-id full100-v1
clinical-nlp --config configs/openrouter_full100.local.yaml infer    --run-id full100-v1
clinical-nlp --config configs/openrouter_full100.local.yaml validate --run-id full100-v1
```

All 100 documents, not 98 — documents 1 and 2 get reprocessed so the whole submission comes from one configuration. Expect ~40–50 minutes wall clock and ~$0.30. Run `infer` in the background and monitor the streamed per-document progress JSON.

If it is interrupted, `--resume` now works correctly because the selection is the full set: it re-validates each completed output against the original text and all audit artifacts, skips only what genuinely passes, and reprocesses the rest.

## Cost control

Measured from the completed runs: document 1 cost $0.00126, document 2 cost $0.00267. Mean ≈ $0.002/document, so **the full 100-document run is roughly $0.20–0.30.**

Guards against overspending:

1. **Smoke test before the full run.** `infer --documents 1 2 3 --run-id full100-smoke` costs about $0.01 and proves the new config, the regex change, and document parallelism all work. Only launch the 100-document run after it passes. This is the main protection against burning a full run on a broken config — the kind of thing that produced seven abandoned run directories already.
2. **Set a spend cap on the OpenRouter key** in their dashboard before launching. Cheapest possible insurance.
3. **The sqlite response cache means interruptions are not re-paid for.** A resumed run re-uses completed work rather than re-billing it.
4. **Watch the streamed progress for retry storms.** `http_attempts` exceeding `api_calls` by a wide margin means retries are being billed. Document 2 already showed 12 attempts for 9 calls. If that ratio worsens under concurrency, lower `document_concurrency`.

Worth knowing but **not** changing: reasoning is ~80% of completion spend (document 2 burned 5,492 reasoning tokens out of 6,851 completion tokens). `reasoning_max_tokens: 1024` already caps it. Lowering it further would cut cost meaningfully but is a direct accuracy trade, which runs against your stated preference — flagging it as a lever, not proposing it.

## Verification

1. `pytest -q` green, including the new lab-value and concurrency cases.
2. `clinical-nlp validate --run-id full100-v1` passes — this re-checks every output against the original source text for offset integrity, non-overlap, and exact substring match.
3. Confirm 100 files in `runs/full100-v1/outputs/`.
4. Check `runs/full100-v1/quality_summary.json` against the baseline distribution — the target is all five types present with `KẾT_QUẢ_XÉT_NGHIỆM` well above zero, and non-empty `assertions` (the 100-document baseline found 169 `isHistorical`, 67 `isNegated`, 6 `isFamily`; documents 1–2 are explainer articles with no patient history, which is why they legitimately produced none).
5. Spot-check a structured lab document by diffing against `runs/baseline-gliner-v1/outputs/<id>.json` to confirm lab values return without the noise spans coming back.
6. Sanity-check one output against [data/examples/sample_output.json](data/examples/sample_output.json) for field shape and ordering.

---

## Caveats you should weigh

**Live API key in the working tree.** `temporarytoken.csv` (74 bytes, mode 0600) sits at the repo root. It is gitignored, but it is a credential in plaintext on disk. The README's own guidance is to keep tokens out of files entirely. Recommend deleting it and exporting `OPENROUTER_API_KEY` in the shell instead — I have not read or touched the file.

**30 of 100 inputs have masked drug names.** Files including `1, 2, 4, 7, 9, 14, 15, 16, 18, 19, 24, 25, 28, 30, 40` contain `************` where a drug name was redacted. Those `THUỐC` entities are unrecoverable as literal spans — an inherent recall ceiling, nothing to fix.

**`'80%'`-class losses are only partly addressed.** The gating change in step 2 recovers values that are well-formed but low-confidence. Spans the reviewer drops on genuine judgment would need the review-prompt work you deferred. Prompt edits also bust the response cache, so that is better done as a separate measured experiment after a submittable run exists.

**Concurrency and rate limits.** 8 in-flight requests may draw 429s from OpenRouter. The backend already retries with `Retry-After` honoured and capped at 30s, so this degrades throughput rather than failing — but if the progress log fills with retries, drop `document_concurrency` to 2.

**Environment mismatch.** `.venv` is CPython 3.12.7 while the README specifies 3.11, and `.pytest_cache` shows pytest 7.4.4 against a `pytest>=8` requirement. Nothing has broken so far; flagging it in case a dependency behaves unexpectedly mid-run.

**Cache reuse will be lower than it looks.** The sqlite cache key includes the full `messages` payload. Changing which entities get routed to review changes batch composition, so most of the 49 cached rows will miss. At $0.30 total this does not matter.
