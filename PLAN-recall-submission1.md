# Raise competition score from 29.18 — recall first

## Handoff notes for the implementing agent

Repo: `/Users/questionminded/coding/ViettelAIrace`. Background analysis: `REVIEW-scoring-improvements.md` in the repo root — read it for the evidence behind these changes; you do not need to re-derive any of it.

**Scope: implement Submission 1 only.** Sections "Submission 2", "Submission 3", and "Submission 4" are the roadmap for *later* runs and must NOT be implemented now. The whole point of this run is a clean single-variable read on whether recall is the bottleneck. Bundling changes destroys that signal. Stop after the density check and hand back.

**Do not submit anything to the competition.** Producing `runs/<run_id>/outputs/` is the deliverable; submitting is the user's action.

Before starting, confirm the branch:
```bash
git -C /Users/questionminded/coding/ViettelAIrace branch --show-current   # expect: review/selective-fast-first5
```
If it is anything else, stop and ask. Line numbers are from tip `9c4ffc3` and may have shifted — locate code by symbol name (`_recover_entities`, `RECOVERY_MAX_NEW_TOKENS`, `_filter_ner_proposals`, `_review_entities`), not by line.

**The one invariant that must not change:** the LLM never emits offsets. It returns `text` + `occurrence`, and `find_occurrence` reconstructs positions host-side against the original document. Every span must still satisfy `original_text[start:end] == entity["text"]`. If a change would let the model produce offsets directly, do not make it.

**Costs real money.** Each full 100-document run is ~$1–2 on OpenRouter. Run the 3-document smoke test first (`infer --documents 1 2 3`, ~$0.05) and confirm it works before launching the full run. Credential comes from `OPENROUTER_API_KEY` in the environment — never put it in YAML or commit it.

## Context

`runs/full100-v1/outputs` scored **29.1764** against a top team at 49.

| Component | Score | Weight |
|---|---|---|
| text_score (`1 − WER`) | 33.07 | 0.3 |
| assertions_score | 36.23 | 0.3 |
| candidates_score | 20.97 | **0.4** |

The official metric makes two things decisive:

1. **`J_X = 1` when gold and prediction are both empty.** Symptoms, test names, and lab results have empty gold candidates — so every correctly-extracted symptom scores a full 1.0 on *both* Jaccard metrics with no linking work at all.
2. **Every metric is gated on span + type.** A gold concept we never predict scores 0 on assertions *and* candidates. A right-text/wrong-type span is counted twice and scores 0 on all three.

Measured against both organizer examples, **we extract about 40% of the expected entities**:

| Source | entities / 1000 chars |
|---|---|
| Organizer example 1 (medication list) | 34.3 |
| Organizer example 2 (`sample_output.json`) | 35.0 |
| GLiNER baseline (no LLM) | 17.0 |
| **Submitted run** | **13.7** |

29 of 100 documents fall below 10 per 1000. `input/100.txt` (1,293 chars) yielded **5 entities**, missing `cục máu đông`, `đi tiêu ra máu`, `đại tiện ra máu đỏ tươi`, `mang thai 22 tuần`, `lo lắng`.

An `assertions_score` of 36.23 is therefore roughly the fraction of concepts we find at all — not a statement about assertion logic. **Recall is the bottleneck and it multiplies into all three components.**

### Decisions taken

No local dev set — the leaderboard is the feedback channel. First submission changes **extraction only**, one variable, so the recall hypothesis gets a clean read. Model stays `qwen/qwen3.5-9b`.

Branch: `review/selective-fast-first5`. Config: `configs/openrouter_full100.local.yaml`.

**Line numbers below are from the current tip (`9c4ffc3`) and may shift — locate by symbol name.**

---

## Submission 1 — recall only

The architecture is backwards for this corpus: an **English biomedical GLiNER** is the primary recall source on Vietnamese prose, and the LLM only prunes. Qwen reads this text far better. Four changes, all in extraction:

**1. Promote `_recover_entities` into a primary extraction pass** ([pipeline.py](src/clinical_nlp/pipeline.py))
- `reasoning_enabled=False` → `True` (~line 1189). This is the one call still running without reasoning; it is now the most important reasoning task in the pipeline.
- `RECOVERY_MAX_NEW_TOKENS` 1536 → 4096 (line 39). With `reasoning_max_tokens: 1024` and the backend's `min(cfg, completion // 2)` cap, reasoning stays at 1024 and ~3072 remains for JSON — this keeps the OpenRouter budget-burn guard the README documents.
- Reframe the prompt from *"recover entities missing from the supplied list"* to an **exhaustive independent extraction** of the chunk. Keep passing `EXISTING_ENTITIES`, but as a de-duplication aid rather than the anchor — the current framing biases the model toward finding only a handful of extras.
- Keep offset reconstruction exactly as-is: the model still emits `text` + `occurrence` and never offsets, and `find_occurrence` maps back host-side. This is the invariant that must not change.

**2. Lower `ner_threshold` 0.35 → 0.20** in `configs/openrouter_full100.local.yaml`. Let review handle precision.

**3. Loosen `_filter_ner_proposals`.** The hardcoded Vietnamese denylist drops generic terms (`dấu hiệu`, `triệu chứng`, `xét nghiệm`). Keep the entries that block genuine non-entities; drop the ones that block plausible clinical mentions.

**4. Rebalance the review keep-bar** in the `_review_entities` system prompt. Under this metric a missed concept costs all three components, while a kept-but-imperfect one costs less. Bias toward keeping any plausible clinical mention — and state explicitly that **assigning the wrong type is worse than keeping an uncertain span**, since type errors are double-counted at zero.

Do **not** touch candidates, assertions, or medication spans in this run.

### Before submitting

```bash
python3 -c "
import json,glob,os
tot_e=tot_c=0
for f in glob.glob('runs/<run_id>/outputs/*.json'):
    d=os.path.basename(f)[:-5]
    tot_c+=len(open(f'input/{d}.txt',encoding='utf-8').read())
    tot_e+=len(json.load(open(f)))
print(f'{tot_e} entities, {1000*tot_e/tot_c:.1f} per 1000 chars (was 13.7, organizer ~34)')
"
```

Target ~25/1000. Below 18 means the change did not take effect — investigate rather than spend a submission. Above 40 suggests runaway false positives.

Also diff against `runs/full100-v1` to eyeball what newly appeared before submitting.

---

## Submission 2 — candidate coverage

- **Never emit an empty candidate list for a genuine `CHẨN_ĐOÁN`/`THUỐC`.** 386 of 705 diagnoses and 119 of 285 drugs currently emit nothing, which is a guaranteed 0 when gold has codes. Retrieval frequently *has* the answer and the reranker discards it — `'bệnh dại'` retrieves `A82.1:0.61` and returns nothing. Fall back to the top retrieval hit.
  Apply **only** to spans that survive as real diagnoses: under `J = 0 when gt empty but pred non-empty`, coding a junk span like `'bệnh'` is now actively harmful.
- **Validate LLM-proposed ICD codes by catalog membership instead of provenance.** `_batch_rerank` raises `"invented a terminology candidate ID"` (~line 964) for any code outside the retrieved set. Retrieval genuinely fails on `bệnh Kawasaki` (returns measles and Chagas) and `amyloidosis` (returns nothing), yet the correct codes are all present — verified via `ICDIndex.contains()`: `M30.3`, `E85`, `I10`, `A82`, `C80` all `True`. Accept a proposed code when `contains()` is true.
  Keep the strict guard for RxNorm, where opaque numeric identifiers invite hallucination.

## Submission 3 — medication spans and SCD codes

Scoped deliberately: only **70 dose occurrences across 34 files** (11 files have route/frequency), so this affects ~25% of drugs, not all of them.

The organizer writes `"amlodipine 10 mg po daily"` as one span and codes it `308135` (SCD), while we emit `'aspirin'` → `1191` (ingredient). `COMPONENT_RE` and `STOP_RE` in [entity_finding/rules.py](src/clinical_nlp/entity_finding/rules.py) already model sig capture and indication stopping correctly — the rule simply loses to bare GLiNER spans during merge. Raise structured-medication priority where a sig is present, then query RxNorm with the full string so SCD codes become reachable.

## Submission 4 — assertions

Do **not** blanket-restore the baseline's 239. Over-asserting now costs a full point per concept. Target section-driven cases only — the organizer marks every drug under a pre-admission medication list `isHistorical`, which is exactly what `AssertionDetector`'s section logic is built for. Also fix `FAMILY_SUBJECT_RE`: zero `isFamily` across 100 documents is wrong when 53 files contain family/history cues.

## Explicitly not doing

**Emitting 2–3 candidates per drug.** The organizer shows exactly one RxNorm code per medication; against a gold of size 1, emitting 3 scores 1/3. This may still apply to diagnoses (`K21.0` + `K21.9`) but cannot be settled without measurement.

---

## Free validation (no labeling required)

Both organizer examples are gold. Add them as regression fixtures alongside the existing `tests/integration/test_golden_medications.py`, and write a scorer implementing the official metric — WER, both Jaccards with the empty-set rules, and the type-error double-penalty. Scoring two documents is a weak signal, but it costs nothing, it validates the metric implementation itself, and it is the only way to check the medication convention in Submission 3 before spending a submission on it.

## Verification

1. `pytest -q` green.
2. `clinical-nlp validate --run-id <id>` passes — re-checks every span against the original text for offset integrity, non-overlap, and exact substring match.
3. 100 files in `outputs/`.
4. Density check above; compare `quality_summary.json` type counts against `runs/full100-v1`.
5. Spot-check `input/100.txt` specifically — it produced 5 entities and should now produce noticeably more.

## Caveats

- **Both organizer examples are dense clinical lists**, not prose Q&A. True density for these documents is likely below 34/1000, so "40% of gold" is a well-supported direction but not a measured magnitude. If Submission 1 moves the score only slightly, the recall hypothesis is weaker than it looks and the remaining budget should shift to candidates.
- **Cost rises.** Reasoning-enabled recovery at 4096 tokens plus more entities to review is roughly 3–4× the current $0.30, so budget ~$1–2 per full run. Still negligible; the smoke test on 3 documents still applies before each full run.
- **The organizer's labeling is inconsistent** — `"sốt đau"` is one symptom while `"lo âu mất ngủ"` is two. Treat the examples as convention hints, not specification.
- **30 of 100 inputs have masked drug names** (`************`), an inherent recall ceiling on `THUỐC`.
- WER's exact computation is still unstated, so text_score predictions remain the least reliable of the three.
