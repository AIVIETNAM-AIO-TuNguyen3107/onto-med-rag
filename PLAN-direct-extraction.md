# Direct extraction by a strong model, then deterministic linking

## Handoff notes — this plan is the brief for a fresh session

Repo: `/Users/questionminded/coding/ViettelAIrace`.
Current branch: **`experiment/leaderboard-last4-20260727`** at `a8f5dae`. Test suite: **147 passing**.

Start this in a **new session**. The extraction pass needs context headroom (~92K input tokens of source text, ~119K tokens of emitted entities) and the originating session is long. Everything needed is written below — do not assume prior context.

**Read `REVIEW-scoring-improvements.md` for the metric derivation.** The rest of the calibration is in this file.

**One thing must be understood before starting: a stronger model does not reveal the ground truth.** It applies better judgement to the same two organizer examples. Every convention below is inferred, not known. The last confident intervention on this project lost 6.29 points.

---

## Kickoff prompt for the new session

Copy this plan into the repo first so the path is stable and the name is meaningful:

```sh
cp /Users/questionminded/.claude/plans/im-joining-a-ai-nifty-nova.md \
   /Users/questionminded/coding/ViettelAIrace/PLAN-direct-extraction.md
```

Then open a new session in `/Users/questionminded/coding/ViettelAIrace` and paste:

> Read `PLAN-direct-extraction.md` in full, then `data/examples/sample_output.json`, `data/examples/sample_input.txt`, and the medication-list example quoted in `REVIEW-scoring-improvements.md`. Do not re-derive the analysis — the metric, the four scored submissions, and the conventions are already settled in those files.
>
> Your job is the extraction pass described in that plan: read all 100 files in `input/` and emit spans, types, and assertions yourself, in batches of 10 documents, writing raw batches to `artifacts/direct-extraction/raw/`. You are replacing GLiNER and Qwen for the language work only — the existing deterministic linker fills ICD/RxNorm codes afterwards. Do not hand-write candidate codes and do not run `clinical-nlp infer`.
>
> Emit `text` + 1-based `occurrence`, never character offsets. Offsets get reconstructed host-side with `find_occurrence` from `src/clinical_nlp/text.py`, and every span must satisfy `original_text[start:end] == entity["text"]`.
>
> Calibration from scored runs: aim for roughly 2,790 entities and ~250 assertions in total. 4,132 entities and 744 assertions scored materially worse. Fewer and more precise beats more.
>
> Start with documents 1–10. Before each batch, re-read the conventions section of the plan rather than recalling it — drift across batches is the main risk. Show me the first batch before continuing so I can check the conventions are right.

That last sentence matters: a convention error caught at batch 1 costs one turn, caught at batch 10 costs all ten.

## Context

`final_score = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score`, where `text_score = mean(1 − WER)` over the 100 documents and the other two are per-concept Jaccard.

Four scored submissions:

| run | entities | assertions | coded | WER | J_assert | J_cand | **score** |
|---|---|---|---|---|---|---|---|
| `full100-v1` | 2790 | 79 | 485 | 66.93 | 36.23 | 20.97 | 29.18 |
| sub-1 candidates | 2790 | 79 | 527 | 66.93 | 36.23 | 20.21 | 28.87 |
| **sub-2 assertions** | 2790 | 256 | 485 | 66.93 | **37.64** | 20.97 | **29.60** |
| `subA-full100` | 4132 | 744 | 984 | 73.05 | 29.12 | 15.16 | 22.88 |

Hard-won lessons, all paid for:

1. **Adding candidates hurts.** +42 coded → −0.76. `J = 0` when gold is empty and prediction is not, so a wrong code is worse than no code.
2. **Assertions peak between 256 and 744.** More is not better.
3. **More entities made WER worse** (2,790 → 4,132 moved WER 66.93 → 73.05). Fewer has never been tried.
4. **Blanket rules lose; audited overlays win.** Three blanket changes together: −6.29. One curated overlay: +0.42.

### Why direct extraction is the remaining lever

`WER 66.93` has never moved except downward. All three components are gated on span+type matching gold — a concept never predicted scores zero on assertions *and* candidates. Yet spans are currently driven **57%** by `Ihor/gliner-biomed-large-v1.0`, an **English biomedical** model reading Vietnamese prose, with `Qwen3.5-9B` reviewing. Overlay tuning cannot fix spans; the narrowed alias work reaches only ~19 rows.

The corpus is small enough to process directly: **203,817 chars across 100 files**, median 1,845, largest 4,481.

| batching | turns | per turn |
|---|---|---|
| 10 docs | 10 | ~9,300 in / ~11,900 out |

**Cost is session quota, not OpenRouter credits.** Ten turns of substantial output. That is the real price of this approach.

---

## Division of labour

**The model does language work. The pipeline does catalog work.**

| | who | why |
|---|---|---|
| spans, types, assertions | **this model** | convention matching and Vietnamese clinical judgement |
| ICD / RxNorm codes | **existing pipeline** | retrieval against a 15,843-concept catalog; hand-generating 886 codes is error-prone and slow |

Do **not** hand-write candidate codes. Emit spans and let the deterministic linker fill them.

## Conventions, inferred from the two organizer examples

Both are in the repo: `data/examples/sample_input.txt` + `sample_output.json`, and the medication-list example quoted in `REVIEW-scoring-improvements.md`. **Read both before extracting.**

- **Medication spans carry the full sig** — `"amlodipine 10 mg po daily"` is one entity — but **stop before the indication**: `điều trị ho` becomes a separate `TRIỆU_CHỨNG` (`"ho"`).
- **Lab name and value are separate paired entities**: `WBC` (`TÊN_XÉT_NGHIỆM`) then `14,43` (`KẾT_QUẢ_XÉT_NGHIỆM`).
- **Granularity is fine-grained**: `lo âu mất ngủ` splits into `lo âu` + `mất ngủ`. But the same example keeps `sốt đau` as one span — the organizer's own labelling is inconsistent, so do not over-fit.
- **Every drug under a pre-admission heading gets `isHistorical`.** Section headings drive assertions.
- Assertions apply only to `CHẨN_ĐOÁN`, `THUỐC`, `TRIỆU_CHỨNG`; max 3.
- **Masked drug names (`************`) are not entities.** 30 of 100 files contain them.
- Section headings (`Lý do nhập viện`, `Bệnh sử hiện tại`) are **not** entities.

Calibration targets from the scored data: **around 2,790 entities total** (that base scored best), and **roughly 250 assertions** — sub-2's 256 outscored both 79 and 744.

## Execution

**Batch 10 documents per turn**, in numeric order. For each document emit:

```json
{"document_id": "7", "entities": [
  {"text": "<exact substring>", "occurrence": 1, "type": "TRIỆU_CHỨNG", "assertions": []}
]}
```

**Never emit character offsets.** Emit `text` + `occurrence` (1-based, which occurrence of that exact string in the document). Offsets are reconstructed host-side — this invariant is what has kept every previous run valid.

Write each batch to `artifacts/direct-extraction/raw/<batch>.json`, then run a converter script that:

1. resolves offsets with `find_occurrence` / `find_occurrence_relaxed` from [src/clinical_nlp/text.py](src/clinical_nlp/text.py);
2. drops any span that is not an exact substring, recording it in an audit file rather than guessing;
3. sorts by start offset and drops overlaps (keep the longer span);
4. asserts `original_text[start:end] == entity["text"]`;
5. writes `artifacts/direct-extraction/outputs/<id>.json` in competition shape via `Entity.output_dict()` from [src/clinical_nlp/schemas.py](src/clinical_nlp/schemas.py).

Then link candidates over those spans with the deterministic linker — `ICDIndex.retrieve` plus `_eligible_candidates` / `_automatic_candidate_selection` thresholds from [src/clinical_nlp/pipeline.py](src/clinical_nlp/pipeline.py), no LLM. **Emit a code only when auto-selection returns one; otherwise `[]`.** Given lesson 1, do not add a top-hit fallback.

Finally package with the existing `package` subcommand in [src/clinical_nlp/submission_variants.py](src/clinical_nlp/submission_variants.py) (`--expected-documents 100`).

## Verification

1. Exactly 100 output files; every stem matches an input stem.
2. `clinical-nlp validate --run-id <id>` passes — offsets, non-overlap, exact substring.
3. Entity count within roughly ±20% of 2,790. Materially above that repeats the `subA-full100` failure.
4. Assertion count near 250, not 750.
5. No entity whose text is entirely asterisks; no section headings among `CHẨN_ĐOÁN`.
6. Spot-check documents 2, 17, 24 against the conventions above before packaging.
7. Diff entity counts per document against `runs/full100-v1/outputs/` and read the ten largest divergences by hand.

## Caveats

- **This is still a guess at the conventions.** Better judgement, same two examples. It could score worse.
- **Not reproducible.** A session-driven extraction cannot be re-run deterministically the way the pipeline can. Keep the raw batch files so at least the conversion is repeatable.
- **Consistency across 10 turns is a real risk** — a convention applied in batch 1 may drift by batch 10. Re-read this section at the start of every batch, and prefer re-reading over recalling.
- **The 0.4-weight candidates component is barely addressed here.** Better spans feed the linker better inputs, but `Bệnh Kawasaki` still fails retrieval until the bracketed-alias work lands (`M30.3` is stored as `'Hội chứng hạch bạch huyết niêm mạc [Kawasaki]'`; 448 safe single-code aliases exist behind brackets). That remains worth doing afterwards, as a separate submission.
- **Keep `runs/full100-v1` and sub-2 intact.** Sub-2 at 29.60 is the fallback if this scores worse.
