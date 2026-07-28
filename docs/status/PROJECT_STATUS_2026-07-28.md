# ViettelAIrace project status — 2026-07-28

## Executive status

The best confirmed leaderboard submission is **Direct V1 at 35.7681**.
Direct V2 strict was submitted and scored **30.8855**, so it is a confirmed
regression rather than an unknown result. Direct V3 force-keep is finished,
independently validated, and packaged, but was not submitted because the daily
submission quota was exhausted.

The next controlled test is the existing Direct V3 ZIP, unchanged:

- Path: `artifacts/direct-extraction-v3-force-keep.zip`
- SHA-256: `907b7150c46fce4b43bef64cb36f2875e26ce15d2c53581436f457fed56c4c7c`
- Size: 89,549 bytes
- Members: `1.json` through `100.json`, with no containing directory
- Status: `pending_submission`

Do not rebuild this archive before submission. Submit the recorded bytes and
copy the complete metric block into the experiment ledger. Direct V1 remains the
canonical best unless V3 exceeds 35.7681.

## Confirmed leaderboard results

The final score is:

`0.3 × (100 − WER) + 0.3 × J_assertion + 0.4 × J_candidates`

| Submission | WER ↓ | Text score ↑ | J_assertion ↑ | J_candidates ↑ | Final ↑ | Evidence |
|---|---:|---:|---:|---:|---:|---|
| `full100-v1` | 66.9345 | 33.0655 | 36.2262 | 20.9721 | 29.1764 | Claude session `24dbd004…`, 2026-07-26 06:05Z |
| `subA-full100` | 73.0459 | 26.9541 | 29.1166 | 15.1564 | 22.8838 | Claude session `24dbd004…`, 2026-07-27 02:44Z |
| audited candidates | 66.9345 | 33.0655 | 36.2262 | 20.2110 | 28.8719 | Claude session `24dbd004…`, 2026-07-27 08:09Z |
| audited assertions | 66.9345 | 33.0655 | 37.6352 | 20.9721 | 29.5991 | Claude session `24dbd004…`, 2026-07-27 08:09Z |
| **Direct V1** | **60.5734** | **39.4266** | **46.3932** | **25.0554** | **35.7681** | Claude session `dd69f5a0…`, 2026-07-28 09:21Z |
| Direct V2 strict | 63.1095 | 36.8905 | 43.3089 | 17.0643 | 30.8855 | Repo-agent transcript, 2026-07-28 10:58Z |

The archived Claude transcript ends after Direct V1. The Direct V2 metric block
was recorded immediately after its implementation in the July 28 repo-agent
transcript, which is why a Claude-only search misses it. The score arithmetic
reproduces 30.8855 exactly.

## What the results mean

Direct V1 improved every leaderboard component over the previous best:

- WER: 66.9345 → 60.5734
- J_assertion: 37.6352 → 46.3932
- J_candidates: 20.9721 → 25.0554
- Final: 29.5991 → 35.7681

Direct V2 changed several things together: it applied reviewed corrections,
forced the top retrieved code for 529 weakly linked rows, omitted `candidates`
on non-linkable types, and dropped 324 diagnosis/medication rows for which the
linker retrieved no code. Compared with Direct V1:

- WER became 2.5361 points worse.
- J_assertion fell 3.0843 points.
- J_candidates fell 7.9911 points.
- Final score fell 4.8826 points.

This establishes that the V2 bundle was harmful, but does not isolate which
change caused how much of the loss.

Direct V3 is the clean follow-up to V2. It keeps V2's reviewed corrections,
strict field omission, and 806 linked rows, but restores all 324 rows V2
dropped. Thus V2 → V3 isolates the effect of retaining unresolved linkable
entities. V1 → V3 still combines multiple changes and should not be described
as a single-variable comparison.

## Current pipeline

### Earlier GLiNER/Qwen pipeline

```text
100 input documents
  → GLiNER proposals + deterministic rule proposals
  → merge and span validation
  → Qwen selective review / miss recovery / reranking
  → assertion rules and review
  → ICD-10 and RxNorm retrieval/linking
  → output validation
  → 100-file ZIP
```

This family produced `full100-v1`, the `subA` recall run, controlled candidate
and assertion overlays, OpenRouter/Qwen canaries, and the later Luna variants.
It improved reproducibility, caching, concurrency, resume behavior, and
validation, but the English biomedical GLiNER model remained a weak fit for
Vietnamese clinical prose.

### Current direct-extraction pipeline

```text
Claude-authored text + occurrence + type + assertion batches
  → Unicode/source-slice canonicalization
  → exact occurrence-to-offset reconstruction with find_occurrence
  → optional reviewed correction overlay
  → deterministic ICD-10 + local RxNorm retrieval
  → candidate policy (conservative, top-or-drop, or top-or-empty)
  → strict schema and source-slice validation
  → 100-file ZIP + SHA-256 manifest
```

No runtime LLM is used during conversion or terminology linking. The language
annotations in the ten raw batches were nevertheless authored during a Claude
session; saying the end-to-end result is "non-LLM" would therefore be
misleading.

The three direct variants are:

| Variant | Entities | Assertions | Linked entities | Policy | Result |
|---|---:|---:|---:|---|---|
| Direct V1 | 2,801 | 487 | 277 | conservative linking, original serialization | **35.7681** |
| Direct V2 strict | 2,477 | 428 | 806 | force top; drop 324 no-hit rows | 30.8855 |
| Direct V3 force-keep | 2,801 | 487 | 806 | force top; retain 324 no-hit rows | pending |

## Trial history

### Scored

- `full100-v1`: first complete selective GLiNER/Qwen run; 29.1764.
- `subA-full100`: high-recall 4,132-entity run; 22.8838. Excess recall and
  assertions were harmful.
- Audited candidate overlay: 28.8719. Adding the reviewed candidate set reduced
  J_candidates and was rejected.
- Audited assertion overlay: 29.5991. This was a small clean improvement and the
  pre-direct fallback.
- Direct V1: 35.7681. Replacing model-based extraction with the reviewed direct
  spans produced the first WER improvement and remains best.
- Direct V2 strict: 30.8855. Dropping no-hit entities plus forcing candidates
  regressed all three components.

### Complete or packaged without a confirmed score

- `baseline-gliner-v1`: complete 100-document GLiNER probe; packaged, deliberately
  not submitted after review.
- `direct-extraction-offline-probe`: complete local comparison; 252 linked rows,
  not a leaderboard submission.
- `direct-extraction-v3-force-keep`: complete, validated, packaged, pending quota.
- `luna-s3-full`: complete 100-document Luna consensus package; portal status is
  not present in the local evidence.
- `luna-s4-recovery`: complete 100-document miss-recovery package; portal status
  is not present in the local evidence.

### Smoke tests, canaries, and incomplete runs

- Four one-document GLiNER/baseline canaries.
- `full100-smoke`, three documents.
- OpenRouter/Qwen preflights, one-document selective runs, and an incomplete
  first-five run that finished two documents.
- Recall S1/S2 three-document variants and document-100 probes.
- `subA` five-document smoke variants.
- `luna-s3-smoke`, three documents.

These establish feasibility or failure modes; they are not comparable
leaderboard submissions.

### Investigated but not run

- `urchade/gliner_multi-v2.1`.
- `cbc-528a/BamiBERT-ViMedNER`; the available backend did not expose the required
  lab-result label.
- Exact-only bracketed ICD aliases. The safe aliases reached only about 19
  repeated rows, so the proposal was not implemented.
- Broad ICD aliases, rejected as ambiguous.
- Threshold-only pruning of 237 GLiNER rows, rejected because 203 had already
  been reviewed and retained.
- SCDC RxNorm crawling. The local catalog contains IN/SCD/SBD only.
- Relaxed ICD thresholds after Direct V1, abandoned because precise sparse codes
  outscored the denser baseline.
- Replacing Qwen with a stronger model in the original pipeline.
- A manually labelled development set.

## Repository state and handoff

- Base branch before handoff: `experiment/leaderboard-last4-20260727`
- Base commit: `e85b6d2fd7e8fb905ab3af74de3a399514ea0984`
- Relation to `team/main` at inspection: 27 commits ahead, 0 behind
- Test result before handoff: 185 passed
- Dependency check before handoff: `uv lock --check` passed

Generated runs, private competition inputs, submission ZIPs, local RxNorm
catalogs, credentials, Claude/Codex transcripts, and package metadata do not
belong in Git. They are archived separately in the team Drive handoff.

## Copy-ready teammate update

> Current best is Direct V1 at 35.7681. Direct V2 strict was submitted and
> scored 30.8855, so dropping unresolved entities and forcing more top
> candidates was harmful overall. Direct V3 is finished, validated, and
> packaged, but has not been submitted because today's quota was exhausted. V3
> restores the 324 entities dropped by V2 while retaining V2's 806 linked
> entities and corrections, making it the next controlled leaderboard test. The
> current implementation and tests are being handed off on
> `handoff/direct-extraction-20260728`; 185 tests and the dependency lock check
> pass. Older GLiNER/Qwen, recall, and Luna experiments are recorded in the
> experiment ledger.

## Immediate next actions

1. Submit the unchanged Direct V3 archive after the quota resets.
2. Record WER, `num_scored`, J_assertion, `num_records`, J_candidates, final
   score, submission time, and a portal screenshot/reference.
3. Promote V3 only if it exceeds 35.7681; otherwise retain Direct V1 as best.
4. Do not combine another pipeline change with the V3 submission.
