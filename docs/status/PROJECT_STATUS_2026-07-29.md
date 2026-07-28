# ViettelAIrace project status — 2026-07-29

## Executive status

The best confirmed leaderboard submission remains **Direct V1 at 35.7681**.
Direct V2 strict scored **30.8855** and Direct V3 force-keep scored **33.7390**.
Both are confirmed regressions, not unknown results.

Direct V3 was submitted unchanged after the quota reset:

- Path: `artifacts/direct-extraction-v3-force-keep.zip`
- SHA-256: `907b7150c46fce4b43bef64cb36f2875e26ce15d2c53581436f457fed56c4c7c`
- Size: 89,549 bytes
- Members: `1.json` through `100.json`, with no containing directory
- Status: `scored`
- WER: `60.5734` (`num_scored=100`)
- J_assertion: `46.3932` (`num_records=100`)
- J_candidates: `19.9827`
- Final: `33.7390`

The submitted archive retains its recorded checksum. Direct V1 remains the
canonical best because V3 scored 2.0291 points lower.

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
| Direct V3 force-keep | 60.5734 | 39.4266 | 46.3932 | 19.9827 | 33.7390 | Competition portal result reported by the user, 2026-07-29 |

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

Direct V3 was the clean follow-up to V2. It kept V2's reviewed corrections,
strict field omission, and 806 linked rows, but restored all 324 rows V2
dropped. Compared with V2:

- WER recovered by 2.5361 points and exactly matched V1.
- J_assertion recovered by 3.0843 points and exactly matched V1.
- J_candidates improved by 2.9184 points but remained 5.0727 below V1.
- Final improved by 2.8535 points but remained 2.0291 below V1.

This is strong evidence that dropping the 324 unresolved rows caused V2's text
and assertion regressions. Retaining them fixed those components. The remaining
loss versus V1 is entirely in candidate scoring, so V2/V3's force-top candidate
bundle and corrections should not replace V1. V1 → V3 still combines candidate
and correction changes, so this leaderboard result cannot attribute the
candidate loss to one individual change.

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
| Direct V3 force-keep | 2,801 | 487 | 806 | force top; retain 324 no-hit rows | 33.7390 |

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
- Direct V3 force-keep: 33.7390. Restoring the 324 dropped rows recovered WER
  and J_assertion to V1 exactly, but candidate scoring remained below V1.

### Complete or packaged without a confirmed score

- `baseline-gliner-v1`: complete 100-document GLiNER probe; packaged, deliberately
  not submitted after review.
- `direct-extraction-offline-probe`: complete local comparison; 252 linked rows,
  not a leaderboard submission.
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
> candidates was harmful overall. Direct V3 was then submitted unchanged and
> scored 33.7390. Restoring V2's 324 dropped entities recovered WER and
> J_assertion exactly to V1 levels, but J_candidates remained lower at 19.9827.
> Direct V1 therefore remains the best submission. The current implementation
> and tests are being handed off on
> `handoff/direct-extraction-20260728`; 185 tests and the dependency lock check
> pass. Older GLiNER/Qwen, recall, and Luna experiments are recorded in the
> experiment ledger.

## Immediate next actions

1. Retain Direct V1 as `BEST_SCORED`; keep V2 and V3 in scored history.
2. Preserve the portal screenshot/reference for V3 alongside the reported
   metrics if one is available.
3. Isolate candidate-policy changes locally before spending another submission:
   start from V1 spans/assertions and change only one linking component.
4. Do not use V2/V3's force-top candidate bundle as the new default.
