# ViettelAIrace handoff archive — 2026-07-28

This is the team handoff for the Phase-1 experiment history. It separates
confirmed scores from pending and status-unknown packages so that a ZIP's
presence is never mistaken for evidence that it was submitted.

## Start here

1. Read `00_STATUS/PROJECT_STATUS_2026-07-28.md`.
2. Use `01_BEST_SCORED/direct-extraction-v1.zip` as the best confirmed result.
3. Submit `03_PENDING_SUBMISSION/direct-extraction-v3-force-keep.zip` unchanged
   after the quota resets.
4. Verify any copied/downloaded file against `00_STATUS/SHA256SUMS`.

## Folder meanings

- `00_STATUS`: human status, score ledger, complete run inventory, and checksums.
- `01_BEST_SCORED`: current best confirmed leaderboard artifact.
- `02_SCORED_HISTORY`: lower-scoring historical submissions.
- `03_PENDING_SUBMISSION`: validated artifact that has not yet been submitted.
- `04_PACKAGED_OTHER`: packages that were not submitted or whose portal status
  is not recoverable from local evidence.
- `05_REPRODUCIBILITY`: raw direct annotations, audits, correction overlay,
  verification scripts, manifests, and local RxNorm catalogs.

`subA-full100-reconstructed.zip` was reconstructed from the preserved 100-file
output directory. It is not claimed to be byte-identical to the original
uploaded ZIP, which is no longer present.

The Luna packages are labelled `submission_unknown`, not `unsubmitted`, because
the local repo and conversations contain no reliable portal evidence either
way.

## Access and redistribution

The archive is intended for an access-controlled team Drive folder. Competition
inputs are not included. The local RxNorm files are included only for internal
reproducibility; their original crawl provenance and redistribution terms were
not recorded, so do not republish them without checking the applicable terms.
