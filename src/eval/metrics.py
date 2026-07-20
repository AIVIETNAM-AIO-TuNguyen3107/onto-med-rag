"""Competition scoring metrics (local evaluation)."""

from __future__ import annotations

from dataclasses import dataclass


def _wer(ref_words: list[str], hyp_words: list[str]) -> float:
    """Word-level WER via Levenshtein on token lists."""
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    if not hyp_words:
        return 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / n


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if (not a and b) or (a and not b):
        return 0.0
    return len(a & b) / len(a | b)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    if end <= start:
        return 0.0
    inter = end - start
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def _match_entities(
    gt: list[dict], pred: list[dict], min_iou: float = 0.5
) -> list[tuple[dict | None, dict | None]]:
    """Greedy match by position IoU then type agreement."""
    unmatched_gt = list(range(len(gt)))
    unmatched_pred = list(range(len(pred)))
    pairs: list[tuple[dict | None, dict | None]] = []

    while unmatched_gt and unmatched_pred:
        best = (-1.0, -1, -1)
        for gi in unmatched_gt:
            for pi in unmatched_pred:
                iou = _overlap(tuple(gt[gi]["position"]), tuple(pred[pi]["position"]))
                if iou > best[0]:
                    best = (iou, gi, pi)
        if best[0] < min_iou:
            break
        _, gi, pi = best
        pairs.append((gt[gi], pred[pi]))
        unmatched_gt.remove(gi)
        unmatched_pred.remove(pi)

    for gi in unmatched_gt:
        pairs.append((gt[gi], None))
    for pi in unmatched_pred:
        pairs.append((None, pred[pi]))
    return pairs


@dataclass
class SampleScore:
    text: float
    assertions: float
    candidates: float
    final: float


def score_sample(gt: list[dict], pred: list[dict]) -> SampleScore:
    pairs = _match_entities(gt, pred)

    ref_texts: list[str] = []
    hyp_texts: list[str] = []
    j_assertions: list[float] = []
    j_candidates: list[float] = []
    cand_weights: list[int] = []

    for g, p in pairs:
        if g is None or p is None:
            if g is not None:
                ref_texts.append(g["text"])
                hyp_texts.append("")
                j_assertions.append(0.0)
                j_candidates.append(0.0)
                cand_weights.append(len(g.get("candidates", [])) + 1)
            elif p is not None:
                ref_texts.append("")
                hyp_texts.append(p["text"])
                j_assertions.append(0.0)
                j_candidates.append(0.0)
                cand_weights.append(1)
            continue

        if g["type"] != p["type"]:
            ref_texts.extend([g["text"], g["text"]])
            hyp_texts.extend([p["text"], ""])
            j_assertions.extend([0.0, 0.0])
            j_candidates.extend([0.0, 0.0])
            cand_weights.extend(
                [len(g.get("candidates", [])) + 1, len(g.get("candidates", [])) + 1]
            )
            continue

        ref_texts.append(g["text"])
        hyp_texts.append(p["text"])
        j_assertions.append(_jaccard(set(g.get("assertions", [])), set(p.get("assertions", []))))
        j_candidates.append(_jaccard(set(g.get("candidates", [])), set(p.get("candidates", []))))
        cand_weights.append(len(g.get("candidates", [])) + 1)

    ref_words = " ".join(ref_texts).split()
    hyp_words = " ".join(hyp_texts).split()
    text_score = 1.0 - _wer(ref_words, hyp_words)
    assertions_score = sum(j_assertions) / len(j_assertions) if j_assertions else 1.0

    weight_sum = sum(cand_weights)
    if weight_sum:
        candidates_score = sum(j * w for j, w in zip(j_candidates, cand_weights)) / weight_sum
    else:
        candidates_score = 1.0

    final = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
    return SampleScore(text_score, assertions_score, candidates_score, final)


def score_dataset(pairs: list[tuple[list[dict], list[dict]]]) -> dict[str, float]:
    if not pairs:
        return {"text": 0.0, "assertions": 0.0, "candidates": 0.0, "final": 0.0}

    scores = [score_sample(gt, pred) for gt, pred in pairs]
    n = len(scores)
    return {
        "text": sum(s.text for s in scores) / n,
        "assertions": sum(s.assertions for s in scores) / n,
        "candidates": sum(s.candidates for s in scores) / n,
        "final": sum(s.final for s in scores) / n,
    }
