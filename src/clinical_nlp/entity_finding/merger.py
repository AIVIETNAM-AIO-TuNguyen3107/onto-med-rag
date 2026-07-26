from __future__ import annotations

import bisect
from collections import defaultdict

from clinical_nlp.schemas import SpanProposal


SOURCE_PRIORITY = {
    "structured_medication_rule": 100,
    "structured_lab_rule": 100,
    "structured_medication_sig": 90,
    "exact_icd_dictionary": 80,
    "rule_dictionary": 75,
    "llm_recovery": 65,
    "gliner": 55,
    "hf_token_classifier": 55,
    "fuzzy_dictionary": 40,
}


DEFAULT_SOURCE_PRIORITY = 30


def _selection_weight(row: SpanProposal) -> float:
    """Score a span for overlap resolution.

    The ``1.0 +`` term is a per-entity bonus: two comparable spans outweigh one
    long span covering both, because the task counts entities rather than
    characters. Scaling by source priority keeps that from shredding a
    high-confidence structured span into low-confidence fragments -- a pair of
    fuzzy hits has to be genuinely plausible before it displaces a rule match.
    """
    priority = SOURCE_PRIORITY.get(row.source, DEFAULT_SOURCE_PRIORITY)
    return (1.0 + row.score) * (priority / 100.0)


def _select_non_overlapping(rows: list[SpanProposal]) -> list[SpanProposal]:
    """Choose the highest-weight flat subset via weighted interval scheduling."""
    if not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (row.end, row.start, row.type.value, row.source),
    )
    ends = [row.end for row in ordered]
    weights = [_selection_weight(row) for row in ordered]

    # best[i] is the optimal total weight over the first i spans.
    best = [0.0] * (len(ordered) + 1)
    taken = [False] * len(ordered)
    for index, row in enumerate(ordered):
        prior = bisect.bisect_right(ends, row.start, 0, index)
        with_row = weights[index] + best[prior]
        if with_row > best[index]:
            best[index + 1] = with_row
            taken[index] = True
        else:
            best[index + 1] = best[index]

    selected: list[SpanProposal] = []
    index = len(ordered)
    while index > 0:
        if taken[index - 1]:
            row = ordered[index - 1]
            selected.append(row)
            index = bisect.bisect_right(ends, row.start, 0, index - 1)
        else:
            index -= 1
    selected.reverse()
    return selected


def merge_proposals(proposals: list[SpanProposal]) -> list[SpanProposal]:
    span_evidence: dict[tuple[int, int], list[SpanProposal]] = defaultdict(list)
    for proposal in proposals:
        span_evidence[(proposal.start, proposal.end)].append(proposal)

    grouped: dict[tuple[int, int, str], list[SpanProposal]] = defaultdict(list)
    for proposal in proposals:
        grouped[(proposal.start, proposal.end, proposal.type.value)].append(proposal)

    combined: list[SpanProposal] = []
    for rows in grouped.values():
        winner = max(
            rows,
            key=lambda row: (
                SOURCE_PRIORITY.get(row.source, 0),
                row.score,
                row.end - row.start,
            ),
        )
        same_span = span_evidence[(winner.start, winner.end)]
        evidence = dict(winner.evidence)
        evidence["sources"] = sorted({row.source for row in rows})
        evidence["supporting_sources"] = sorted(
            {row.source for row in same_span}
        )
        evidence["alternative_types"] = sorted(
            {row.type.value for row in same_span}
        )
        evidence["type_conflict"] = len(evidence["alternative_types"]) > 1
        combined.append(winner.model_copy(update={"evidence": evidence}))

    selected = _select_non_overlapping(combined)
    return sorted(selected, key=lambda row: (row.start, row.end))
