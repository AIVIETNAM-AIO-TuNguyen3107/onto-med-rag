from __future__ import annotations

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


def _overlaps(left: SpanProposal, right: SpanProposal) -> bool:
    return left.start < right.end and right.start < left.end


def merge_proposals(proposals: list[SpanProposal]) -> list[SpanProposal]:
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
        evidence = dict(winner.evidence)
        evidence["sources"] = sorted({row.source for row in rows})
        combined.append(winner.model_copy(update={"evidence": evidence}))

    ordered = sorted(
        combined,
        key=lambda row: (
            -SOURCE_PRIORITY.get(row.source, 0),
            -row.score,
            -(row.end - row.start),
            row.start,
        ),
    )
    selected: list[SpanProposal] = []
    for proposal in ordered:
        if any(_overlaps(proposal, existing) for existing in selected):
            continue
        selected.append(proposal)
    return sorted(selected, key=lambda row: (row.start, row.end))

