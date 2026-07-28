from __future__ import annotations

import re
from dataclasses import dataclass, field

from clinical_nlp.icd_linking import ICDIndex
from clinical_nlp.normalization import normalize_search
from clinical_nlp.schemas import EntityType, SpanProposal


DEFAULT_DRUGS = {
    "acetaminophen",
    "amlodipine",
    "aspirin",
    "atenolol",
    "clonazepam",
    "docusate",
    "docusate sodium",
    "doxycycline",
    "guaifenesin",
    "ibuprofen",
    "lisinopril",
    "metformin",
    "metoprolol",
    "metoprolol succinate",
    "nitroglycerin",
    "nystatin",
    "pravastatin",
    "senna",
    "sorafenib",
    "trimetazidine",
    "vastarel",
}

DEFAULT_SYMPTOMS = {
    "ảo giác",
    "ban đỏ",
    "buồn nôn",
    "chảy máu",
    "chảy nước mũi",
    "chóng mặt",
    "căng thẳng",
    "đánh trống ngực",
    "đau bụng",
    "đau đầu",
    "đau ngực",
    "đau nhức",
    "đau thượng vị",
    "đi ngoài phân đen",
    "đi tiêu ra máu",
    "đỏ mắt",
    "đổ mồ hôi",
    "ho",
    "ho khan",
    "ho đờm",
    "hồi hộp",
    "khó thở",
    "lo âu",
    "mất ngủ",
    "mệt mỏi",
    "mờ mắt",
    "nôn",
    "phát ban",
    "phù",
    "sốt",
    "sốt cao",
    "sốt đau",
    "sưng",
    "táo bón",
    "tiêu chảy",
    "tức ngực",
    "tim đập nhanh",
    "vàng da",
    "vàng mắt",
    "chậm phát triển trí tuệ",
    "rối loạn vận động",
}

DEFAULT_TESTS = {
    "albumin",
    "alp",
    "alt",
    "ast",
    "bilirubin",
    "bilirubin toàn phần",
    "công thức máu",
    "crp",
    "ecg",
    "glucose",
    "hba1c",
    "máu lắng",
    "men gan",
    "neut%",
    "spo2",
    "wbc",
}

FORM_WORDS = {
    "acetate",
    "besylate",
    "capsule",
    "extended",
    "hr",
    "release",
    "sodium",
    "solution",
    "succinate",
    "sulfate",
    "suspension",
    "tablet",
    "tartrate",
    "oral",
    "xl",
    "xr",
    "er",
}

COMPONENT_RE = re.compile(
    r"(?ix)^(?:"
    r"\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?|"
    r"mcg|μg|ug|mg|g|kg|ml|l|meq|unit|units|iu|%|"
    r"po|iv|im|sc|sq|sl|inh|inhaled|topical|"
    r"(?:qd|od|daily|bid|tid|qid|qhs|qam)(?::?prn)?|"
    r"q\d+h(?::?prn)?|prn|stat|"
    r"x|uống|"
    + "|".join(re.escape(word) for word in sorted(FORM_WORDS))
    + r")$"
)
TOKEN_RE = re.compile(r"\S+")
STOP_RE = re.compile(
    r"(?i)(?:\s+(?:điều trị|because of|due to|for)\b|"
    r"\s+cho\s+(?=[^\d])|\*\s+\*\d+\.|\n\s*(?:[-•]|\d+\.)|\s*[,;])"
)

NUMBER = r"[+-]?\d+(?:[.,]\d+)?"
RANGE = rf"{NUMBER}\s*[-–]\s*{NUMBER}"
COMPARATOR = r"(?:<=|>=|<|>)?"
UNIT = (
    r"(?:%|g/L|mg/L|mmol/L|µmol/L|umol/L|U/L|IU/L|"
    r"10\^?\d+/L|x10\^?\d+/L|mmHg|bpm)"
)
RESULT_RE = re.compile(
    rf"(?ix)^(?:\s*[:=]\s*|\s+)"
    rf"(?P<result>{COMPARATOR}\s*(?:{RANGE}|{NUMBER})(?:\s*{UNIT})?|"
    r"âm\s+tính|dương\s+tính|negative|positive|trace|\+{1,4}|-{1,4})"
)


def _compile_terms(terms: set[str]) -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(term) for term in sorted(terms, key=lambda value: (-len(value), value))
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


@dataclass
class RuleEntityFinder:
    icd_index: ICDIndex | None = None
    drugs: set[str] = field(default_factory=lambda: set(DEFAULT_DRUGS))
    symptoms: set[str] = field(default_factory=lambda: set(DEFAULT_SYMPTOMS))
    tests: set[str] = field(default_factory=lambda: set(DEFAULT_TESTS))

    def __post_init__(self) -> None:
        self._drug_re = _compile_terms(self.drugs)
        self._symptom_re = _compile_terms(self.symptoms)
        self._test_re = _compile_terms(self.tests)
        self._diagnosis_trie = self._build_diagnosis_trie()

    def _build_diagnosis_trie(self) -> dict:
        root: dict = {}
        if self.icd_index is None:
            return root
        symptom_keys = {normalize_search(value) for value in self.symptoms}
        for name, code in self.icd_index.all_aliases(min_length=5):
            key = normalize_search(name)
            tokens = key.split()
            if not tokens or key in symptom_keys:
                continue
            if len(tokens) == 1 and len(key) < 8:
                continue
            node = root
            for token in tokens:
                node = node.setdefault(token, {})
            node.setdefault("_values", []).append((name, code))
        return root

    def find(self, text: str) -> list[SpanProposal]:
        proposals: list[SpanProposal] = []
        proposals.extend(self._find_medications(text))
        proposals.extend(self._find_labs(text))
        proposals.extend(self._find_terms(text, self._symptom_re, EntityType.SYMPTOM, 0.82))
        proposals.extend(self._find_diagnoses(text))
        return proposals

    def _find_terms(
        self,
        text: str,
        pattern: re.Pattern[str],
        entity_type: EntityType,
        score: float,
    ) -> list[SpanProposal]:
        rows: list[SpanProposal] = []
        for match in pattern.finditer(text):
            value = normalize_search(text[match.start() : match.end()])
            tail = normalize_search(text[match.end() : match.end() + 6])
            if value == "phù" and tail.startswith("hợp"):
                continue
            rows.append(
                SpanProposal(
                start=match.start(),
                end=match.end(),
                text=text[match.start() : match.end()],
                type=entity_type,
                source="rule_dictionary",
                score=score,
            )
            )
        return rows

    def _find_medications(self, text: str) -> list[SpanProposal]:
        rows: list[SpanProposal] = []
        occupied: list[tuple[int, int]] = []
        for match in self._drug_re.finditer(text):
            start, base_end = match.span()
            end = self._expand_medication(text, base_end)
            rows.append(
                SpanProposal(
                    start=start,
                    end=end,
                    text=text[start:end],
                    type=EntityType.MEDICATION,
                    source="structured_medication_rule",
                    score=0.97,
                )
            )
            occupied.append((start, end))

        generic = re.compile(
            r"(?i)(?<!\w)([A-Za-z][A-Za-z-]{2,}"
            r"(?:\s+(?:sodium|succinate|tartrate|acetate|sulfate))?)"
            r"(?=\s+\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*"
            r"(?:mcg|mg|g|ml|%))"
        )
        for match in generic.finditer(text):
            if any(match.start() >= left and match.start() < right for left, right in occupied):
                continue
            end = self._expand_medication(text, match.end())
            rows.append(
                SpanProposal(
                    start=match.start(),
                    end=end,
                    text=text[match.start() : end],
                    type=EntityType.MEDICATION,
                    source="structured_medication_sig",
                    score=0.80,
                )
            )
        return rows

    def _expand_medication(self, text: str, cursor: int) -> int:
        candidate_end = min(len(text), cursor + 100)
        stop = STOP_RE.search(text, cursor, candidate_end)
        if stop:
            candidate_end = stop.start()
        end = cursor
        for token_match in TOKEN_RE.finditer(text, cursor, candidate_end):
            raw = token_match.group(0)
            token = raw.strip("()[]{}.,;").casefold()
            token = token.replace(":", ":")
            if COMPONENT_RE.fullmatch(token):
                end = token_match.end()
                continue
            break
        return end

    def _find_labs(self, text: str) -> list[SpanProposal]:
        rows: list[SpanProposal] = []
        for match in self._test_re.finditer(text):
            if (
                match.start() > 0
                and text[match.start() - 1] == "-"
                or match.end() < len(text)
                and text[match.end()] == "-"
            ):
                continue
            rows.append(
                SpanProposal(
                    start=match.start(),
                    end=match.end(),
                    text=text[match.start() : match.end()],
                    type=EntityType.TEST_NAME,
                    source="structured_lab_rule",
                    score=0.95,
                )
            )
            tail = text[match.end() : min(len(text), match.end() + 60)]
            result = RESULT_RE.match(tail)
            if result:
                start = match.end() + result.start("result")
                end = match.end() + result.end("result")
                while start < end and text[start].isspace():
                    start += 1
                rows.append(
                    SpanProposal(
                        start=start,
                        end=end,
                        text=text[start:end],
                        type=EntityType.TEST_RESULT,
                        source="structured_lab_rule",
                        score=0.96,
                        evidence={"paired_test_start": match.start()},
                    )
                )
        return rows

    def _find_diagnoses(self, text: str) -> list[SpanProposal]:
        if not self._diagnosis_trie:
            return []
        tokens = [
            (match.group(0), match.start(), match.end())
            for match in re.finditer(r"\w+", text, re.UNICODE)
        ]
        normalized_tokens = [normalize_search(token) for token, _, _ in tokens]
        rows: list[SpanProposal] = []
        for index in range(len(tokens)):
            node = self._diagnosis_trie
            cursor = index
            best: tuple[int, list[tuple[str, str]]] | None = None
            while cursor < len(tokens) and normalized_tokens[cursor] in node:
                node = node[normalized_tokens[cursor]]
                cursor += 1
                if "_values" in node:
                    best = (cursor, node["_values"])
            if best is None:
                continue
            final_cursor, values = best
            start = tokens[index][1]
            end = tokens[final_cursor - 1][2]
            rows.append(
                SpanProposal(
                    start=start,
                    end=end,
                    text=text[start:end],
                    type=EntityType.DIAGNOSIS,
                    source="exact_icd_dictionary",
                    score=0.86,
                    evidence={"codes": sorted({code for _, code in values})[:5]},
                )
            )
        return rows
