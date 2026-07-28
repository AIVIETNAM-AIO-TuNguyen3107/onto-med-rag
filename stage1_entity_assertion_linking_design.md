# Stage 1 Technical Design
## Entity Finding, Assertion Detection, and ICD-10/RxNorm Entity Linking

**Audience:** Coding agent implementing the first-stage competition pipeline.

**Scope:** Stage 1 only. No model fine-tuning. Use pretrained zero-shot NER, deterministic rules, local terminology databases, and a self-hosted LLM under 9B parameters.

---

# 1. High-level pipeline

```text
Free-form clinical text
        ↓
1. Entity finding
        ↓
2. Assertion finding
        ↓
3. Entity linking
   ├── CHẨN_ĐOÁN → ICD-10
   └── THUỐC → RxNorm
        ↓
4. Deterministic JSON validation
```

These three tasks must remain modular. Do not combine the entire pipeline into one LLM prompt.

---

# 2. Stage 1 design principles

1. Preserve the original text exactly.
2. Final character offsets must refer to the original text.
3. Do not allow the LLM to invent character offsets.
4. Do not allow the LLM to invent ICD-10 codes or RxCUIs.
5. All ICD-10 and RxNorm candidates must come from local terminology indexes.
6. Use the LLM only for missing-entity recovery, type conflict resolution, ambiguous assertion scope, and candidate reranking.
7. Final inference must work without external APIs.
8. Every module must be independently testable.

Mandatory invariant:

```python
original_text[start:end] == entity_text
```

Treat the end offset as exclusive unless the organizer confirms otherwise.

---

# 3. Module 1 — Entity finding

## 3.1 Objective

Identify:

- exact entity text,
- start/end character offsets,
- entity type.

Required types:

```text
TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
CHẨN_ĐOÁN
THUỐC
```

Example:

```json
{
  "text": "đau thượng vị",
  "position": [54, 68],
  "type": "TRIỆU_CHỨNG"
}
```

## 3.2 Three proposal sources

```text
Rule/dictionary extraction
        +
Zero-shot NER
        +
LLM missing-entity recovery
        ↓
Proposal merger
        ↓
Final entity spans and types
```

Run all three independently. Do not use a strict fallback chain.

```text
rules/dictionary ─┐
NER ──────────────┼──→ proposal merger
LLM ──────────────┘
```

Agreement between sources should increase confidence.

---

# 4. Rule-based and dictionary entity extraction

## 4.1 Best use cases

Rules and dictionaries are strongest for:

- medication names,
- dose and concentration,
- route,
- frequency,
- laboratory test names,
- laboratory result values,
- common diagnoses,
- common symptoms,
- abbreviations.

## 4.2 Medication extraction

Example:

```text
amlodipine 10 mg po daily
```

The dictionary finds `amlodipine`. The parser expands the span to include `10 mg`, `po`, and `daily`.

Required medication components:

```text
drug name
ingredient
brand
strength
concentration
dose form
route
frequency
PRN marker
release type
```

Example patterns:

```python
NUMBER = r"\d+(?:[.,]\d+)?"

UNIT = (
    r"(?:mcg|μg|ug|mg|g|kg|ml|mL|l|L|"
    r"meq|mEq|unit|units|iu|%)"
)

STRENGTH = rf"{NUMBER}\s*{UNIT}(?:\s*/\s*{UNIT})?"

ROUTE = (
    r"(?:po|oral|iv|intravenous|im|sc|sq|sl|"
    r"topical|inh|inhaled)"
)

FREQUENCY = (
    r"(?:qd|od|daily|bid|tid|qid|qhs|qam|"
    r"q\d+h|prn|stat)"
)
```

Algorithm:

```python
drug_span = dictionary_match
cursor = drug_span.end

while next_tokens_match_medication_component(cursor):
    cursor = matched_component.end

final_span = original_text[drug_span.start:cursor]
```

Stop expansion at markers such as:

```text
điều trị
for
due to
because of
```

Example:

```text
guaifenesin ml po q6h:prn điều trị ho
```

Expected:

```text
THUỐC: guaifenesin ml po q6h:prn
TRIỆU_CHỨNG: ho
```

## 4.3 Laboratory extraction

Examples:

```text
WBC:14,43
NEUT%:76,4
Glucose 7,2 mmol/L
HbA1c = 8.2 %
```

Expected:

```text
WBC      → TÊN_XÉT_NGHIỆM
14,43    → KẾT_QUẢ_XÉT_NGHIỆM
```

Patterns:

```python
VALUE = r"(?:<|>|<=|>=)?\s*[+-]?\d+(?:[.,]\d+)?"

VALUE_WITH_RANGE = (
    r"\d+(?:[.,]\d+)?\s*[-–]\s*"
    r"\d+(?:[.,]\d+)?"
)

QUALITATIVE = (
    r"(?:âm tính|dương tính|negative|positive|"
    r"trace|\+{1,4}|-{1,4})"
)
```

Keep internal pairing:

```python
{
    "test_name_span": ...,
    "result_span": ...,
    "pair_confidence": 0.97
}
```

## 4.4 Diagnosis dictionary

Each record should include:

```text
code
official_name_en
official_name_vi
aliases
abbreviations
common misspellings
parent_code
inclusions
exclusions
ICD edition
```

Example:

```json
{
  "code": "K21.9",
  "official_name_en": "Gastro-oesophageal reflux disease without oesophagitis",
  "official_name_vi": "Trào ngược dạ dày-thực quản không viêm thực quản",
  "aliases": [
    "trào ngược dạ dày thực quản",
    "trào ngược DD-TQ",
    "GERD",
    "GORD"
  ]
}
```

## 4.5 Symptom dictionary

Each record:

```text
canonical symptom
Vietnamese aliases
English aliases
abbreviations
common misspellings
priority
```

Examples:

```text
ho
ho khan
ho đờm
ho đờm xanh
khó thở
đau ngực
tức ngực
đau thượng vị
ợ hơi
sốt
mất ngủ
lo âu
```

## 4.6 Dictionary matching order

1. Exact case-sensitive match
2. Exact case-insensitive match
3. Strict normalized exact match
4. Loose token-window exact match
5. Fuzzy token-window match
6. Abbreviation rules

Suggested thresholds:

```text
length 1–3:  exact only
length 4–6:  >= 95
length 7–12: >= 90
length >12:  >= 86
```

Use RapidFuzz `ratio`, `token_sort_ratio`, and `token_set_ratio`. Avoid `partial_ratio` for short medical terms.

---

# 5. Zero-shot NER

## 5.1 Role

Use zero-shot biomedical NER as a proposal generator.

Recommended initial model:

```text
Ihor/gliner-biomed-large-v1.0
```

Do not assume it is sufficient for Vietnamese.

## 5.2 Labels

```python
labels = [
    "patient symptom or clinical finding",
    "disease or medical diagnosis",
    "medication including dose route and frequency",
    "laboratory test name",
    "laboratory test result value and unit",
]
```

Mapping:

```python
GLINER_TO_COMPETITION = {
    "patient symptom or clinical finding": "TRIỆU_CHỨNG",
    "disease or medical diagnosis": "CHẨN_ĐOÁN",
    "medication including dose route and frequency": "THUỐC",
    "laboratory test name": "TÊN_XÉT_NGHIỆM",
    "laboratory test result value and unit": "KẾT_QUẢ_XÉT_NGHIỆM",
}
```

## 5.3 Inference

```python
from gliner import GLiNER

model = GLiNER.from_pretrained(
    "models/gliner-biomed",
    local_files_only=True,
)

entities = model.predict_entities(
    text,
    labels=labels,
    threshold=0.35,
)
```

Validate:

```python
assert text[e["start"]:e["end"]] == e["text"]
```

## 5.4 Chunking

For long notes:

1. Split by paragraph.
2. Split oversized paragraphs by sentence.
3. Retain the chunk's original document offset.
4. Convert local to global offsets.

```python
document_start = chunk_start + local_start
document_end = chunk_start + local_end
```

Do not call `.strip()` without accounting for removed characters.

---

# 6. LLM entity recovery

## 6.1 Role

Use Qwen3-8B only to:

- recover missed entities,
- resolve type conflicts,
- handle complex clinical expressions.

Do not use it as the sole extractor.

## 6.2 Prompt

System prompt:

```text
You are a clinical information extraction component.

Extract only medical concepts that appear explicitly in the supplied text.
Every returned "text" value must be copied exactly, character for character,
from the input text.

Never:
- paraphrase,
- translate,
- normalize,
- correct spelling,
- invent a concept,
- generate character offsets.

Allowed types:
- TRIỆU_CHỨNG
- TÊN_XÉT_NGHIỆM
- KẾT_QUẢ_XÉT_NGHIỆM
- CHẨN_ĐOÁN
- THUỐC

Rules:
1. Ignore patient demographics.
2. For medications, include contiguous dose, concentration, route,
   and frequency when written as part of the medication phrase.
3. Separate laboratory test names from results.
4. Do not infer a diagnosis from a symptom.
5. Return only entities missing from EXISTING_ENTITIES.
6. If the same substring appears multiple times, return occurrence,
   counted from 1 in the supplied text.
7. Output valid JSON only.
```

User prompt:

```text
TEXT:
<<<
{chunk_text}
>>>

EXISTING_ENTITIES:
{existing_entities_json}

Return:
{
  "entities": [
    {
      "text": "exact substring",
      "occurrence": 1,
      "type": "allowed type"
    }
  ]
}
```

Post-processing:

1. Confirm literal substring.
2. Locate the requested occurrence.
3. Derive original offset.
4. Reject absent substrings.
5. Reject invalid types.
6. Deduplicate.

---

# 7. Proposal merging

Agreement example:

| Source | Span | Type | Confidence |
|---|---|---|---:|
| Dictionary | đau thượng vị | TRIỆU_CHỨNG | 0.95 |
| NER | đau thượng vị | TRIỆU_CHỨNG | 0.84 |
| LLM | đau thượng vị | TRIỆU_CHỨNG | 0.91 |

Keep the entity and increase confidence.

Conflict example:

| Source | Span | Type |
|---|---|---|
| Dictionary | hen phế quản | CHẨN_ĐOÁN |
| NER | hen phế quản | TRIỆU_CHỨNG |
| LLM | hen phế quản | CHẨN_ĐOÁN |

Resolve using:

- section,
- nearby cues,
- source confidence,
- LLM type adjudication.

Recommended source priority:

```text
structured medication/lab rule
> exact dictionary
> LLM exact-substring proposal
> zero-shot NER
> fuzzy dictionary
```

For overlapping same-type spans:

1. Prefer higher confidence.
2. Prefer medically complete span.
3. Prefer medication spans including dose/route/frequency.
4. Do not merge repeated entities at different positions.

---

# 8. Module 2 — Assertion finding

## 8.1 Objective

Assertion finding assigns context labels to already extracted:

```text
TRIỆU_CHỨNG
CHẨN_ĐOÁN
THUỐC
```

Allowed labels:

```text
isNegated
isFamily
isHistorical
```

Example:

```text
Bệnh nhân không ho.
```

Result:

```json
{
  "text": "ho",
  "assertions": ["isNegated"]
}
```

## 8.2 Architecture

Use:

```text
Section detection
        +
ConText-style trigger and scope rules
        +
LLM fallback for ambiguous cases
```

Do not use NER for assertion detection.

---

# 9. Assertion section detection

Recognize:

```text
medication_history
past_medical_history
family_history
current_medication
diagnosis
laboratory
```

Examples:

```text
Danh sách thuốc trước nhập viện
Tiền sử
Tiền sử gia đình
Thuốc đang dùng
Chẩn đoán ra viện
```

Example:

```text
DANH SÁCH THUỐC TRƯỚC NHẬP VIỆN

1. amlodipine 10 mg
2. aspirin 81 mg
3. metoprolol 50 mg
```

All medications receive:

```text
isHistorical
```

---

# 10. Negation rules

Triggers:

```text
không
không có
không ghi nhận
phủ nhận
chưa thấy
chưa phát hiện
âm tính với
without
denies
no evidence of
```

Example:

```text
Bệnh nhân không ho, không sốt nhưng đau ngực.
```

Result:

```text
ho       → isNegated
sốt      → isNegated
đau ngực → no assertion
```

Scope stops at punctuation, contrast conjunctions, or new sections.

---

# 11. Historical rules

Triggers:

```text
tiền sử
tiền căn
trước đây
đã từng
có lần
thuốc trước nhập viện
thuốc dùng tại nhà
history of
previously
past medical history
```

Example:

```text
Bệnh nhân có tiền sử hen phế quản.
```

Result:

```text
hen phế quản → isHistorical
```

---

# 12. Family rules

Triggers:

```text
bố
cha
mẹ
anh trai
chị gái
em
con
ông
bà
người nhà
gia đình
họ hàng
father
mother
sibling
family history
```

Example:

```text
Mẹ bệnh nhân bị đái tháo đường.
```

Result:

```text
đái tháo đường → isFamily
```

---

# 13. Multiple assertions

Example:

```text
Không ghi nhận tiền sử gia đình mắc ung thư đại trực tràng.
```

Possible annotation:

```json
{
  "text": "ung thư đại trực tràng",
  "assertions": [
    "isNegated",
    "isFamily",
    "isHistorical"
  ]
}
```

Confirm this with organizer guidelines.

Until confirmed:

- apply deterministic trigger rules,
- flag ambiguous cases,
- call the LLM adjudicator.

---

# 14. LLM assertion adjudication

Use only when:

- triggers overlap,
- scope crosses clauses,
- section semantics are unclear,
- negation is uncertain,
- medication history/current use is ambiguous.

Prompt:

```text
Determine which assertions apply to the exact target concept.

Allowed assertions:
- isNegated
- isFamily
- isHistorical

Use only the supplied context.
A concept may have zero or multiple assertions.

Return JSON only:
{"assertions": ["..."]}
```

Input:

```text
SECTION:
{section_name}

CONTEXT:
{sentence_plus_neighboring_sentence}

TARGET:
{exact_entity_text}
```

---

# 15. Module 3 — Entity linking

Entity linking applies only to:

```text
CHẨN_ĐOÁN → ICD-10
THUỐC     → RxNorm
```

Use:

```text
Exact/fuzzy lexical search
        +
BM25 retrieval
        +
Embedding similarity
        +
Candidate fusion
        +
Structured or LLM reranking
```

Precise name:

```text
hybrid retrieval-based medical entity linking
```

The system may be described as ontology RAG, but code generation is forbidden.

---

# 16. ICD-10 entity linking

## 16.1 Knowledge base

Each ICD concept should contain:

```text
code
official English name
official Vietnamese name
aliases
abbreviations
inclusion terms
exclusion terms
parent code
chapter
edition
```

Example:

```json
{
  "code": "K21.9",
  "official_name_en": "Gastro-oesophageal reflux disease without oesophagitis",
  "official_name_vi": "Trào ngược dạ dày-thực quản không viêm thực quản",
  "aliases": [
    "trào ngược dạ dày thực quản",
    "GERD",
    "GORD",
    "trào ngược DD-TQ"
  ],
  "parent_code": "K21"
}
```

## 16.2 Exact ICD edition

Determine whether the organizer uses:

```text
WHO ICD-10
ICD-10-CM
Vietnamese national adaptation
organizer-provided subset
```

Do not combine editions.

---

# 17. ICD retrieval layers

## 17.1 Exact lookup

```text
trào ngược dạ dày thực quản → K21.9
```

## 17.2 Fuzzy matching

Use for misspellings and formatting differences.

```text
Levenshtein
RapidFuzz
character n-gram
```

## 17.3 BM25

Use BM25 over:

- official names,
- aliases,
- inclusion terms,
- Vietnamese translations.

Example query:

```text
viêm phổi do vi khuẩn
```

Candidates:

```text
J15.9 Bacterial pneumonia, unspecified
J18.9 Pneumonia, unspecified organism
J12.9 Viral pneumonia, unspecified
```

## 17.4 Embedding similarity

Embed:

```text
entity mention
ICD official names
ICD aliases
ICD inclusion terms
```

Compare with cosine similarity.

Embedding is candidate generation only, not final truth.

---

# 18. ICD candidate fusion

Retrieve:

```text
exact/fuzzy top 10
BM25 top 20
embedding top 20
```

Deduplicate by ICD code.

Preferred fusion:

```python
RRF(code) = sum(
    1 / (k + rank_in_retriever)
)
```

Recommended `k = 60`.

Weighted fusion is acceptable:

```python
final_score = (
    0.40 * lexical_score
    + 0.30 * bm25_score
    + 0.30 * embedding_score
)
```

---

# 19. ICD LLM reranking

Input:

```text
MENTION:
"bệnh trào ngược dạ dày - thực quản"

CONTEXT:
"Được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản."

CANDIDATES:
K21.0 — GERD with esophagitis
K21.9 — GERD without esophagitis
R12 — Heartburn
K30 — Functional dyspepsia
```

Prompt:

```text
Rank ICD-10 candidates for the extracted diagnosis.

Choose only from supplied codes.
Never invent or modify a code.
Prefer the most specific code supported by the text.
Do not assume unstated details.

Return at most 3 codes in ranked order.
Return JSON only:
{"candidates": ["CODE"], "confidence": 0.0}
```

---

# 20. ICD candidate count policy

Because scoring uses Jaccard similarity:

```text
high-confidence top 1 with large margin → return 1
top 2 are close and text is ambiguous → return 2
otherwise → return no more than 3
```

Initial thresholds:

```text
top score >= 0.92 and margin >= 0.12 → top 1
top two >= 0.80 and margin < 0.12 → top 2
otherwise → top 1–3 after reranking
```

Tune later on validation data.

---

# 21. RxNorm entity linking

RxNorm linking requires parsing:

```text
ingredient
brand
strength
concentration
dose form
route
release type
frequency
```

Example:

```text
amlodipine 10 mg po daily
```

Parse:

```json
{
  "ingredient": "amlodipine",
  "strength": "10 mg",
  "route": "po",
  "frequency": "daily"
}
```

Frequency usually does not determine RxCUI.

---

# 22. RxNorm terminology types

| TTY | Meaning |
|---|---|
| IN | Ingredient |
| PIN | Precise Ingredient |
| MIN | Multiple Ingredients |
| BN | Brand Name |
| SCD | Semantic Clinical Drug |
| SBD | Semantic Branded Drug |
| SCDF | Semantic Clinical Drug Form |
| SBDF | Semantic Branded Drug Form |

Include at least:

```text
IN
PIN
MIN
BN
SCD
SBD
```

---

# 23. RxNorm structured parsing

Example:

```text
metoprolol succinate xl 50 mg po daily
```

Parse:

```json
{
  "ingredient": "metoprolol succinate",
  "release_type": "extended release",
  "strength": "50 mg",
  "route": "oral",
  "frequency": "daily"
}
```

Important distinctions:

```text
metoprolol succinate vs metoprolol tartrate
25 mg vs 50 mg vs 100 mg
```

Do not rely on embeddings alone.

---

# 24. RxNorm retrieval layers

## 24.1 Exact structured lookup

Search variants:

```text
full medication span
medication without route/frequency
drug name + strength
drug name only
```

## 24.2 BM25 and fuzzy retrieval

Search normalized RxNorm names.

## 24.3 Embedding retrieval

Use for:

- brand/generic equivalence,
- abbreviations,
- misspellings,
- alternative dose-form wording.

Embedding should not decide strength.

---

# 25. RxNorm structured scoring

```python
score = (
    0.40 * lexical_name_score
    + 0.25 * ingredient_match
    + 0.20 * strength_match
    + 0.10 * dose_form_match
    + 0.05 * tty_preference
)
```

Strength mismatch should strongly penalize SCD/SBD candidates.

TTY preference:

```text
ingredient + strength + form → prefer SCD/SBD
ingredient + strength only   → consider SCD/SCDF and IN/PIN
ingredient only              → prefer IN/PIN/MIN
brand only                   → prefer BN, then related SBD
```

---

# 26. RxNorm LLM reranking

Example input:

```text
MENTION:
"metoprolol succinate xl 50 mg po daily"

PARSED:
ingredient = metoprolol succinate
strength = 50 mg
release = extended release
route = oral

CANDIDATES:
A. RXCUI ... metoprolol succinate 50 MG Extended Release Oral Tablet
B. RXCUI ... metoprolol tartrate 50 MG Oral Tablet
C. RXCUI ... metoprolol succinate 100 MG Extended Release Oral Tablet
D. RXCUI ... metoprolol
```

Prompt:

```text
Rank RxNorm candidates for the extracted medication.

Choose only supplied RxCUIs.
Never invent or modify an RxCUI.

Use:
- ingredient or brand,
- strength,
- concentration,
- dose form,
- release type,
- route when relevant.

Frequency such as daily, bid, q6h, or prn is usually not encoded
in the RxNorm concept.

Prefer SCD/SBD when strength and form are explicit.
Prefer IN/PIN/MIN/BN when only ingredient or brand is identifiable.

Return at most 3 RxCUIs.
Return JSON only:
{"candidates": ["RXCUI"], "confidence": 0.0}
```

---

# 27. Complete ICD pipeline

```text
CHẨN_ĐOÁN entity
        ↓
Normalize mention and expand abbreviation
        ↓
Exact dictionary lookup
        ├── fuzzy matching
        ├── BM25 retrieval
        └── embedding retrieval
                ↓
        Candidate fusion
                ↓
        LLM reranking
                ↓
        Final ICD candidate list
```

---

# 28. Complete RxNorm pipeline

```text
THUỐC entity
        ↓
Parse ingredient/brand/strength/form/route/frequency
        ↓
Exact structured lookup
        ├── fuzzy/BM25 name retrieval
        └── embedding retrieval
                ↓
Structured filtering:
ingredient + strength + form + TTY
                ↓
        LLM reranking
                ↓
        Final RxCUI candidate list
```

---

# 29. Is this RAG?

It can be called:

```text
ontology RAG
terminology RAG
```

The more precise term is:

```text
hybrid retrieval-based medical entity linking
```

Traditional RAG:

```text
query → retrieve documents → LLM generates answer
```

This pipeline:

```text
entity mention
    ↓
retrieve ontology candidates
    ↓
rerank candidates
    ↓
return existing identifier
```

The source of truth is the ICD/RxNorm database. The LLM only interprets context and ranks candidates.

---

# 30. Final Stage 1 pipeline

```text
STEP 1 — Entity finding
Rules/dictionaries
+ zero-shot GLiNER
+ Qwen3-8B recovery
        ↓
Merge spans and resolve types

STEP 2 — Assertion finding
Section detection
+ ConText-style trigger/scope rules
        ↓
Qwen3-8B only for ambiguous cases

STEP 3 — Entity linking

CHẨN_ĐOÁN:
exact/fuzzy
+ BM25
+ embedding
        ↓
candidate fusion
        ↓
Qwen reranking
        ↓
ICD-10 candidates

THUỐC:
drug parser
+ exact/fuzzy
+ BM25
+ embedding
        ↓
structured strength/form/TTY filtering
        ↓
Qwen reranking
        ↓
RxNorm candidates
```

---

# 31. Implementation order

## Milestone 1

- Original-text preservation
- Offset-safe tokenizer
- Pydantic schemas
- JSON validator

## Milestone 2

- Section detection
- Medication rules
- Laboratory rules
- Diagnosis/symptom dictionaries
- Assertion rules

## Milestone 3

- GLiNER-BioMed
- Qwen3-8B local endpoint
- Entity recovery prompt
- Type adjudication prompt
- Assertion prompt

## Milestone 4

- ICD exact index
- Fuzzy search
- BM25
- Embedding index
- RRF fusion
- LLM reranking

## Milestone 5

- Medication parser
- Local RxNorm database
- Exact/fuzzy/BM25
- Embedding retrieval
- Structured scoring
- LLM reranking

## Milestone 6

- Batch inference
- Offline execution
- Debug reports
- Output validation
- Packaging

---

# 32. Non-negotiable safeguards

1. LLM-generated text must be an exact substring.
2. LLM-generated codes are forbidden.
3. ICD codes must exist in the selected ICD edition.
4. RxCUIs must exist in the packaged RxNorm release.
5. Embedding similarity is candidate generation, not final truth.
6. BM25 is lexical retrieval, not final truth.
7. Medication strength mismatch must be heavily penalized.
8. Limit candidate count because of Jaccard scoring.
9. Preserve exact original offsets.
10. Final inference must work with the network disabled.

---

# 33. Expected deliverables

```text
src/entity_finding/
src/assertion_detection/
src/icd_linking/
src/rxnorm_linking/
src/llm/
src/validation/
configs/
tests/
scripts/
README.md
```

Required commands:

```bash
python -m clinical_nlp.cli build-icd
python -m clinical_nlp.cli build-rxnorm
python -m clinical_nlp.cli infer
python -m clinical_nlp.cli validate
python -m clinical_nlp.cli package
```

Required unit tests:

- exact offsets,
- repeated substrings,
- medication parsing,
- lab name/result splitting,
- assertion scope,
- ICD retrieval,
- RxNorm strength matching,
- candidate validation,
- final JSON schema.
