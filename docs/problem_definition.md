# Problem Definition

**Competition:** Bài 2 — Ontological Reasoning in Medical Knowledge Retrieval

## Goal

Build an AI system that processes **free-form Vietnamese clinical text** (doctor notes, discharge summaries, lab results, EHR) to:

1. Detect and normalize medical concepts
2. Classify concept types
3. Map diagnoses to **ICD-10** and drugs to **RxNorm**
4. Infer contextual assertions (negation, family history, patient history)
5. Infer relationships between concepts in the text

Knowledge bases provided: **ICD-10** (diseases), **RxNorm** (drugs).

---

## Input

A single free-form clinical text string. Examples of source documents:

- Clinical examination notes
- Discharge summaries
- Doctor notes
- Imaging / lab reports
- EHR records

**Example input:**

```
Bệnh nhân bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi,
được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản.
```

Each input contains **multiple** medical concepts.

---

## Output

A JSON **list of entity objects**. Each entity has:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Exact span extracted from input |
| `position` | `[int, int]` | Character offsets `[start, end)` — 0-indexed |
| `type` | string | One of the entity types below |
| `assertions` | `list[string]` | Context flags (max 3); only for CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG |
| `candidates` | `list[string]` | Standard codes; only for CHẨN_ĐOÁN (ICD-10) and THUỐC (RxNorm) |

### Entity types

| Label | Meaning |
|-------|---------|
| `TRIỆU_CHỨNG` | Symptom the patient has |
| `TÊN_XÉT_NGHIỆM` | Name of a lab test performed |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Lab result value (with unit if present) |
| `CHẨN_ĐOÁN` | Diagnosis by the physician |
| `THUỐC` | Drug the patient is taking / was prescribed |

### Assertions

Only apply to `CHẨN_ĐOÁN`, `THUỐC`, `TRIỆU_CHỨNG`. At most 3 per entity:

| Value | Meaning | Example |
|-------|---------|---------|
| `isNegated` | Concept is negated | "không ho" |
| `isFamily` | Related to family member | "bố bệnh nhân đau bụng tương tự" |
| `isHistorical` | Patient medical history | "có tiền sử hen suyễn" |

### Candidates

- `CHẨN_ĐOÁN` → ICD-10 codes (e.g. `K21.0`, `K21.9`)
- `THUỐC` → RxNorm codes (e.g. `360047`, `1660761`)
- Other types: `candidates` is empty `[]`

---

## Full example

**Input:**

```
Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi,
được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. Bệnh nhân có tiền sử sử dụng
Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành tổng phân tích tế bào máu
bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4;
LYPH% (Tỷ lệ bạch cầu lympho):12,8;
```

**Expected entities (summary):**

| Type | Examples | Candidates / Assertions |
|------|----------|-------------------------|
| CHẨN_ĐOÁN | "bệnh trào ngược dạ dày - thực quản" | ICD: K21.0, K21.9 |
| TRIỆU_CHỨNG | "ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi" | — |
| TÊN_XÉT_NGHIỆM | "WBC", "NEUT% (...)", "LYPH% (...)" | — |
| KẾT_QUẢ_XÉT_NGHIỆM | "14,43", "76,4", "12,8" | — |
| THUỐC | "Chlorpheniramine 0.4 MG/ML", "Capsaicin 0.38 MG/ML" | RxNorm: 360047, 1660761; assertion: `isHistorical` |

See [`data/examples/sample_output.json`](../data/examples/sample_output.json) for the JSON format.

---

## Phase 1 dataset (submission format)

- **Test set:** 100 records — [Google Drive folder](https://drive.google.com/drive/folders/1GEARAJjBU3726Et4kZnPjvKGN1O7ghO3?usp=drive_link)
- **Download with gdown (via uv):**

```bash
uv sync

uv run gdown --folder "https://drive.google.com/drive/folders/1GEARAJjBU3726Et4kZnPjvKGN1O7ghO3" \
  -O data/test --remaining-ok
```

- **Input layout:**

```
data/test/
└── input/
    ├── 1.txt
    ├── 2.txt
    ...
    └── 100.txt
```

- **Required output:** one `.json` file per `.txt`, same basename → `data/test/output/`
- Teams must create additional training data beyond the provided test set

> Personal info (name, age, address, phone) in the data is **synthetic**, not real patients.
