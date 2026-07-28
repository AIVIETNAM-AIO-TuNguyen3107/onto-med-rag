# Score review: 29.18 → target ~48

Submitted run: `runs/full100-v1/outputs` (100 files, 2,790 entities).

| Component | Score | Weight |
|---|---|---|
| text_score (`1 − WER`) | 33.07 | 0.3 |
| assertions_score | 36.23 | 0.3 |
| candidates_score | 20.97 | **0.4** |
| **final_score** | **29.1764** | top team: 49 |

Official formula, confirmed:

```
final_score = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score
```

## The rule that reframes everything

```
J_X(i) = 1                     if len(gt) = 0 and len(pred) = 0
J_X(i) = 0                     if len(gt) = 0 and len(pred) ≠ 0
J_X(i) = |gt ∩ pred| / |gt ∪ pred|   otherwise
```

Plus: a span with the right text but the **wrong type is counted twice and scores 0 on all three metrics**.

Two consequences that invert the obvious strategy:

**1. A correctly-found symptom is worth as much as a perfectly-linked diagnosis.** `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, and `KẾT_QUẢ_XÉT_NGHIỆM` have empty gold candidates. Empty vs empty scores **J = 1.0**. So every symptom we correctly extract earns a full point on *both* Jaccard metrics with zero linking work — while a diagnosis we find but mis-code earns 0 on candidates.

**2. Every metric is gated on span + type accuracy.** A gold concept we never predict scores 0 on assertions *and* candidates. You cannot fix candidates without first finding the concept.

## The dominant problem is recall, not linking

Both organizer examples agree on entity density:

| Source | entities / 1000 chars |
|---|---|
| Organizer example 1 (medication list) | 34.3 |
| Organizer example 2 (`sample_output.json`) | 35.0 |
| GLiNER baseline (no LLM) | 17.0 |
| **Our submission** | **13.7** |

At ~34/1000 the 203,817-character corpus implies roughly **6,900 gold entities. We submitted 2,790 — about 40%.** 29 of 100 documents fall below 10 entities per 1000 characters.

`input/100.txt` (1,293 chars) shows what this looks like concretely. We extracted **5 entities**:

```
CHẨN_ĐOÁN    'tiền sản giật' ×3   → O14  (correct)
TRIỆU_CHỨNG  'chảy máu'
THUỐC        'aspirin'            → 1191
```

Missed in the same text: `cục máu đông` (blood clot), `đi tiêu ra máu` (bloody stool), `đại tiện ra máu đỏ tươi` (fresh rectal bleeding), `mang thai 22 tuần` (22-week pregnancy), `xét nghiệm` (test), `biến chứng` (complications), `lo lắng` (anxiety). A careful annotator would mark 15–25 here; organizer density implies ~44.

Missing ~60% of concepts caps all three scores near the observed values — which is exactly what we see. `assertions_score` of 36.23 is close to the fraction of concepts we identify at all, not a statement about assertion logic.

**Correction to my earlier review:** I ranked candidate-filling first, before the metric definition was available. That was wrong. Recall dominates, because it multiplies into all three components.

## Two convention errors worth fixing

**Medication spans must include the full sig.** The organizer writes `"amlodipine 10 mg po daily"` as one `THUỐC` span — dose, route, and frequency included, but excluding the indication (`điều trị ho` becomes a separate `TRIỆU_CHỨNG`). Only **6% of our 285 THUỐC spans contain a digit**; we emit bare `'omeprazole'`, `'aspirin'`, `'metoprolol'`.

**RxNorm codes must be SCD, not ingredient.** Gold uses `308135` (amlodipine 10 MG tablet), `243670` (aspirin 81 MG). We emit `7646` (omeprazole IN), `6057` (metoprolol IN), `1191` (aspirin IN). Truncating the span makes the correct SCD unreachable — one root cause, two metrics lost.

## Ranked plan

### 1. Raise recall — affects all three metrics
The architecture is backwards for this corpus: rules + GLiNER propose, the LLM prunes. But GLiNER is an **English biomedical** model reading Vietnamese prose, and it is the primary recall source. Qwen understands this text far better.

Invert it — make the LLM the primary extractor over each chunk, with rules and GLiNER as supplements rather than gatekeepers. Concretely:
- Promote the existing `_recover_entities` pass into a full extraction pass (it already reconstructs offsets host-side via `find_occurrence`, so span integrity is preserved).
- Enable reasoning on it — it currently runs with `reasoning_enabled=False`.
- Drop `ner_threshold` 0.35 → ~0.2 and let review handle precision.
- Loosen review's keep bar; a missed concept costs all three metrics, a kept-but-imperfect one costs less.

### 2. Fix medication spans + SCD linking — text + candidates
Extend the span to absorb dose/route/frequency where present, stop at the indication. Then query RxNorm with the full string so SCD codes become reachable.

### 3. Fill empty candidates for genuine diagnoses — candidates only
386 of 705 diagnoses emit nothing, which is a guaranteed 0 when gold has codes. Retrieval often *has* the right answer and the reranker discards it (`'bệnh dại'` retrieves `A82.1:0.61` and returns nothing).

Do this **only for spans that survive as real diagnoses.** Under the `J=0 when gt empty but pred non-empty` rule, coding a junk span like `'bệnh'` is now actively harmful, not neutral.

### 4. Let the LLM propose codes, validated by catalog membership
`_batch_rerank` raises `ValueError("LLM invented a terminology candidate ID")` for codes outside the retrieved set. Retrieval genuinely fails on `bệnh Kawasaki` (returns measles and Chagas; `M30.3` absent) and `amyloidosis` (returns nothing). All the correct codes **are in the catalog** — verified via `ICDIndex.contains()`: `M30.3`, `E85`, `I10`, `A82`, `C80` all `True`.

Swap provenance validation for membership validation. Keep the strict guard for RxNorm, where numeric identifiers invite hallucination.

### 5. Assertions — handle with care
Do **not** blanket-restore the baseline's 239. Over-asserting now costs a full point per concept (`gt` empty, `pred` non-empty → 0). Target section-driven cases only: the organizer marks every drug under a pre-admission medication list as `isHistorical`, which is exactly what `AssertionDetector`'s section logic is built for. `isFamily` at 0 across 100 documents is still wrong.

### Do not do
**Emitting 2–3 candidates per drug.** The organizer's example shows exactly one RxNorm code per medication. With gold of size 1, emitting 3 scores 1/3. My earlier suggestion to widen candidate counts applies at most to diagnoses (`K21.0` + `K21.9`), and even there it needs measurement first.

## Use the submission budget deliberately

20 submissions over 4 days, 4 today. Recall work is a large change with no local ground truth, so bundle by theme and keep one variable per submission:

| # | Change | Tests |
|---|---|---|
| 1 | Recall only (items 1) | Does density 13.7 → ~25 lift all three? |
| 2 | + medication spans & SCD (2) | text + candidates |
| 3 | + candidate fill & LLM codes (3, 4) | candidates in isolation |
| 4 | + assertions (5) | assertions in isolation |

Hand-labeling 10–15 documents is still the highest-value hour available — it converts the remaining 16 submissions from guesses into confirmations, and it is the only way to settle diagnosis candidate cardinality.

## Caveats

- **Both organizer examples are dense clinical lists**, not prose Q&A. True density for these documents is probably below 34/1000, so "40% of gold" is a hypothesis, not a measurement. The direction is well-supported; the magnitude is not.
- The organizer's own labeling is inconsistent — `"sốt đau"` is one symptom while `"lo âu mất ngủ"` is split into two. Treat the examples as convention hints, not as a specification.
- WER's exact computation (token alignment over concatenated text vs per-entity) is still unstated, so text_score predictions are the least reliable.
