# Stage 1 Implementation Plan

## 1. Status and implementation boundary

This document is the review gate before implementation.

- No implementation starts until this plan is approved.
- The 100 files under `input/` are final test inputs, not labeled development data.
- The supplied ICD workbook and RxNorm source are temporary resources for the current test. Both must be replaceable later.
- The supplied medication example is the only golden end-to-end example currently available.
- The system may use cloud-hosted models during development and current testing, then move to a dedicated GPU without changing task logic.
- The submission system allows at most five submissions per day, so every scored run must be reproducible and change one controlled variable.

## 2. Locked output contract

Each document produces one UTF-8 JSON file containing a top-level array ordered
by ascending start offset.

Output names preserve the input stem:

```text
input/1.txt   → runs/<run_id>/outputs/1.json
input/37.txt  → runs/<run_id>/outputs/37.json
input/100.txt → runs/<run_id>/outputs/100.json
```

There must be exactly one output JSON file for every input text file and no
cross-document entities.

```json
{
  "text": "amlodipine 10 mg po daily",
  "type": "THUỐC",
  "candidates": ["308135"],
  "assertions": ["isHistorical"],
  "position": [58, 83]
}
```

Rules:

1. Offsets are Python Unicode character offsets over the unmodified original string.
2. End offsets are exclusive.
3. Every entity must satisfy:

   ```python
   original_text[start:end] == entity["text"]
   ```

4. Entities are flat and non-overlapping.
5. Each span has exactly one entity type.
6. Allowed entity types:

   - `TRIỆU_CHỨNG`
   - `TÊN_XÉT_NGHIỆM`
   - `KẾT_QUẢ_XÉT_NGHIỆM`
   - `CHẨN_ĐOÁN`
   - `THUỐC`

7. Allowed assertion values:

   - `isNegated`
   - `isFamily`
   - `isHistorical`

8. An entity may have zero, one, two, or all three assertions.
9. `assertions` is emitted for every entity; ineligible or assertion-free entities receive `[]`.
10. `candidates` is emitted only for `CHẨN_ĐOÁN` and `THUỐC`.
11. `CHẨN_ĐOÁN` candidates are ICD-10 codes from the active ICD index.
12. `THUỐC` candidates are RxCUIs from the active RxNorm index.
13. A laboratory-result span:

    - includes the value and a contiguous unit;
    - excludes separators, reference ranges, and abnormal flags;
    - includes qualitative values such as `âm tính` and `dương tính`.

Baseline output decisions:

- A linkable entity with no acceptable link receives `"candidates": []`.
- ICD output uses the canonical code without trailing `*` or `†` notation.
- All explicit medical entities remain eligible for extraction, including educational or spliced passages. Discourse context informs assertions but does not silently remove entities.

## 3. Proposed package structure

```text
pyproject.toml
configs/
  base.yaml
  models/
  experiments/
src/clinical_nlp/
  cli.py
  config.py
  schemas.py
  text.py
  pipeline.py
  validation/
  entity_finding/
    rules.py
    merger.py
  assertion_detection/
    sections.py
    context_rules.py
  ner/
    base.py
    gliner.py
    hf_token_classifier.py
    ensemble.py
  llm/
    base.py
    qwen_transformers.py
    openai_compatible.py
    openai_backend.py
    anthropic_backend.py
  icd_linking/
    source.py
    index.py
    retriever.py
  rxnorm_linking/
    source.py
    parser.py
    index.py
    retriever.py
  experiments/
    manifest.py
    diff.py
  supervision/
    contracts.py
    stage_runner.py
    reports.py
tests/
  fixtures/
  unit/
  integration/
scripts/
artifacts/
runs/
```

Generated indexes, model weights, secrets, and run outputs will not be committed as source files.

## 4. Core data model

The internal model keeps proposals separate from final entities.

```text
Document
  id
  original_text

Chunk
  text
  document_start
  document_end

SpanProposal
  start
  end
  text
  proposed_type
  source
  source_score
  evidence

Entity
  start
  end
  text
  type
  assertions
  candidates

LinkCandidate
  identifier
  canonical_name
  terminology_type
  component_scores
  retrieval_sources
```

Normalized strings are search-only views. Any normalized view that can produce a span must retain a mapping back to original character positions.

## 5. Entity-finding pipeline

### 5.1 Proposal sources

Every document runs through:

1. deterministic rules and dictionaries;
2. the configured NER backend or NER ensemble;
3. LLM missing-entity recovery.

The LLM sees existing proposals so it can focus on missed entities, but its outputs remain independently attributed as LLM proposals.

### 5.2 Deterministic extraction

Implement high-precision rules for:

- medication name, salt, strength, concentration, form, route, frequency, release type, and PRN marker;
- laboratory test/result pairing;
- exact diagnosis lookup from the current ICD aliases;
- configurable symptom lexicon;
- common abbreviations and formatting variants;
- masked text, which must not be treated as a medication name.

Medication expansion stops before indication cues such as `điều trị`, `cho`, `do`, or `because of`. The indication is separately eligible as a symptom or diagnosis.

### 5.3 Proposal merging

Merge in two stages:

1. Resolve identical spans:

   - combine source evidence;
   - resolve type conflicts using structured-rule priority, section context, calibrated source reliability, and LLM adjudication when needed.

2. Resolve partial overlaps:

   - prefer medically complete medication and laboratory spans;
   - prefer exact structured rules over fuzzy proposals;
   - use a deterministic weighted interval selection so the final spans are flat and non-overlapping;
   - never merge repeated occurrences at different offsets.

Raw confidence values from different model families are not directly comparable. Source-specific weights remain configuration values until score feedback provides evidence for adjustment.

## 6. Assertion detection

Assertion detection applies meaningful labels to symptoms, diagnoses, and medications. Laboratory entities receive an empty assertion list.

### 6.1 Deterministic layer

Implement:

- section detection;
- ConText-style forward and backward triggers;
- clause, punctuation, contrast, and section scope termination;
- medication-history section propagation;
- support for simultaneous assertion labels.

### 6.2 Family rules

Family terms are contextual triggers, not automatic entity-wide labels.

Positive evidence includes:

- explicit `tiền sử gia đình` sections;
- kinship subject plus a medical predicate, such as `mẹ em bị ...`;
- phrases such as `gia đình có người mắc ...`.

Required disambiguation:

- `em` alone is a first-person pronoun; only compounds such as `em trai` or `em gái` indicate kinship.
- `con` must not match animal or classifier uses such as `con chó`.
- `mẹ`, `cha`, `bố`, `ông`, and `bà` must be connected to the target condition, not merely occur in educational advice.
- pediatric Q&A requires distinguishing the child as the current patient from a family-history mention.

### 6.3 LLM adjudication

Use the LLM only when deterministic scope or subject attribution is ambiguous. The LLM receives the exact target span, section, sentence, neighboring sentence, and deterministic evidence. Its output is restricted to the three allowed labels.

## 7. ICD-10 source and linking

### 7.1 Current source

Use `DM ICD10-19_8_BYT.xlsx` through an `ICDSource` adapter.

Observed source characteristics:

- 36,689 populated rows;
- 13,078 usable canonical codes after notation handling;
- many duplicate code/name rows;
- non-NFC Vietnamese names;
- trailing `*` and `†` notation;
- three visibly corrupt rows that must be rejected.

### 7.2 Build process

`build-icd` will:

1. record the source checksum and workbook metadata;
2. read the header from row 5 and data from row 6;
3. preserve raw code/name fields for provenance;
4. create canonical codes by separating trailing `*` and `†`;
5. reject malformed codes such as `T112233`, `I787`, and `I65565`;
6. deduplicate canonical code/name pairs;
7. retain distinct names as aliases;
8. build exact, normalized, fuzzy, BM25, and optional embedding indexes.

### 7.3 Retrieval and reranking

Retrieve candidates using:

- exact alias match;
- strict normalized match;
- fuzzy character/token match;
- BM25;
- optional multilingual embeddings.

Fuse candidates by code. Exact high-confidence matches bypass LLM reranking. Ambiguous candidates are reranked by the LLM, which may select only supplied codes.

The source adapter allows a later ICD release to replace the current workbook without changing retrieval or pipeline code.

## 8. RxNorm source and linking

### 8.1 Current source

Use `how-to-query-the-rxnorm-data.ipynb` as a reference for the BigQuery public RxNorm dataset.

The notebook is not an inference implementation: it uses `bq_helper`, requires network access, and demonstrates pathway queries rather than complete mention linking.

### 8.2 Build process

Implement two source adapters:

- `BigQueryRxNormSource` for the current test;
- `RrfRxNormSource` for a later official local release.

The preferred current workflow is a one-time BigQuery export followed by a local index. Do not query BigQuery once per medication during final batch inference.

The local index stores:

- RxCUI;
- preferred and alternative names;
- TTY;
- suppression status;
- ingredient, brand, form, and related-concept relationships.

Include at least:

- `IN`
- `PIN`
- `MIN`
- `BN`
- `SCD`
- `SBD`
- `SCDF`
- `SBDF`

### 8.3 Medication parsing and scoring

Parse:

- ingredient or brand;
- salt;
- strength or strength range;
- concentration;
- dose form;
- route;
- release type;
- frequency and PRN, which usually do not determine the RxCUI.

Candidate scoring prioritizes ingredient, then strength, form, release type, and TTY. A mismatch is strongly penalized but not always rejected because the supplied example demonstrates permissive backoff:

- incomplete `guaifenesin` information maps to an SCD;
- incomplete nystatin concentration maps to an ingredient;
- `clonazepam 1.5 mg` maps to the available 1 MG tablet candidate.

The supplied example is implemented as an exact regression fixture; it is not treated as enough evidence to generalize every backoff rule.

## 9. Plug-and-play NER

Define one model-neutral interface:

```python
class NERBackend(Protocol):
    def predict(
        self,
        document: Document,
        chunks: list[Chunk],
        labels: list[str],
    ) -> list[SpanProposal]: ...
```

Initial adapters:

- GLiNER biomedical;
- GLiNER multilingual;
- Hugging Face token classification;
- ensemble;
- no-model backend for deterministic tests.

Initial model experiments:

1. `Ihor/gliner-biomed-large-v1.0`
2. `urchade/gliner_multi-v2.1`
3. `cbc-528a/BamiBERT-ViMedNER`

All adapters must:

- accept raw text without destructive normalization;
- convert chunk offsets back to document offsets;
- verify every returned substring;
- expose model ID, revision, labels, threshold, runtime, and device in the run manifest.

## 10. Plug-and-play LLM with hidden high-effort reasoning

Define a task-neutral structured interface:

```python
class LLMBackend(Protocol):
    def generate_json(
        self,
        task: LLMTask,
        request: BaseModel,
        response_schema: type[BaseModel],
    ) -> BaseModel: ...
```

Initial adapters:

- Qwen3.5 Transformers;
- local or hosted OpenAI-compatible server;
- OpenAI;
- Anthropic;
- deterministic mock.

### 10.1 Qwen3.5 behavior

Primary model:

```text
Qwen/Qwen3.5-9B
```

Requirements:

- thinking mode enabled;
- sufficient reasoning-token budget;
- only the final answer after the thinking block is parsed;
- reasoning text is discarded and never stored in normal logs;
- final content must be JSON and pass the task schema;
- failures caused by an unfinished thinking block or invalid JSON follow a bounded retry policy;
- the exact model revision and Transformers/runtime version are recorded.

For servers with a Qwen reasoning parser, discard `reasoning_content` and parse only `content`.

For OpenAI or Anthropic reasoning-capable models, request high reasoning effort where supported and retain only the final structured response. API keys come from environment variables and never from configuration files.

### 10.2 LLM safety constraints

The LLM may:

- propose exact-substring entities;
- choose a type;
- choose assertions;
- rank supplied terminology candidates.

The LLM may not:

- create offsets;
- return absent substrings;
- invent ICD codes or RxCUIs;
- bypass deterministic output validation.

## 11. Validation and failure behavior

Validation occurs after every module and before output.

Reject or quarantine:

- invalid types or assertions;
- absent substrings;
- incorrect offsets;
- overlaps in the final entity set;
- duplicate identical entities;
- terminology identifiers absent from the active index;
- model text surrounding the final JSON;
- masked placeholders presented as actual medication names.

The batch pipeline must continue processing other files when one document fails. Failures go to a separate diagnostic report and must not silently produce corrupted JSON.

## 12. Testing without ground truth

No accuracy, precision, recall, or F1 claim will be made from the final test inputs.

### 12.1 Automated tests

Implement:

- supplied medication example as an exact end-to-end golden fixture;
- original-offset and end-exclusive tests;
- repeated-substring tests;
- non-NFC Vietnamese tests;
- flat overlap-resolution tests;
- medication parser tests;
- lab value/unit and qualitative-result tests;
- negation, historical, family, and multi-assertion tests;
- family false-positive cases involving `em`, `con chó`, and educational `cha mẹ`;
- ICD cleaning and retrieval tests;
- RxNorm strength/form/backoff tests;
- candidate-membership tests;
- final JSON schema and ordering tests;
- mocked LLM hidden-reasoning tests.

### 12.2 Model comparison without labels

Model reports may compare:

- valid-span rate;
- invalid JSON rate;
- rule/model agreement;
- unique proposal coverage;
- pairwise output differences;
- latency, memory, and cloud cost;
- deterministic repeatability.

Higher entity count is not treated as higher quality.

## 13. Modular supervision and first full run

### 13.1 Module contract

Every module is independently replaceable and must expose the same lifecycle:

```text
load configuration
validate declared inputs
run one document or a batch
emit typed artifacts
emit counts, warnings, timing, and model usage
validate declared outputs
```

A module may not mutate artifacts from an earlier stage. Re-running a module
creates a new stage artifact or a new run. Each module returns a `StageResult`
with:

- status;
- input and output artifact paths;
- document counts;
- proposal/entity/candidate counts;
- rejection reasons;
- warning and error counts;
- elapsed time and model usage;
- configuration and implementation hashes.

This permits:

- replacing one NER or LLM backend;
- rerunning linking without repeating NER;
- rerunning assertion logic without repeating entity discovery;
- supervising one document before a full batch;
- resuming after a failed cloud or model call;
- comparing stage outputs before integration.

### 13.2 Inspectable document artifacts

The first baseline run creates:

```text
runs/<run_id>/
  config.yaml
  manifest.json
  source_manifest.json
  stages/
    00_preflight_summary.json
    01_chunking_summary.json
    02_rules_summary.json
    03_ner_summary.json
    04_llm_recovery_summary.json
    05_merge_summary.json
    06_assertions_summary.json
    07_retrieval_summary.json
    08_reranking_summary.json
    09_validation_summary.json
  documents/
    1/
      chunks.json
      rule_proposals.json
      ner_proposals.json
      llm_proposals.json
      merged_entities.json
      assertions.json
      medication_parses.json
      icd_candidates.json
      rxnorm_candidates.json
      reranked_entities.json
      validation.json
  outputs/
    1.json
    2.json
    ...
    100.json
  diagnostics/
  diff_from_parent.json
```

Reasoning content is never written to these artifacts. Only final structured
LLM responses, timing, token counts, and validation results are retained.

### 13.3 Stage sequence

#### Stage 00 — Preflight

Inputs:

- configuration;
- 100 input text files;
- temporary ICD and RxNorm sources or built indexes;
- configured model endpoints.

Checks:

- exactly 100 uniquely named `.txt` inputs;
- readable UTF-8 without modifying normalization;
- source and model availability;
- terminology/model/configuration checksums;
- output directory is new or explicitly resumable;
- golden and unit tests pass.

Output:

- immutable source manifest and preflight report.

#### Stage 01 — Offset-safe loading and chunking

For each document:

- preserve the exact original string;
- detect sections, paragraphs, and safe chunk boundaries;
- record chunk start/end offsets;
- validate reconstruction against the original text.

Output:

- `chunks.json`.

#### Stage 02 — Deterministic proposals

Run medication, laboratory, dictionary, abbreviation, and exact terminology
rules independently.

Output:

- `rule_proposals.json`;
- rejection and rule-hit summary.

#### Stage 03 — NER proposals

Run the configured NER backend on the same chunks. Convert local spans to
document offsets and reject any substring mismatch.

First baseline:

- `Ihor/gliner-biomed-large-v1.0`;
- configured labels and a recorded initial threshold;
- no silent fallback to a different model.

Output:

- `ner_proposals.json`.

#### Stage 04 — LLM recovery

Use `Qwen/Qwen3.5-9B` with thinking enabled to recover missing entities and
adjudicate explicit type conflicts. Discard reasoning and retain only validated
final JSON.

Output:

- `llm_proposals.json`;
- request status, timing, token counts, retries, and final-response validity.

#### Stage 05 — Proposal merging

Resolve identical spans, type conflicts, and partial overlaps. Produce one
flat, non-overlapping entity set without assertions or terminology links.

Output:

- `merged_entities.json`;
- accepted/rejected proposal provenance.

#### Stage 06 — Assertion detection

Run section rules, contextual scope rules, contextual family logic, and
high-reasoning LLM adjudication only for ambiguous cases.

Output:

- `assertions.json`;
- deterministic evidence and final labels, without model reasoning.

#### Stage 07 — Structured parsing and candidate retrieval

For diagnoses:

- query the temporary ICD index through exact, fuzzy, BM25, and configured
  embedding retrieval.

For medications:

- parse ingredient, strength, concentration, form, route, release type, and
  frequency;
- query the temporary RxNorm index;
- apply structured candidate scoring.

Output:

- `medication_parses.json`;
- `icd_candidates.json`;
- `rxnorm_candidates.json`.

#### Stage 08 — Candidate reranking

Bypass the LLM for exact/high-margin links. Send only ambiguous candidate sets
to Qwen, with thinking enabled and hidden. The model may select only supplied
identifiers.

Output:

- `reranked_entities.json`;
- per-candidate component scores and selection provenance.

#### Stage 09 — Final validation

For each document, verify:

- every span is an exact original substring;
- end-exclusive Python offsets;
- ascending order;
- flat non-overlap;
- allowed types and assertions;
- required/omitted fields by entity type;
- every ICD code and RxCUI exists in the active index;
- no reasoning or surrounding model prose;
- JSON serialization succeeds.

Output:

- `validation.json`.

No invalid document is silently promoted to final output.

#### Stage 10 — One-file-per-input output

Serialize each validated document independently:

```text
input/<stem>.txt → runs/<run_id>/outputs/<stem>.json
```

Then validate the output directory:

- exactly 100 JSON files;
- stems match the 100 input stems;
- every file contains one JSON array;
- no additional temporary files in `outputs/`;
- directory checksum recorded in the run manifest.

### 13.4 First output canary

Before running all 100 documents, execute the complete Stage 01–10 chain for
`input/1.txt`.

The canary is successful when:

- `runs/<run_id>/outputs/1.json` exists;
- all stage artifacts for document `1` exist;
- every invariant passes;
- all LLM reasoning has been discarded;
- the supervision report contains no unhandled error;
- the same configuration can resume into the remaining 99 inputs without
  recomputing successful document `1`.

This validates integration, not extraction accuracy, because `input/1.txt` has
no ground truth.

### 13.5 First baseline batch

After the canary:

1. freeze the run configuration and source/model manifests;
2. resume the same run for inputs `2.txt` through `100.txt`;
3. quarantine failed documents and retry only their failed stage;
4. require all 100 documents to pass Stage 09;
5. produce exactly 100 files under `outputs/`;
6. run a final directory-level validator;
7. create a score-submission manifest without changing the JSON files.

The first baseline batch uses:

- deterministic extraction and assertion rules;
- GLiNER-BioMed as the initial NER;
- Qwen3.5-9B with hidden high-effort reasoning;
- the supplied temporary ICD and RxNorm resources;
- ambiguity-gated LLM terminology reranking.

### 13.6 Supervision commands

The orchestrator will support:

```bash
python -m clinical_nlp.cli run-stage --stage rules --document 1
python -m clinical_nlp.cli run-stage --stage ner --document 1
python -m clinical_nlp.cli run-document --document 1
python -m clinical_nlp.cli resume-run --run-id <run_id>
python -m clinical_nlp.cli inspect-document --run-id <run_id> --document 1
```

Exact flags may be adjusted during implementation, but single-stage,
single-document, resume, and inspection capabilities are mandatory.

## 14. Submission experiment strategy

Each run creates:

```text
runs/<run_id>/
  config.yaml
  manifest.json
  outputs/
  diagnostics/
  diff_from_parent.json
  submission_manifest.json
```

The manifest records:

- source-tree or Git commit identifier;
- configuration hash;
- input checksums;
- terminology checksums;
- model IDs and revisions;
- prompts and schema versions;
- generation settings;
- output checksum;
- runtime, device, latency, and cloud cost;
- resulting competition score when available.

Rules for the five daily submissions:

1. Establish a stable baseline first.
2. Change one major variable per experimental submission.
3. Inspect a structured output diff before submitting.
4. Do not spend a submission on a run that fails invariants.
5. Reserve at least one submission for rollback or correction.
6. Combine improvements only after their individual score effects are understood.
7. Watch for leaderboard overfitting because the feedback is a single score over an unlabeled test set.

Suggested first sequence:

1. deterministic rules + initial NER + Qwen + baseline linking;
2. change only the NER model or ensemble;
3. change only the LLM adjudicator;
4. change only terminology candidate scoring;
5. reserve, rollback, or combine confirmed improvements.

## 15. CLI deliverables

Planned commands:

```bash
python -m clinical_nlp.cli build-icd
python -m clinical_nlp.cli build-rxnorm
python -m clinical_nlp.cli infer
python -m clinical_nlp.cli run-stage
python -m clinical_nlp.cli run-document
python -m clinical_nlp.cli resume-run
python -m clinical_nlp.cli inspect-document
python -m clinical_nlp.cli validate
python -m clinical_nlp.cli diff-runs
python -m clinical_nlp.cli prepare-submission
```

Every command supports an explicit configuration file and produces machine-readable diagnostics.

## 16. Implementation phases and review gates

### Phase 0 — Plan approval

Deliverable:

- this reviewed implementation plan.

Gate:

- user approval of this plan.

### Phase 1 — Core contracts and validation

Deliverables:

- package scaffold;
- schemas;
- raw-text/chunk utilities;
- output validator;
- golden medication fixture.

Gate:

- all offset, overlap, schema, repeated-string, and non-NFC tests pass.

### Phase 2 — Temporary terminology indexes

Deliverables:

- ICD workbook adapter and index;
- RxNorm BigQuery/RRF source interfaces;
- local terminology schema;
- candidate-membership validation.

Gate:

- reproducible indexes with checksums;
- corrupt ICD rows excluded;
- known example codes resolve correctly.

### Phase 3 — Deterministic entity and assertion baseline

Deliverables:

- medication and lab extraction;
- diagnosis/symptom dictionaries;
- section and assertion rules;
- contextual family logic;
- deterministic overlap resolver.

Gate:

- unit fixtures pass;
- every output is offset-valid and non-overlapping.

### Phase 4 — Model adapters

Deliverables:

- NER protocol and initial adapters;
- LLM protocol and Qwen/OpenAI-compatible adapters;
- hidden-reasoning handling;
- model-independent prompts and response schemas.

Gate:

- adapters pass contract tests;
- reasoning is absent from saved final outputs and normal logs;
- invalid model responses cannot reach final JSON.

### Phase 5 — Linking and end-to-end integration

Deliverables:

- ICD hybrid retrieval;
- medication parser and RxNorm structured scoring;
- ambiguity-gated LLM reranking;
- modular stage supervision;
- document `1` canary;
- resumable batch inference and diagnostics;
- one JSON output per input.

Gate:

- supplied golden example passes;
- all candidates exist in the active terminology index;
- `outputs/1.json` passes the complete canary run;
- full input batch produces exactly 100 JSON files without schema violations.

### Phase 6 — Controlled model trials and submission

Deliverables:

- run manifests;
- model/configuration diff reports;
- first baseline submission package;
- score ledger.

Gate:

- submitted output is reproducible from its manifest and checksum.

## 17. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| No labeled development set | Rely on invariants, synthetic fixtures, one golden example, controlled submission changes, and conservative rules |
| Noisy and spliced documents | Extract explicit spans; use section/subject context for assertions; do not silently discard passages |
| Non-NFC Vietnamese text | Preserve original text and maintain explicit normalized-to-original mappings |
| Temporary terminology sources | Hide ingestion behind replaceable source interfaces and pin source checksums |
| RxNorm example contains permissive matches | Preserve structured scores and configurable backoff; regression-test the supplied example |
| Model confidence scores are incomparable | Use source-specific weights and agreement evidence rather than raw score equality |
| Qwen reasoning leaks into output | Separate/discard reasoning, parse final content only, and test logs and artifacts |
| Cloud and local backends differ | Enforce one common structured backend contract |
| Five-submission daily limit | One-variable experiments, pre-submission diffs, reproducible manifests, and a reserved submission |
| Leaderboard overfitting | Prefer changes with general technical justification and retain a stable baseline |
| Failure inside a monolithic batch | Immutable stage artifacts, per-document status, resume support, and single-stage reruns |

## 18. Deferred decisions that do not block the first implementation

1. The exact upload container, if the platform later requires the 100 JSON files
   to be placed in a ZIP or another archive.
2. Replacement ICD and RxNorm releases for a later final round.
3. Whether a stronger OpenAI or Anthropic model replaces or supervises Qwen in
   a later scored experiment.
4. Whether version control should be initialized. The first implementation will
   not initialize Git without a separate request; run manifests will use source
   file hashes.
