# Competition Timeline

## Phases

| Phase | Round | Dates | Deliverable |
|-------|-------|-------|-------------|
| **Phase 1** | Vòng 1 — Sơ loại | 02/07/2026 → 30/07/2026 | ZIP submission (GPU) |
| **Phase 2** | Vòng 2 — Sơ khảo | 17/08/2026 → 19/08/2026 | API endpoint (GPU) |
| **Phase 3** | Vòng 3 — Chung kết | 09/09/2026 → 10/09/2026 | API endpoint (GPU) |

---

## Phase 1 — Team sprint plan (~3 weeks)

### Week 1 — Knowledge gap

Fill the team's knowledge gap on the problem domain and data.

- Read ICD-10, RxNorm, and clinical NLP documentation
- Explore datasets and download knowledge bases
- Write research notes (`research/`)
- Build shared understanding of the pipeline and output format

**Deliverables:**
- Research markdown docs (ICD-10, RxNorm, NER, assertion, linking)
- Knowledge bases in `data/kb/`
- High-level pipeline agreement

---

### Week 2 — Baselines, architecture & proof-of-concept

Identify and evaluate suitable baseline methods through literature review and preliminary experimentation.

**Literature review**
- Each member reviews relevant papers
- Compare existing approaches for:
  - Medical Named Entity Recognition (NER)
  - Entity Linking (ICD-10 / RxNorm)
  - Assertion Detection
- Write summaries in `research/papers/` (target: 5–6 papers)

**Baseline selection**
- Compare candidate models/methods per task
- Select the most appropriate baseline for each pipeline stage
- Document decisions in `docs/baseline_decisions.md`

**Architecture & PoC**
- Design the overall implementation architecture (`docs/architecture.md`)
- Build an initial proof-of-concept pipeline in `src/`
- Run the baseline on the provided test dataset (`data/test/`)
- Log experiments in `experiments/`

**Evaluation**
- Establish reference performance on the 100-record test set
- Identify strengths and weaknesses per module
- Define the implementation plan for Week 3 improvements

**Deliverables:**
- Paper summaries + baseline comparison
- Documented baseline decisions
- Working PoC pipeline (end-to-end JSON output)
- Initial performance benchmark + error analysis
- Week 3 implementation plan

---

### Week 3 — Fix, iterate & submit

Focus on improving the baseline and delivering the best possible Phase 1 submission.

- Fix errors identified in Week 2 evaluation
- Iterate on weak modules (NER, classification, assertion, linking, ranking)
- Create / refine training data as needed (`data/raw/`)
- Run experiments, compare against Week 2 benchmark
- Package final predictions → submission ZIP

**Deliverables:**
- Best-performing pipeline version
- `data/test/output/` — 100 JSON files
- Phase 1 ZIP submission (deadline: **30/07/2026**)

---

## Milestones

| Date | Milestone |
|------|-----------|
| End Week 1 | Research docs complete, KBs downloaded, team aligned on problem |
| End Week 2 | Baselines chosen, PoC running, benchmark + error analysis done |
| 30/07/2026 | Phase 1 final submission |
