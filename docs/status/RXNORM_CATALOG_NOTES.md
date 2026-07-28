# Local RxNorm catalog notes

The workspace contains three untracked local catalog files:

| File | Term type | Role |
|---|---|---|
| `rxnorm/in.json` | IN | Ingredient concepts |
| `rxnorm/scd.json` | SCD | Semantic clinical drug concepts |
| `rxnorm/sbd.json` | SBD | Semantic branded drug concepts |

They are required to reproduce Direct V1/V2/V3 terminology linking. Their
upstream crawl command, source endpoint, retrieval date, and redistribution terms
were not recorded in the repository, so the files must not be committed to the
public/team Git history as ordinary source.

The access-controlled Drive archive contains the exact local files and their
SHA-256 hashes for team reproducibility. Anyone publishing or redistributing
them outside the team must first confirm the applicable RxNorm/UMLS terms.

No SCDC catalog was crawled or tested. The SCDC idea remains an investigated but
unrun follow-up, not a completed experiment.
