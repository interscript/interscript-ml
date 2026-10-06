# TODO.final — the consolidation campaign (2026-10-06)

Closes every remaining thread after the 2026 SOTA campaign and the
secryst→interscript migration. Predecessors: TODO.sota-2026 (closed),
interscript/interscript TODO.complete (closed), TODO.restructure
(closed via interscript-ruby #792).

| # | item | priority | blocked on |
|---|------|----------|------------|
| 01 | pip interscript release | P0 | previous developer (PyPI TP env) |
| 02 | website collapse | P0 | — |
| 03 | trainer rename (rababa → interscript-train) | P0 | — |
| 04 | corpora consolidation | P1 | 03 |
| 05 | secryst deprecation shims | P1 | owner version numbers |
| 06 | legacy archival | P1 | 04, 05 |
| 07 | hub cleanup | P1 | — |
| 08 | ruby provisioning paths | P1 | — |
| 09 | catalog rename → interscript-models + index-v8 | P1 | owner go |
| 10 | version-generation policy | P2 | owner decision |
| 11 | neural parity goldens | P2 | — |
| 12 | plane family expansion (heb/urd) | P2 | — |
| 13 | API edge unification | P2 | — |

Rules carried from prior campaigns: honest specs only (numbers from
measurement); artifacts via Releases+sha256 only; archive never delete;
versions are the owner's to state; TDD on all code paths.
