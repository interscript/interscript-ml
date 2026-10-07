# WO05 — External benchmark adoption

**Priority:** P2 · **Depends:** WO01 (multiref protocol), WO07 (nikud→IPA
for g2p comparability) · **Gate:** RESULTS.md tables with explicit
protocol columns for at least: heb plane on heb-g2p-benchmark, Arabic
models on reconciled WikiNews-2024.

## Targets

1. **heb-g2p-benchmark** (Phonikud line; ReNikud reports 85.1% word
   accuracy on it): adopt their test set. Our plane emits nikud; with
   WO07's nikud→IPA rules we can report both (a) nikud-restoration
   DER/WER on their vocalized test text and (b) composed IPA word
   accuracy — each labeled with its protocol. If their set is
   nikud-free (plain Hebrew → IPA), report (b) only.
2. **WikiNews-2024 multiref** (from WO01): becomes the OFFICIAL Arabic
   OOD protocol alongside SadeedDiac-25; every future Arabic RESULTS.md
   entry carries both.
3. **MILIM** (ReNikud's spoken benchmark): track its release; evaluate
   only when data is actually obtainable — do not fabricate numbers
   (memory law).

## Deliverables

- Eval scripts under `interscript-models/benchmarks/` (reproducible,
  pinned inputs + sha256s).
- RESULTS.md: new "External benchmarks" section; every number carries
  protocol + input provenance; missing data (MILIM) listed as "awaiting
  release", never estimated.

## Non-goals

Protocol shopping: we adopt THEIR definitions where feasible; where a
definition is ambiguous we publish BOTH numbers labeled, never pick the
flattering one silently.
