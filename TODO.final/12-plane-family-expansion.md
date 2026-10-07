# 12 — Plane family expansion (Hebrew nikud, Urdu)

**Status: HEBREW ARM DONE, END-TO-END** (2026-10-07) — verdict 12.48 DER (gate 16.44 cleared, −3.96pp; first read 32.64 was a window-stitching harness bug, disclosed + corrected from cached predictions, zero mismatches); artifact shipped: release + sha sidecar, index entry #255, curated card, HF live (27 repos), collection updated; smoke = correct nikud, byte-exact round-trip. Urdu arm PAUSED (new corpus coming).
language-agnostic; corpora already sit in the data layer (hewiki,
rababa-hebrew-distilled, urdu corpora).

## Steps
1. Hebrew nikud plane: inventory (54-class analog for nikud combos),
   train (rababa/interscript-train run), full-set eval vs heb-diac-1.1
   (16.44 DER), gates, artifact, index entry, HF publish.
2. Urdu plane: same shape vs urd-diac-1.0 (3.74 CER).
3. Each lands through the standard gates; no index entry without
   provenance.
