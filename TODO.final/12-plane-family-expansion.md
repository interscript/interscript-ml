# 12 — Plane family expansion (Hebrew nikud, Urdu)

**Status: HEBREW ARM LAUNCHED** (2026-10-07) — nikud_planes.py TDD 7/7 (order-preserving clusters: corpus writes בְּ sheva-then-dagesh AND שָׁ dot-then-qamats - no uniform canon; render-exactness beats normalization); train_hebrew_plane.py = the run-018 recipe on the v4 combined corpus (50,303 units, 135 classes, 6,036 steps, A100 detached); gate = nakdimon greedy DER vs heb-diac-1.1's 16.44 on the exact seq2seq_der protocol; per-marker eval progress (E2 lesson applied). Urdu arm PAUSED (new corpus coming).
language-agnostic; corpora already sit in the data layer (hewiki,
rababa-hebrew-distilled, urdu corpora).

## Steps
1. Hebrew nikud plane: inventory (54-class analog for nikud combos),
   train (rababa/interscript-train run), full-set eval vs heb-diac-1.1
   (16.44 DER), gates, artifact, index entry, HF publish.
2. Urdu plane: same shape vs urd-diac-1.0 (3.74 CER).
3. Each lands through the standard gates; no index entry without
   provenance.
