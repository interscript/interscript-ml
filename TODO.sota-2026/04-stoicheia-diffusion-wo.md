# 04 — WO (queued, unscheduled): plane-factorized char-level masked diffusion

Status: QUEUED (2026-10-01) — entry criteria at the bottom
Literature basis: Stoicheia (arXiv 2608.07249) — 405M character-level
masked-diffusion encoder for Ancient Greek; input factors into five
aligned, independently maskable planes (letters, boundaries,
**diacritics**, capitalization, punctuation). One backbone restores
lacunae / re-segments / **accentuates** / punctuates. Beats Ithaca
24.6 → 15.5 CER with matched random-init controls (methodology kin).

## Why it is the highest-upside architecture item

- Diacritics as an explicit maskable plane IS our task's factorization
  (base letters given; only harakat positions are unpredictable).
- Non-autoregressive parallel decode = potential large CPU-latency win
  over our KV-cache seq2seq (our runtime bottleneck).
- Independent planes compose tasks without retokenization — one model
  could serve diacritization + our other normalization tasks.

## Why it is queued, not scheduled

- Ledger: architecture transfers measured negative at our scale
  (depth cut, lexical memory, PKM; Hebrew depth-cut catastrophic).
  Stoicheia's evidence is restoration/scansion at 405M with heavy
  pretraining (380M words) — not a matched transfer case.
- New runtime contract: non-AR iterative decode does not fit IMF v1
  KV-cache graphs; needs its own export + parity path.
- Diffusion decode needs step-count/quality calibration per language.

## Spec (if entered)

1. Corpus: reuse Arabic combined + news + YallaMorph-aux; planes =
   (base letters, harakat, word boundaries). Letters plane held
   (skeleton-preserving, as our students already do).
2. Backbone: ~300M encoder, plane-aligned embeddings; pretrain
   masked-plane objective, then SFT on diacritization.
3. Gates: same windowed zero-skip SadeedDiac harness; must beat
   4.5701 (student rung) AND 2.2864 (teacher rung) to matter; CPU
   decode latency benchmarked against ara-diac-small-int8static-2.1.
4. Entry criteria: (a) 01 (run-009) lands and re-ranks the teacher
   frontier; (b) owner authorizes a new architecture line; (c) a
   decode-parity design exists for non-AR models (IMF v2 question).
