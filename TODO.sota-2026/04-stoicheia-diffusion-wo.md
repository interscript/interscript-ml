# 04 — WO (queued, unscheduled): plane-factorized char-level masked diffusion

Status: RUNG GATE CLEARED (2026-10-04) — teacher-tier phase in flight

## Entry design decision (2026-10-03)

Diacritization is a LENGTH-PRESERVING PLANE PROJECTION: the letters
plane is the given skeleton; the haraqat plane is a per-letter
classification. So the model is an ENCODER-ONLY plane predictor, not a
decoder:

- Init: **ByT5-small encoder** (pretrained backbone — the
  from-scratch-collapse law; byte+3 table reused).
- Input: byte embeddings of the SKELETON + a diacritic-plane
  embedding per position (MASK at pass 1; pass k>1 carries pass k-1
  predictions) — Mask-Predict self-conditioning, trained at random
  corruption levels; inference with K passes of FULLY PARALLEL
  prediction.
- Head: per-position classification over the corpus haraqat-combo
  inventory (label = canonical combining-mark combo following each
  letter; non-letter positions labeled none).
- Why this can beat the seq2seq rungs: the AR students decode
  left-to-right; iʿrāb (33% of the residual) depends on sentence
  structure AHEAD. Bidirectional conditioning attacks exactly the
  residual's largest component. Encoder-only + K parallel passes is
  also a CPU-latency win vs byte-by-byte KV decode.
- Corpus: benchmark-convention RUNNING TEXT ONLY (r5-units +
  arabic-combined + news mix) — the r9 convention-drift lesson
  applied by design; NO paradigm tables.
- Gates (pre-registered): windowed zero-skip SadeedDiac-25 — rung
  gate beat 4.5701 (student tier, on-device class); SOTA-dedicated
  gate beat 2.2864 (teacher tier); CPU latency benchmark vs
  ara-diac-small-int8static-2.1.
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


## Results (measured)

**run-017 (v1, 350k units, 1 epoch, K=4):** full-set Total DER
4.7615 / Morph 3.0530 — 0.19pp off the 4.5701 rung. Grid (subset,
labeled): quality monotonically improving 3.295→2.991→2.934 across
checkpoints (UNDERtrained, not overfit); K flat 2/4/8.

**run-018 (v2, 551k units uncapped, 2 epochs, K=2):** full-set
**Total DER 3.5905 / Morph 2.2149** — the 4.5701 rung gate CLEARED by
0.98pp (21% relative error reduction). ONNX int8 artifact: 219 MB
(-45% vs the shipped 491 MB seq2seq int8static) and 1,145 ms/window
on the same machine where the shipped artifact takes 3,459 ms (-67%).
**The plane model is the new on-device frontier on all three axes.**
Gap to the teacher tier (2.2864): 1.30pp.

**Teacher-tier phase (run-019):** same recipe on the ByT5-LARGE
encoder — the SOTA-dedicated gate (beat 2.2864) is the target.

**run-019 epoch-2 (final, 2026-10-05):** full-set **Total DER 2.7397
/ Morph 1.6717** (epoch-1: 3.0078/1.8486). SOTA-dedicated gate (2.2864)
NOT cleared — +0.45pp — and the per-epoch gain curve (−1.17 small,
−0.27 here) makes a third epoch unfavorable. Podium: r7 2.2864 >
plane-large-e2 2.7397 > plane-large 3.0078 > Gemini-Flash 3.1926.
Teacher-tier phase CLOSED on evidence: the plane architecture's edge is
the on-device frontier (run-018), not the teacher tier. The conditional
run-020 full-corpus labeling pass is likewise closed (its trigger was
"only if epoch-2 falls short" — it did; but its own reading stands:
unique-data volume is the lever, and the frontier model run-018 already
holds the efficient tier).

**WO status: ENTERED, executed, closed.** Deliverables shipped:
haraqat_planes.py (byte-exact split/render), run-017/018/019/020
verdicts, secryst-py PlaneModel runtime (PR merged), ONNX export path.
Open packaging: plane v2/large artifact zips + index entries + HF
publish (after this verdict).
