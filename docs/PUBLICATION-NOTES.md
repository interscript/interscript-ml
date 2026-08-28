# Publication notes — what is worth publishing, and where it lives

Status: 2026-08-28. Rule: a claim is publishable when it is
harness-measured and recorded in RESULTS.md (models/protocol numbers)
or EXPERIMENTS.md (registered experiments). Nothing below cites an
un-landed number.

## Complete, measured, citable now

### 1. The artifact contract (IMF v1) and cross-runtime parity
One zip, three ONNX graphs, fixed byte table, per-member SHA-256,
byte-identical output from Ruby/Python/TypeScript; opset-14 floor for
old consumer runtimes. Paper: sections 3–4 (section-imf). The framing
contribution — neural models under the same discipline as
deterministic transliteration maps.

### 2. The decode-protocol correction
Beam-4 with length normalization inflates flat byte-student PER 4.2×
(12.06 published vs 2.85 real, same artifact, greedy). Language-
dependent at the teacher tier (Hebrew gains ~12 DER points from beam,
Arabic nothing). Paper: section-decode. This is the paper's most
quotable calibration result.

### 3. Pretrained-or-collapse (the client-tier frontier law)
Random-init byte students collapse at every capacity tested (33M
75.80 PER, 70M 78.51); the pretrained rung is the whole cliff (300M
→ 2.85). The Arabic replica was label-corrupted in transit —
disclosed, retracted as evidence, kept as a reproducibility lesson.
Paper: section-frontier + section-repro.

### 4. Controlled aux-representation ablation (r8, 2026-08-27)
Identical teacher/corpus/seed/init; only the aux stream's output
representation varies. Morphology 2.5793 < phonemic IPA 2.6588 < none
2.6775 (full 1,200-paragraph windowed zero-skip); the IPA projection
itself was learned (2.3% CER probe). **The strong form of "phonemes
help diacritization" fails; lexical-morphological knowledge is the
active ingredient.** Paper: leaderboard section. RESULTS.md (rababa):
r8 section.

### 5. Margin-aware parity + the head-fp32 quantization fix (E1)
Teacher-forced margin analysis (flip rate / KLD / near-tie share)
measured across the entire catalog — invisible to the CER gate.
Headline diagnosis: the shipped Hebrew int8 artifact flipped 9.34% of
positions, 80% at confident margins. Controlled probe matrix isolated
the quantized *head* (not weight granularity: per-channel recovered
almost nothing at +25% size); keeping the 1.2M-parameter head in fp32
cut flips 36× (0.26%, all near-tie) at +0.4% size. Now the export
default + a release gate + a regression test. Paper: IMF section
"Margin-aware parity" bullet. EXPERIMENTS.md E1.

### 6. The memory-layer capacity experiment (E2, 2026-08-28)
Single-variable: shipped ByT5-small student vs +3 product-key memory
layers (+85.9M params, near-zero compute; gates verified engaged).
8.259 → 7.555 full-set DER — 0.704pp of the 5.677pp teacher-student
gap (12.4% relative), **below the pre-registered ≥1.0pp bar**. Honest
conclusion: the distillation gap is dominated by optimization and
domain coverage, not parameter capacity. First controlled test of the
LongCat/Qwen "embedding scaling" axis on a byte-level seq2seq student.
Paper: frontier section (fourth question). RESULTS.md run-003-pkm.

### 7. Measurement-discipline findings
- Subset-selection artifact: first-300-paragraphs 3.66 vs full-set
  8.26 (teacher reproduces 2.5815/2.5793 — harness soundness proven).
  Paper: leaderboard section. Standing rule: full-set-only publication.
- Leaderboard positioning: r6 2.5793 is the best dedicated model under
  the protocol (behind Claude-3.7-Sonnet's 1.3941, ahead of GLM-5.2
  2.6911, Gemini-Flash 3.1926, GPT-4 3.8645, Sadeed-1.5B 7.2915).

### 8. Negative results (honesty assets, all in the log)
RL teacher polishing flat/negative ×3; microkimi bridges improve
structure but not accuracy; teacher beam-search unnecessary for
Arabic; per-channel int8 rejected on measurement; the 30 MiB tier
closed as infeasible without pretraining.

## In flight — slots reserved, no claims yet

### 9. Muon optimizer A/B on the memory student (E3)
Same architecture/data/seed as E2, optimizer swapped (Newton–Schulz
orthogonalized momentum; embedding-like params on AdamW per the E1
access-pattern policy). Training CE ~0.007 vs AdamW's ~0.02 at equal
steps, ~1.2s/step vs ~3.4s — DER eval in flight; adopt gate ≥0.3pp.
Lands: run-004-pkm-muon/final_eval.json → EXPERIMENTS.md E3 → paper
training-methods note.

### 10. Arabic news-domain adaptation (r7) — ID LANDED 2026-08-28
**2.2864 / 1.3343 full-set windowed zero-skip — new best dedicated
model, −0.29pp over r6** (the news mix improved in-domain, not just
OOD; behind only Claude-3.7-Sonnet's published 1.3941). OOD half
(WikiNews-2024 multi-ref, gate: beat r6's 19.82/12.46) running via the
auto-launched actor. → rababa RESULTS.md (recorded) → paper leaderboard
table + OOD note once OOD confirms; canonical-teacher promotion
pending that check.

## Venue fit (working notes)

The spine is systems-with-measurements: an artifact contract +
distillation discipline + three calibration/measurement corrections
(decode, subset, quantization-margins) that generalize beyond our
stack. The frontier law (pretrained-or-collapse) and the controlled
capacity/aux/optimizer experiments give it empirical heft. Package as
one paper (current paper.adoc); the margin/head finding alone is also
a strong short workshop paper if a split is ever wanted.
