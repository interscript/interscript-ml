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

### 9. Muon optimizer A/B on the memory student (E3) — LANDED 2026-08-28
**4.8287 vs 7.5553 full-set (−2.727pp from the optimizer alone); adopt
gate (≥0.3pp) exceeded 9×. ADOPTED.** Training CE ~0.007 vs ~0.02 at
equal steps, ~1.2s/step vs ~3.4s, no stability events. With E2 this
completes a controlled decomposition of the distillation gap at the
ByT5-small rung: ≈0.70pp capacity + 2.73pp optimization + 2.25pp
residual (domain coverage). The strongest training-methods result in
the paper — first controlled Muon measurement on byte-level seq2seq
distillation. Paper: frontier section. EXPERIMENTS.md E3, RESULTS.md
run-004. **Factorial cell 4 (vanilla+Muon, run-005): 5.2945 — the 2×2 closes cleanly** (optimizer alone −2.96pp, memory alone −0.70pp, combined −3.43pp, roughly additive). Paper carries the full table.

### 10. Arabic news-domain adaptation (r7) — COMPLETE 2026-08-28, NEW CANONICAL TEACHER
**ID 2.2864/1.3343 (−0.29pp over r6) AND OOD WikiNews-2024 multi-ref
17.38/11.83 (vs r6's 19.82/12.46)** — r7 sweeps both surfaces; the
teacher-lineage domain trade-off is gone. Best dedicated model under
the protocol, behind only Claude-3.7-Sonnet's published 1.3941, now
well clear of GLM-5.2 (2.6911). r7 replaces r6 as the canonical
Arabic teacher for future distillations. Paper leaderboard table
updated; rababa RESULTS.md carries both tables. Follow-on (user
decision): ara-diac-2.0 teacher release; students re-distilled from
r7 with Muon (E3) are the natural ara-diac-small-2.0.

## Venue fit (working notes)

The spine is systems-with-measurements: an artifact contract +
distillation discipline + three calibration/measurement corrections
(decode, subset, quantization-margins) that generalize beyond our
stack. The frontier law (pretrained-or-collapse) and the controlled
capacity/aux/optimizer experiments give it empirical heft. Package as
one paper (current paper.adoc); the margin/head finding alone is also
a strong short workshop paper if a split is ever wanted.

## ara-diac-tiny: the "clean-label re-run" wasn't (2026-08-29)

Narrative caution for the paper: the Aug-24 retraction flagged that the
83.08 tiny-collapse number was measured on mojibake labels. The
attempted clean re-run (run-004) silently reused the same Aug-23
snapshot via a baked-in `labels_file` path — and reproduced 83.0797
exactly, which is itself the tell: identical-to-4-decimals means same
data, not same capacity law. Two lessons for the experiments section:
(1) label-provenance must be content-hashed into the run record, not
inferred from filenames; (2) a reproduced number is only evidence of
reproducibility when the input pipeline is versioned. run-005
(regenerating labels live from the r6 teacher) is the honest test.

## ara-diac-tiny run-005: the law holds (2026-08-29)

The clean-label rerun scores 74.68 vs the poisoned run's 83.08 (gate
3.07): corruption explained 8pp, capacity explains the rest. The
"pretrained backbone or nothing" claim now has Arabic evidence
matching the Thai ablation, closing the retraction arc
(verdict -> retraction -> poisoned rerun -> clean rerun). For the
paper: report the pair (83.08, 74.68) as the label-quality and
capacity bounds of the same from-scratch rung, with the teacher
reproducing 1.3205 across all three measurements as the harness
control. Next rung on the frontier: stitch-down from ByT5-small
pretraining (ridge-fit width bridge), which changes init, not data or
capacity alone.
