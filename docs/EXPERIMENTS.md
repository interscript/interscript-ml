# Experiment registry

Pre-registered experiments: hypothesis, protocol, and gate recorded
**before** results exist. Numbers graduate to RESULTS.md only when
harness-verified; paper.adoc claims only after a RESULTS.md entry.
The registry diff is the pre-registration evidence.

Origin: the Qwen3.8-Flash-Next technique review (2026-08-27; basis paper
arXiv 2601.21204, LongCat-Flash-Lite — verified). Full working notes:
`TODO.qwen-next/` in the rababa working tree (uncommitted by design).

---

## E1 — Margin-aware parity gates

- **Status:** implemented + validated across every shipped artifact;
  policy adopted.
- **Hypothesis:** byte students have flat top-1 margins, so quantization
  flips near-tie argmaxes at KLD ~1e-5 — invisible to the CER-delta
  release gate.
- **Protocol:** teacher-forced forward on both sides (torch decoder vs
  ONNX zip graphs) over the parity probe pairs (first 300 test pairs per
  model); per-position top1−top2 reference margins, argmax flip rate,
  KL(reference||zip), share of flips below the corpus p10 margin.
  `imf.parity.run_margin_analysis`; emitted by every `modal_export
  parity` gate and standalone via `modal_export margins` (read-only for
  published zips; JSONs on secryst-models:/imf/<model>/).
- **Measured across the catalog (2026-08-27):**

| Artifact | Precision | Flips | Rate | KLD | ref. margin p50 | near-tie share |
|---|---|---|---|---|---|---|
| fas-g2p-1.0 | fp32 | 0/15,103 | 0.00% | 0 | 0.200 | — |
| tha-g2p-base-1.0 | fp32 | 0/6,932 | 0.00% | 0 | 0.421 | — |
| heb-diac-1.1 | fp16 | 6/34,178 | 0.02% | ~0 | 0.205 | 1.00 |
| urd-diac-1.0 | fp16 | 0/7,870 | 0.00% | 0 | 0.213 | — |
| urd-g2p-1.0 | fp16 | 3/6,281 | 0.05% | ~0 | 0.169 | 1.00 |
| khm-latn-1.0 | fp16 | 84/2,869 | 2.93% | 6.3e-06 | 0.121 | 0.77 |
| **heb-diac-1.1** | **int8** | **3,193/34,178** | **9.34%** | 2.4e-05 | 0.205 | **0.20** |
| urd-g2p-1.0 | int8 | 107/6,281 | 1.70% | 5.1e-06 | 0.169 | 0.97 |
| khm-latn-1.0 | int8 | 71/2,869 | 2.47% | 1.1e-05 | 0.121 | 0.90 |
| tha-g2p-small-1.0 | int8 | 63/6,932 | 0.91% | 1.2e-03 | 0.421 | 0.89 |
| urd-diac-1.0 | int8 | 27/7,870 | 0.34% | 1.6e-05 | 0.213 | 1.00 |
| tha-g2p-small-1.0 | int4 | 18/6,932 | 0.26% | 4.6e-04 | 0.421 | 1.00 |

All rows passed the CER parity gate at release. Readings:
- fp32 is exact everywhere (harness sanity); fp16 is benign (≤0.05%)
  except khm-latn — the flattest margins in the catalog (p50 0.121),
  2.93% flips, 77% near-tie.
- **heb-diac-1.1 int8 is the outlier: 9.34% flip rate with only 20% of
  flips at near-tie positions** — 80% of its argmax flips occur where
  the reference was confident. That is the dangerous class the CER gate
  cannot see.
- **Root cause found and fixed at the export default (2026-08-27,
  controlled probes on the same 300 pairs):** the culprit is the
  quantized *head*. Per-channel weights alone: 8.50% flips, 78% still
  confident-position, +25% artifact size — rejected. Keeping
  `/lm_head/MatMul` in fp32 (body int8): **0.26% flips (36x fewer),
  KLD 47x lower, 100% of remaining flips near-tie, +0.4% size**;
  head-fp32 + per-channel adds nothing. Quantizing the node that
  computes argmax moves the decision boundary directly.
  `export_zips` now excludes the head MatMul from int8 by default
  (`imf.export.head_matmul_names` + `nodes_to_exclude`). Shipped int8
  zips predate this; re-exporting them is a release decision.
- The distilled students behave as the decode analysis predicts: flat
  but consistent — tha-g2p-small's shipped int4 flips 0.26% of
  positions, all near-tie.
- **Policy (adopted):** embedding-like tensors — byte embeddings, tied
  lm_head, relative-attention bias, and any memory-layer lookup tables —
  are a separate quantization class: fp16 (≥ int8 floor) when the body
  is quantized. Precision floors are keyed on access pattern, not
  tensor size (the Qwen/Unsloth lesson).

## E2 — PKM memory-layer student (ara-diac-small run-003-pkm)

- **Status:** COMPLETE (2026-08-28). **Outcome: positive but below the
  pre-registered win bar.**
- **Measured (full 1,200 paragraphs, windowed zero-skip, greedy):**
  PKM student **7.5553** DER-CE vs run-002 vanilla ByT5-small **8.259**
  (teacher reproduces 2.5815) — 0.704pp of the 5.677pp teacher-student
  gap closed (12.4% relative) at +85.9M lookup params and near-zero
  added compute (gathers, not matmuls).
- **Verdict per the pre-agreed rule (≥1.0pp = win): NOT met.** The
  capacity axis is real but not the dominant term of the gap; the rest
  is modeling/optimization (E3 tests the optimization half) and domain
  coverage (the subset lesson from run-002). Reported exactly as
  measured — no threshold-moving.
- Engagement was verified independent of outcome: gates moved off zero
  by step-500 (0.0008-0.0028) and settled at 0.034-0.053 by step-10,500
  (15-20x) — the memory branch contributed measurably but modestly,
  matching the 0.70pp outcome. CE 2.07 → 0.02 over 10,995 steps.
- Launched 2026-08-27, A10G, labels reused from run-002 (single-variable
  design). Survived one mid-run eviction: resumed from step-2000.
- **Hypothesis:** the 5.68pp teacher→student gap (r6 2.5815 → ByT5-small
  8.259 full-set) is partly a *capacity* gap that lookup memory closes
  at near-zero compute — parameters and compute are separable (arXiv
  2601.21204; PKM on character-level LM: Lample et al., NeurIPS 2019).
- **Design:** google/byt5-small backbone + 3 product-key memory layers
  on decoder blocks [-2, -4, -6] (128² = 16,384 slots, top-32 reads,
  ~+25M params), zero-init output gates (pretrained function preserved
  at step 0 — tested). Teacher r6 frozen; identical corpus, labels,
  limits, seed, and 3-epoch schedule as run-002.
- **Protocol:** windowed zero-skip Misraj DER-CE, full 1,200 paragraphs
  (the published harness; `modal_distill evaluate_der`).
- **Pre-agreed gate:** ≤ 3.07 windowed DER-CE (the run-002 gate).
- **Pre-agreed verdict rule:** PKM wins if it closes ≥ 1.0pp of the
  5.68pp full-set gap at equal decode-time compute. No movement ⇒ the
  gap is modeling/optimization, not capacity — publishable negative.
- **Comparison targets:** run-002 student 8.259 / teacher 2.5815
  (full-set, published in RESULTS.md).

## E3 — Muon optimizer A/B (run-004-pkm-muon)

- **Status:** COMPLETE (2026-08-28). **Outcome: ADOPTED — the gate is
  exceeded 9x.**
- **Measured (full 1,200, windowed zero-skip, greedy):** Muon arm
  **4.8287** DER-CE vs the single-variable AdamW arm (run-003-pkm)
  7.555 — **−2.727pp from the optimizer alone** (adopt gate ≥0.3pp).
  Against the shipped vanilla student (8.259): 3.430pp of the 5.677pp
  canonical gap closed (60.4%). Teacher reproduced at 2.5997 in this
  container (range across eval containers 2.5793–2.5997, bf16
  autocast; protocol consistent).
- Training-side corroboration: CE ~0.007 vs the AdamW arm's ~0.02 at
  equal steps; step time ~1.2s vs ~3.4s (the <15% overhead gate met
  with margin — Newton–Schulz is cheap next to 1450-byte windows); no
  stability events.
- **Factorial cell 4 (run-005-muon, vanilla + Muon, 2026-08-28):
  5.2945.** The 2x2 closes cleanly:

  | Full-set DER | AdamW | Muon |
  |---|---|---|
  | vanilla ByT5-small | 8.259 | 5.295 |
  | + PKM memory | 7.555 | **4.829** |

  Decomposition (teacher ~2.6): optimizer alone −2.96pp; memory alone
  −0.70pp (−0.47 under Muon); combined −3.43pp — roughly additive,
  slightly sub-additive on the memory term. Optimization is the
  dominant recoverable term at this rung; capacity is second; residual
  ~2.2pp is domain coverage (the subset lesson).
- **Hypothesis:** orthogonalized-momentum updates (Newton–Schulz; the
  optimizer Qwen3.8-Flash-Next / LongCat report) help even in
  knowledge-limited distillation fine-tunes — unmeasured territory for
  byte-level seq2seq students. Prior expectation: small (RL negative;
  data-side levers won before).
- **Design:** identical to E2 except optimizer — Muon on 2D hidden
  matrices (lr 0.01, momentum 0.95, wd 0.01, cosine), AdamW group for
  embedding-like params including the memory tables (random-access
  class per E1 policy). Same seed, data, schedule.
- **Pre-agreed adopt gate:** ≥ 0.3pp DER improvement at equal steps, no
  stability regressions, step-time overhead < 15%.
- **Report:** either direction goes to the paper's training-methods
  appendix.

## E4 — ara-diac-small-2.0 candidate (run-006-r7-muon)

- **Status:** COMPLETE (2026-08-29). **PASSED — 4.8218** full-set
  windowed DER (gate ≤ 6.26; registered prediction 4.3–5.0; teacher r7
  reproduces 2.289 vs documented 2.2864). −3.44pp / 42% relative vs the
  shipped 1.0 at identical architecture and size; matches the PKM arm's
  4.829 without memory layers. Released as ara-diac-small-2.0.
- **Hypothesis:** the two measured wins compound — the r7 canonical
  teacher (better labels; ID 2.2864 vs 2.5793) plus the E3-adopted
  Muon optimizer (−2.727pp on r6 labels) — moving the client rung far
  below the shipped 8.259.
- **Design:** identical corpus/limits/schedule to run-002/003/004/005;
  teacher = run-007-news/best (fresh greedy labels, teacher_labels_r7);
  optimizer = Muon (embedding-like group on AdamW); vanilla ByT5-small
  (the shipped architecture — the 2.0 ships what was measured).
- **Pre-agreed gate:** beats the shipped student by ≥2.0pp full-set
  windowed DER (i.e. ≤ 6.26) to justify the 2.0 release; the strict
  teacher+0.5pp budget remains the disclosed north star.
- **Prediction (registered):** 4.3–5.0, by E3's 4.829 on weaker labels.

## Parked

- **Speculative decoding** (LongCat converts sparsity→speed): revisit
  only if API latency data shows p95 decode binding. No code, by design.
