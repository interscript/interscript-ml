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

- **Status:** implemented + validated on shipped artifacts; policy adopted.
- **Hypothesis:** byte students have flat top-1 margins (median 0.12
  logits on khm-latn), so quantization flips near-tie argmaxes at KLD
  ~1e-5 — invisible to the CER-delta release gate.
- **Protocol:** teacher-forced forward on both sides (torch decoder vs
  ONNX zip graphs) over the parity probe pairs; per-position top1−top2
  reference margins, argmax flip rate, KL(reference||zip), share of
  flips below the corpus p10 margin. `imf.parity.run_margin_analysis`;
  emitted by every `modal_export parity` gate and standalone via
  `modal_export margins` (read-only for published zips).
- **Measured (khm-latn-1.0, 300-pair probe, 2,869 positions):**
  fp32 — 0 flips, KLD 0; fp16 — 84 flips (2.93%), KLD 6.28e-06,
  77% of flips at near-tie margins; int8 — 71 flips (2.47%),
  KLD 1.14e-05, 90% near-tie. CER parity gate passed all three.
- **Policy (adopted):** embedding-like tensors — byte embeddings, tied
  lm_head, relative-attention bias, and any memory-layer lookup tables —
  are a separate quantization class: they stay fp16 (≥ int8 floor) when
  the body is quantized. Precision floors are keyed on access pattern,
  not tensor size (the Qwen/Unsloth lesson).

## E2 — PKM memory-layer student (ara-diac-small run-003-pkm)

- **Status:** in flight (launched 2026-08-27, A10G, labels reused from
  run-002; single-variable design vs run-002).
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

- **Status:** queued behind E2 (same A10G slot, serialized).
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

## Parked

- **Speculative decoding** (LongCat converts sparsity→speed): revisit
  only if API latency data shows p95 decode binding. No code, by design.
