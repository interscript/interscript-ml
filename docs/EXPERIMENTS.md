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
  cannot see. Open item: re-examine the Hebrew int8 artifact
  (per-channel quantization of the ByT5-base graphs, or serve fp16)
  and add a margin-gate threshold to the release policy.
- The distilled students behave as the decode analysis predicts: flat
  but consistent — tha-g2p-small's shipped int4 flips 0.26% of
  positions, all near-tie.
- **Policy (adopted):** embedding-like tensors — byte embeddings, tied
  lm_head, relative-attention bias, and any memory-layer lookup tables —
  are a separate quantization class: fp16 (≥ int8 floor) when the body
  is quantized. Precision floors are keyed on access pattern, not
  tensor size (the Qwen/Unsloth lesson).

## E2 — PKM memory-layer student (ara-diac-small run-003-pkm)

- **Status:** in flight (launched 2026-08-27, A10G, labels reused from
  run-002; single-variable design vs run-002). Survived one mid-run
  eviction: resumed from step-2000 via the checkpoint guard.
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

- **Status:** queued behind E2 (same A10G slot, serialized; the
  server-side `qwen_next_chain` orchestrator sequences both arms from
  durable volume markers — best/config.json, final_eval.json,
  chain_log.jsonl — with 20-min stall detection and respawn).
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
