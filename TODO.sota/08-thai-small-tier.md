# WO08 — Thai small-tier student (HF training)

**Priority:** P3 · **Cost:** ~2–4 h `l4x1` · **Gate:** **≤ 3.5% PER
(greedy, corpus-level, the runtime protocol)** at **< 10 MB** int8.
Ship only if both hold; else documented negative.

## Why

FastThaiG2P wins voice-agent deals on sub-ms latency. A sub-10MB
neural student that holds ~2.85%-class PER gives us a latency tier to
quote next to theirs (and WO03's hybrid stacks on top of it).

## Recipe

1. **Student**: char-level encoder-decoder transformer, d=256, 4+4
   layers, ~10–15M params, BOS/EOS IPA-target vocab (small, ~500) —
   NOT byte-level (byte seq2seq needs ByT5 capacity to phonotactically
   decode; a tiny model needs the compact target vocab; CTC is ruled
   out by the Thai CTC limitation verdict).
2. **Data**: teacher-generated pairs — umt5-thai-g2p-v2-0.5k teacher
   (public HF, 4.43% PER lineage) or the existing tha-g2p-small
   (2.85% greedy, local artifact) labeling Thai text (Kaikki train
   side + hewiki-style Thai wiki text if needed). Sequence-level KD:
   train student on teacher OUTPUTS (hard labels) — the recipe that
   already worked for tha-g2p-small.
3. **Training**: HF Jobs `l4x1` (small model), bf16, cosine.
4. **Export**: int8 static ONNX; measure size + greedy PER with the
   exact runtime harness (1,219 Kaikki test, corpus-level, true
   Levenshtein, same numbers as RESULTS.md).

## Deliverables

- `train_thai_tiny_hf.py` + distill-data builder.
- Verdict in RESULTS.md either way (size table: 219MB student vs tiny).
- **If gate passes**: ship chain, owner-confirmed version; the hybrid
  (WO03) gets a `--tiny` model option.

## Honest risks

10–15M params may fail phonotactic decoding entirely (mode collapse
lineage in this repo's history is real). Gate is cheap; a miss is a
clean negative that closes the sub-10MB idea rather than an open
question.
