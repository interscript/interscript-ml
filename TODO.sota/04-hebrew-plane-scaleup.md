# WO04 — Hebrew plane scale-up (HF training)

**Priority:** P2 · **Cost:** ~10–15 h on `a100-large` · **Gate:**
**≤ 10.0 DER** on the nakdimon test (exact run-021 protocol: v4 test,
windowed 1400 bytes, original-separator stitching, K-pass greedy,
seq2seq_der). Shipped baseline: heb-diac-plane-1.0 at 12.48.

## Recipe (run-022)

- Encoder: **byt5-base** (up from small). Plane embedding + per-position
  head unchanged (the ReNikud-validated factorization).
- Corpus: hebrew-v4 (rebuilt locally from
  nakdimon+sefaria+distilled-v1/v2+expanded-v2 per
  `train_hebrew_v4.py`), uploaded to
  `hf://datasets/Interscript/hebrew-v4`, mounted ro.
- Epochs 4 (up from 2), warmup 500, cosine, bs 16 (grad-accum if
  memory-bound on base), lr 1e-4, bf16 autocast, K_PASSES=3 at eval.
- Checkpoints → `hf://buckets/Interscript/isx-training` every 500 steps
  + `labels.sha` resume contract + per-marker eval progress file
  (run-019 lesson) + original-separator stitching (run-021 lesson,
  already correct in the trainer).

## Deliverables

1. `train_hebrew_plane_hf.py` — the HF Jobs port of run-021's trainer
   (paths: /data, /ckpt; volumes; no Modal).
2. Launch on `a100-large -d`; monitor via `hf jobs logs`.
3. Export plane.onnx int8 from the best checkpoint; package plane zip
   (metadata kind=plane + sha256s + classes.json); golden rows vs the
   py runtime.
4. **If gate passes**: full ship chain — release (version = owner
   decision, ASK before publishing), index entry, card, HF mirror.
   If it fails: verdict in RESULTS.md, keep 1.0 shipped.

## Honest risks

byt5-base on 50K units may overfit by epoch 3 — the gate decides, not
the hope. 12.48 → 10 is ambitious; a plateau at ~11 is a documented
negative, not a failure of the campaign.
