# 11 — Neural cross-runtime parity goldens

**Status: pending.** Goldens cover maps (7388/7388 parity). Model
inference (IMF seq2seq + plane K-pass) has no cross-runtime corpus:
each runtime implements decode independently.

## Steps
1. Extend golden-v1 with model-inference vectors: fixed inputs →
   expected outputs per runtime (tolerance policy for fp variants).
2. CI job diffs py/ts/ruby on the neural goldens (the maps-parity job
   pattern).
3. Gate: no runtime release with a neural-golden divergence.
