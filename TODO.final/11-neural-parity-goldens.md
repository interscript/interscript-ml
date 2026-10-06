# 11 — Neural cross-runtime parity goldens

**Status: DONE** (2026-10-07, minus the ts plane port) — golden-v2 released (both plane models, py-reference); RUBY GAINED A PLANE RUNTIME (gap the corpus exposed: plane existed only in py) with cycling-fixture specs, 314 examples green; neural-parity.yml workflow green on both legs. CONTRACT FINDINGS: (a) int8 artifacts are not byte-reproducible across onnxruntime builds/platforms, not even for the reference runtime; (b) K-pass decoding cascades single flips into clustered per-row divergence. Hence two tiers: tight (per-row 2% + corpus 0.5%) and smoke (corpus 2%), byte-exact reserved for deterministic artifacts. REMAINDER: port PlaneModel to interscript-ts (py->ruby pattern), add the ts leg to the workflow.
inference (IMF seq2seq + plane K-pass) has no cross-runtime corpus:
each runtime implements decode independently.

## Steps
1. Extend golden-v1 with model-inference vectors: fixed inputs →
   expected outputs per runtime (tolerance policy for fp variants).
2. CI job diffs py/ts/ruby on the neural goldens (the maps-parity job
   pattern).
3. Gate: no runtime release with a neural-golden divergence.
