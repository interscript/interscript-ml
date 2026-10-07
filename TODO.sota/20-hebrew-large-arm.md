# WO20 — Hebrew byt5-large arm (run-026)

RUNNING (a100-large): byt5-large encoder, 3 epochs (large overfits 50K
units faster), K=4, bs4/accum4. Same gate protocol as WO04.

- Gate: artifact-level DER < 8.18 (beat base) — else negative verdict.
- Ship: export int8, package, 3-leg parity (goldens on linux/ORT-1.29
  per WO23), owner-confirmed version.
- Honest risk: 1.2B params on 50K units may overfit by epoch 2; the
  per-epoch gate decides.
