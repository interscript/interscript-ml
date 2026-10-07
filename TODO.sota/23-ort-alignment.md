# WO23 — ORT generation alignment (the parity root cause, corrected)

WO18 mis-attributed the ruby-vs-py 9.25% skew to "the gem's older ORT".
The gem bundles ORT **1.29.0** — NEWER than the golden-generation
platform (py 1.20.1). The skew is a GOLDEN-PLATFORM version mismatch.

Fix (eval infrastructure only — no library requirement changes):
1. Golden generation pins onnxruntime==1.29.0 (HF cpu job).
2. neural-parity py leg pins the same.
3. Re-dispatch with DEFAULT bounds; if ruby passes, the calibrated
   tiers stand and the per-dispatch override reverts to escape-hatch.
4. ts onnxruntime-node bump (1.27→1.29) is a runtime requirement change
   — owner decision, only if the ts leg misses tolerance.
