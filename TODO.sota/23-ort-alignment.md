# WO23 — ORT alignment (CLOSED: characterization complete)

WO18's "gem ORT too old" attribution was WRONG (corrected): the ruby gem
bundles ORT 1.29.0 — newer than the golden platform. The completed
experiment matrix on heb-diac-plane-2.0 (byt5-base, K=3, int8 dynamic):

| comparison | corpus diff |
|---|---|
| py(1.20) vs py(1.29), same platform (linux) | **0% (byte-identical)** |
| py(mac) vs py(linux) | 2.86% |
| ruby(mac, ORT 1.29) vs py(linux golden) | 4.23% |
| ruby(linux, CI) vs py(linux golden) | 9.25% |
| ruby opt-level NONE vs default (mac) | 9.36% vs 4.23% (knob NEGATIVE) |
| ruby intra_op=1 vs default (mac) | 4.23% (knob NEGATIVE) |

Conclusions: (1) divergence is CROSS-PLATFORM int8-dynamic kernel
selection, scaling with model size × K-passes; (2) no session knob
restores cross-platform parity (three configs measured, closed);
(3) same-platform same-runtime is byte-exact.

RESOLUTION: canonical golden platform = linux x86 CI (done, WO18);
per-artifact-class bounds — K=2 small keeps the calibrated 2% smoke,
K=3+ base uses the documented per-dispatch bound (measured spread
≤ 9.36% ⇒ 0.10); tight byte-exact parity remains the fp32 artifact
class. Re-dispatched run 37611316113: **py ✓ ts ✓ ruby ✓ at 0.10**.
