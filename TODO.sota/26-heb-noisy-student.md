# WO26 — Hebrew noisy student (defend and extend the crown)

8.18 is text-only SOTA; byt5-large showed capacity is NOT the lever at
50K gold units — DATA is. Stage 1 (RUNNING): plane-2.0 pseudo-labels
40K hewiki windows with per-position margins (NO LLM). Stage 2:
gold v4 (50K) + pseudo (cap 40K, keep_frac >= 0.9) on byt5-base.

- Gate: artifact-level DER < 8.18 else negative.
- Lesson applied: capacity arms are closed; only data moves Hebrew.
