# WO25 — Arabic news specialist at full scale (the OOD SOTA play)

RUNNING (run-029, a100-large): register-PURE — NO classical tashkeela —
QCRI silver at full scale (900K units ≈ 4.5M words), 2 epochs, r7-init.
r8b proved 1M words of silver moves OOD 6.35 WER; 4.5x the data and
zero register dilution is the honest shot at their 2.70 (in-domain for
them, trained on the same corpus family).

- Gate: WikiNews-2024 multiref <= 5.0 (stretch <= 3.0); ID reported
  as-is (specialist doctrine — two families, never averaged).
- Ship: ara-diac-news-1.0 via export_imf_hf (one command).
