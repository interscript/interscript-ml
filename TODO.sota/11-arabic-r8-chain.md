# WO11 — Arabic r8 chain (WO06 remainder)

In flight on HF (a100-large): label job (r7 teacher over arwiki,
margin>=1nat frac>=0.9) -> train job (r7-init, gold 400K + pseudo 200K,
WINDOW=600 protocol, in-job SadeedDiac-25 gate).

1. On LABEL_DONE: launch train_arabic_r8_hf.py (same mounts).
2. On EVAL_DONE: read DER vs gate 2.2864.
3. PASS -> export int8 IMF artifact + ship chain (ara-diac-2.1) + index
   bump; FAIL -> RESULTS.md negative verdict with keep-rate stats +
   margin-threshold sensitivity (single-seed caveat applies below 0.15).
