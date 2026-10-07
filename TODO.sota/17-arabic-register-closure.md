# WO17 — Arabic register closure (the WikiNews gap attack)

**Why:** WO01 proved the 17.38-vs-2.70 WikiNews-2024 gap is register,
not vocabulary (OOV 1.61%). QCRI's own README confirms their news model
is trained on 5M words of Wikipedia SILVER labels from their BiLSTM —
the same noisy-teacher technique as our r8. Two arms decide which
teacher's silver closes our gap:

- **r8a (self)**: r7 pseudo-labels arwiki (RUNNING, label-v4; keep_frac
  filter at train time). Our teacher: 2.2864 on the harder
  expert-reviewed SadeedDiac-25.
- **r8b (QCRI silver)**: their published Wikipedia_20240420.diac.jsonl
  (95MB, ~5M words, machine-labeled by their BiLSTM ~3% WER). Published
  dataset — same standing as Tashkeela/Nakdimon in our stack; NOT an LLM
  (no-LLM rule intact). Provenance + license recorded in the dataset
  repo.

**Dual-surface gate (both arms, in-job):** SadeedDiac-25 DER (hold the
2.2864 line) AND WikiNews-2024 multiref WER/DER (move 17.38/11.83).
A model that wins OOD but collapses ID is a regression, not a win.
Ship the arm that dominates both; if neither does, negative verdict.
