# Paper alignment (2026-09-01)

The LaTeX paper (rababa docs/paper-arabic/main.tex) — "Data Quality
over Scale: A 30M-Parameter Encoder for Arabic Diacritization" —
predates the campaign's later results. Slotting map for refresh:

| Their section | Current evidence to slot in |
|---|---|
| Data | r7 news-domain adaptation (ID 2.2864 / OOD 17.38-11.83 — dual-surface) |
| Model | r8 aux ablation (morph 2.5793 < IPA 2.6588 < none 2.6775) |
| Experiments | CIs via paired bootstrap; GLM-5.3-Flash row (8.59/8.90); 2.0 client tier 4.82; frontier incl. lite 5.78 |
| Reproducibility | labels sha256 provenance; final_preds.jsonl; five-instance subset phenomenon |

**FLAG for the user (title-level):** the capacity law measured AFTER
the draft — from-scratch 33M collapses at 74.68 full-set even with
clean labels (run-005), and naive stitching fails at both ratios —
sits in direct tension with a "30M-parameter encoder" as a headline
claim, UNLESS the paper's 30M encoder is a differently-trained
data-quality artifact we have not re-measured. The title/thesis needs
the user's decision: refresh the numbers (and possibly the claim), or
reposition as teacher-tier "data quality over scale" (r7 vs frontier
LLMs) with the client tier as the measured boundary.

My paper-a.adoc scaffold (teacher thesis) remains an alternative
split per the 2026-08-29 two-paper decision.
