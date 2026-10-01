# 02 — Paper/docs: 2026-09-30 arXiv sweep citations

Status: SPECIFIED (2026-10-01)

Sweep window Aug 26 → Sep 30, 2026 (cs.CL/cs.LG: distillation,
byte-level, diacritization, Muon). Four findings change what the
papers should cite. Protocol rule stands: never quote cross-protocol
numbers.

## Citations to add (paper.adoc + RESULTS.md)

1. **arXiv 2609.12303 — "Breaking the Token Ceiling: Distilling
   Smaller, Stronger Byte Models"** (Meta/FAIR). First large-scale
   distillation × tokenization study (~1B params, up to 1T bytes):
   distilled byte students start worse but surpass token students with
   compute — higher asymptote (+4% predicted), 6× data efficiency,
   256-vocab avoids top-k logit truncation. External scaling-law
   validation of our byte-student lineage (ByT5-style, byte+3 table).
   Where: background/related work + one sentence in the frontier
   discussion.
2. **arXiv 2609.37510 — MAESTRO ("From Dissonance to Orchestration")**.
   Teacher intervention in on-policy distillation adds off-policy
   load; always-on intervention yields diminishing returns; adaptive
   disagreement-gated takeover is the fix. Cite as the mechanistic
   account of our measured GKD negative (6.0036) — our always-on GKD
   is the maximum-intervention point of their axis. Do NOT re-run
   (ledger: student-side levers closed).
3. **arXiv 2608.27729 — "Below the Noise Floor"** (bimodal seed
   collapse in small-model KD). Per-seed σ 2.8–48.7pp; single-seed KD
   gains <5pp are unresolvable; 3/7 KD variants collapse bimodally.
   Cite in the evaluation-methodology discussion: strengthens our
   paired-bootstrap/full-set discipline AND adds the honest caveat
   that our single-run arm verdicts (e.g. headwise Muon +0.2267pp)
   measure prediction-resampled variance, not seed variance.
4. **arXiv 2609.10153 — YallaMorph** (EMNLP 2026). Cite as the r8
   teacher's data lever (morphological aux coverage) and as evidence
   the field's Arabic-morphology energy moved to LLM evaluation, not
   text-diacritization SOTA.

## Also record (no citation needed)

- No new text-only SadeedDiac-25 competitor in the window; KSAA-2026
  speech-diacritization winner (2605.25928, 23.26% WER) is a speech
  modality — not protocol-comparable; noted in RESULTS.md sweep entry.

## Steps

1. [ ] RESULTS.md: "2026-09-30 sweep" entry (4 citations + the
       no-new-competitor note).
2. [ ] paper.adoc: related-work sentences for 2609.12303 + MAESTRO;
       methodology caveat sentence citing 2608.27729; YallaMorph in
       the discussion where the r8 teacher lever is named.
3. [ ] PR to interscript/interscript-ml (branch off default, no AI
       attribution, explicit-path staging).

## Result

(to be written only after merge)
