# WO34 — the synthesis arm: char-level BiLSTM + word-embedding feature

run-035 (WO33) verdict, 2026-10-10: word-level variant-disambiguation
tagger scored **18.07 / 11.44** (WikiNews-2024 multiref) and Sadeed
Total DER **18.75** — both gates FAILED (<14.30, <10.13). Diagnostic
value: the word-only design emits bare (undiacritized) words for the
18.1% OOV types, which alone collapses ID DER; the char-only BiLSTM
(run-034, 14.30/10.10) composes unseen words but has no word-identity
channel. Each half in isolation loses to the shipped specialist
(10.13/8.98).

Convention-style hypothesis CLOSED same day (local probe): marked-letter
density is 0.8127 (WikiNews-2024 refs, first-alt) / 0.7625
(wikinews2014 gold) / 0.7432 (news silver) — same full-diacritization
family; no partial-vs-full style mismatch to harvest.

The arm: run-036 — Fadel's actual joint design at our scale. Char-emb
128 ⊕ per-char word-embedding (bare-form vocab, isalpha core,
300K cap) → 3×BiLSTM-512 → per-position plane-combo head. Same corpus
as run-033/034/035, fp32, bs 64, 6 epochs, a100.

Gates:
- WN-2024 multiref < 14.30 → word channel validated at char level;
  hybrid becomes the stacking backbone.
- WN-2024 multiref < 10.13 → news-successor candidate; stack SOTA
  layers (K-pass conditioning, larger corpus, preserve mode) and run
  the IMF v1 export chain.
- ≥ 14.30 → word-identity is NOT the residual gap at our data scale;
  remaining hypotheses: tuning maturity (lr/epoch sweep), their
  evaluator-side conventions (closed), or data scale beyond 900K
  silver units. Doctrine stands.

Staged follow-ups (armed post-verdict):
1. Variant-constrained decode — DONE, closed (2026-10-10): ID-only
   tweak (Sadeed 5.90→5.75), multiref bit-identical to free decode
   (13.5268/9.9211). Not worth integrating; the OOD residual is not
   lexical-variant selection.
2. Oracle-min complement probe — DONE, CLOSED NEGATIVE (2026-10-10):
   regen of run-029 reproduced 10.032/8.9521 exactly; oracle over
   {specialist, hybrid} = 9.33/8.77 — 0.71 WER over best solo,
   below the >2 threshold. Errors overlap; routing opens nothing.
3. run-037 (STAGED, script uploaded + hardened): run-036's
   architecture with the scratch word channel replaced by pretrained
   fastText cc.ar.300 vectors (in-job download, coverage-reported,
   trainable init; ft_init cache + step-checkpoint resume) — the
   lexical-density hypothesis: our word channel is undertrained on
   ~3M words, not wrong; Fadel-class systems ride on billion-word
   vectors. Same gates. Launches when a GPU slot frees regardless of
   run-036's outcome (independent axis: scratch vs pretrained).
4. Tuning-maturity sweep — READY: hybrid trainer now exposes
   HYB_LR / HYB_HID env knobs; grid {lr 1e-3/5e-4} × {hid 512/768} ×
   {epochs 6/10} queued after the three primary arms read out.
5. If run-033 (byt5-large plane) lands <10, race it against run-036/37
   gates; the dominant arm takes the successor slot.
