# WO01 — WikiNews-2024 multiref reconciliation (Arabic)

**Priority:** P1 · **Cost:** hours · **Gate:** a protocol-matched statement
of where our models stand on WikiNews-2024, in RESULTS.md.

## Problem

Our OOD number on WikiNews-2024 is 17.38 WER / 11.83 DER while the EMNLP
2025 QCRI paper (Mohamed & Mubarak, 2025.emnlp-main.846) reports
**2.70% WER** with a BiLSTM on the same benchmark name. A 6× gap on the
same named benchmark is almost certainly protocol divergence, not model
divergence — but we cannot quote ANY WikiNews number until the delta is
explained. Suspects:

1. Different test split (our slice may be the harder OOD half).
2. Different reference policy (multiref semantics: correct if the word
   matches ANY reference; ref count per word may differ).
3. Different WER definition (undiacritized word identity vs
   fully-diacritic-exact match).
4. Normalization (ta-marbuta/ha, alef variants, tatweel, punct).
5. Different model class being scored (our number may come from a plane
   or seq2seq student, theirs from a task-dedicated BiLSTM).

## Deliverables

1. Obtain the official WikiNews-2024 release (paper's repo/QCRI) —
   test paragraphs + per-word references + (if published) their scorer.
   Record provenance + sha256 of the files we use.
2. Reconcile harnesses: run `interscript-train/eval_wikinews_multiref.py`
   (ours) and their protocol on THE SAME predictions; diff every metric
   definition line-by-line; write a delta table.
3. Score, under the reconciled protocol: released ara-diac-2.0,
   ara-diac-plane-1.0, ara-diac-plane-large-1.0 (+ r7 if weights are
   reachable). Greedy, no conditioning, runtime protocol.
4. RESULTS.md entry: one table, protocol column explicit, plus a
   one-paragraph verdict (mismatch explained or real gap owned).

## Out of scope

Retraining; changing the shipped artifacts; quoting any WikiNews number
externally before this WO lands.
