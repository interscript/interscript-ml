# WO06 — Arabic r8: noisy-student / self-ensemble (HF training)

**Priority:** P2 · **Cost:** teacher labeling (~hours on GPU) + training
(~6–8 h `a100-large`) · **Gate:** **beat 2.2864 DER** (r7/ara-diac-2.0)
on SadeedDiac-25 under the exact zero-skip haraqat-projected protocol;
stretch ≤ 2.0.

## PIVOT (standing rule)

The sweep notes floated an "LLM-ensemble teacher". **Forbidden** —
memory law: NEVER use Claude/GPT/Gemini/any LLM as a diacritization
label source (hallucinated haraqat bake into the student). r8 uses only
signals we control:

1. **Self-training (noisy student):** our best dedicated model
   pseudo-labels UNLABELED Arabic text (`data/arwiki`, 115 MB local);
   keep predictions where the model is confident (per-position class
   margin above threshold; span-keep, not char-spaghetti).
2. **Mixture:** gold tashkeela-full + pseudo-arwiki (cap pseudo at ~50%
   of gold tokens; ablate the ratio only if the first run misses the
   gate).
3. **Self-ensemble (optional second arm):** per-position vote across
   OUR OWN checkpoints (r6+r7 lineages + plane-large) as pseudo-labels —
   model diversity without external teachers.
4. Excluded: yallamorph-aux (run-009 measured NEGATIVE — do not re-add).

## Deliverables

1. `pseudo_label_arabic_hf.py` — teacher inference + margin filter
   (ONNX int8 teacher on GPU is fine; greedy; emit arwiki-pseudo.jsonl
   with kept-token stats).
2. `train_arabic_r8_hf.py` — the proven dedicated seq2seq lineage
   (byt5-small, IMF contract), mixture loader, same eval protocol as
   run-019's corrected harness (per-marker progress, exact separators).
3. HF Jobs launch + monitoring; export/package per release system.
4. **If gate passes:** ship chain with owner-confirmed version; else
   RESULTS.md verdict with kept-token stats and margin-threshold
   sensitivity.

## Honesty anchors

Single-seed caveat (2608.27729): a gain under ~0.15 DER is within seed
noise — mark it as such rather than claiming SOTA movement. Keep-token
rate is a first-class reported number.
