# ara-diac-2.0

Arabic diacritization (haraqat restoration), ByT5-base, 580M parameters.

## What changed from 1.0

ara-diac-1.0 (r6) was trained with a morphological auxiliary task
(qalsadi iʿrāb labels) on top of the paragraph-context r5 lineage.
ara-diac-2.0 (r7) adds a news-domain adaptation stage: 13,986
teacher-labeled news units mixed at 0.85% with the r5-units anchor,
plus 400 gold WikiNews-2014 lines, initialized from r6.

The result is a full sweep over 1.0 — both surfaces improved:

| Surface | ara-diac-1.0 (r6) | ara-diac-2.0 (r7) |
|---|---|---|
| SadeedDiac-25, full 1,200, windowed zero-skip, greedy | DER 2.5793 / Morph 1.5317 | **DER 2.2864 / Morph 1.3343** |
| WikiNews-2024 multi-ref (QCRI protocol, full) | WER 19.8191 / DER 12.4613 | **WER 17.3794 / DER 11.8273** |

The earlier lineage's in-domain/out-of-domain trade-off is gone: the
news mix improved in-domain substantially, not just OOD.

## Positioning

Best dedicated (task-trained, runnable-locally) model measured on
SadeedDiac-25 under the zero-skip Misraj protocol — behind only
Claude-3.7-Sonnet's published 1.3941, ahead of GLM-5.2 (2.6911
zero-skip reproduction), Gemini-Flash-2.0 (3.1926), and GPT-4 (3.8645),
at 580M parameters against Sadeed-1.5B's 7.2915.

## Contract

IMF v1 zip: byte tokenizer (id = byte + 3, trailing EOS), KV-cache
decoder, opset 14. Greedy decode is the runtime protocol. int8 zips
keep the tied head in fp32 (the margin-gate fix); every precision is
gated by CER parity AND the argmax-flip margin budget.

Provenance: rababa `train_arabic_r7.py`, run `rababa_arabic_byt5/run-007-news`;
verdict tables in `rababa/docs/RESULTS.md` (r7 sections). BSD-3-Clause.
