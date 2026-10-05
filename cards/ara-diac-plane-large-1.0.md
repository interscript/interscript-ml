## Benchmark

Full-set SadeedDiac-25 — 1,200 paragraphs, zero skipped, greedy decode
(protocol-matched, Misraj's evaluator):

| model | Total DER | notes |
|---|---|---|
| ara-diac-2.0 (teacher, 580M seq2seq) | 2.29% | best dedicated |
| **ara-diac-plane-large-1.0 (this model)** | **2.74%** | **866 MB int8** |
| Gemini-Flash-2.0 (frontier LLM) | 3.19% | |
| ara-diac-plane-1.0 (plane small) | 3.59% | 219 MB int8 |
| ara-diac-small-2.1 (seq2seq student) | 4.57% | 491 MB int8 |

The second-best dedicated system overall, ahead of Gemini-Flash, and
the best that ships as one self-contained int8 ONNX graph. Morph DER
1.67% (iʿrāb quality); total WER 8.61%.

## Architecture

Plane-factorized diacritization at the ByT5-LARGE encoder scale: a
byte-level encoder-only transformer (no decoder) with a diacritic-plane
embedding and one classification head over 54 haraqat classes per
character position. Decoding is K=2 mask-predict passes — parallel
forwards feeding the predicted plane back; no KV cache, no
autoregressive loop. Skeleton round-trip is byte-exact.

The artifact is a single self-contained int8 ONNX graph in the plane
zip contract (metadata.yaml with member sha256s + plane.onnx +
classes.json), loadable by secryst-py `PlaneModel.from_zip`. The
canonical distribution is the Secryst IMF artifact (sha256-verified);
this repository mirrors those exact bytes.

Provenance: rababa run-019, 2 epochs over 551k unique units. The
epoch-2 number is the corrected measurement — the first epoch-2 eval
silently re-scored epoch-1 predictions and was re-run in full (see the
interscript-ml RESULTS ledger).
