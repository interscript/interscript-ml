## Benchmark

Full-set SadeedDiac-25 — 1,200 paragraphs, zero skipped, greedy decode
(protocol-matched, Misraj's evaluator):

| model | Total DER | notes |
|---|---|---|
| ara-diac-2.0 (teacher, 580M seq2seq) | 2.29% | best dedicated |
| **ara-diac-plane-1.0 (this model)** | **3.59%** | **219 MB int8** |
| ara-diac-small-2.1 (seq2seq student) | 4.57% | 491 MB int8 |

On-device frontier on all three axes: -45% size and -67% CPU latency
(~1,145 ms/window vs ~3,459 ms) against the shipped seq2seq int8
student, at 0.98pp better DER.

## Architecture

Plane-factorized diacritization: a ByT5-small-class byte-level encoder
(no decoder) with a diacritic-plane embedding; one classification head
predicts a haraqat class per character position over 54 classes.
Decoding is K=2 mask-predict passes — parallel forwards with the
predicted plane fed back as input; no KV cache, no autoregressive
loop. Skeleton round-trip is byte-exact (plane marks close into the
preceding base letter; leading marks are preserved via a pre-marker).

The artifact is a single self-contained int8 ONNX graph in the plane
zip contract (metadata.yaml with member sha256s + plane.onnx +
classes.json), loadable by secryst-py `PlaneModel.from_zip`. The
canonical distribution is the Secryst IMF artifact (sha256-verified);
this repository mirrors those exact bytes.

Provenance: rababa run-018 — 551k unique units, 2 epochs, checkpoint
sha256 26f49bd0ebbf3ca93cebf2f36bdd8b488f6bb17e99e3f61f9613e53d1852862.
