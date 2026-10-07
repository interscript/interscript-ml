## Benchmark

Nakdimon test set, greedy decode, `seq2seq_der` — the exact protocol of
the shipped seq2seq student:

| model | Total DER | notes |
|---|---|---|
| **heb-diac-plane-1.0 (this model)** | **12.48%** | **219 MB int8** |
| heb-diac-1.1 (seq2seq student) | 16.44% | larger artifact |
| DictaBERT (fine-tuned BERT baseline) | ~35.6% | per-protocol reference |

−3.96pp (24% relative error reduction) over the shipped student at the
plane's on-device economics. Median per-example DER 9.8%;
223,872 scored positions.

## Architecture

Plane-factorized nikud restoration: a ByT5-small-class byte-level
encoder (no decoder) with a nikud-plane embedding and one
classification head per character position over 135 order-preserving
nikud classes. Decoding is K=2 mask-predict passes — fully parallel
forwards with the predicted plane fed back; no KV cache, no
autoregressive loop. Skeleton round-trip is byte-exact; nikud cluster
order is preserved as written (the corpus has no uniform canon).

The artifact is a single self-contained int8 ONNX graph in the plane
zip contract (metadata.yaml with member sha256s + plane.onnx +
classes.json), loadable by `interscript[ml]` `PlaneModel.from_zip`.
The canonical distribution is the sha256-verified release artifact;
this repository mirrors those exact bytes.

Provenance: interscript-train run-021 — v4 combined corpus (nakdimon +
sefaria + distilled v1/v2 + expanded_v2), 50,303 units, 2 epochs. The
verdict ledger records a disclosed harness correction (window-stitching
bug; first read 32.64%, corrected 12.48%).
