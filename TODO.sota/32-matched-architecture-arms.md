# WO32 — the matched-architecture program (same corpus MUST win)

Audit: no contamination either side; seq2seq converged at 10.03 → gap
is architecture + teacher. Arms:

- run-033: byt5-large plane + full 900K silver (news-pure, K=3) —
  their recipe class, stronger backbone. Gate: WN-2024 multiref < 10
  (stretch: single digits).
- run-034: literal BiLSTM tagger on the same corpus — reproduces their
  class from scratch. Gate: WN-2024 < 15 validates the class at our
  scale; comparison anchor.
- SOTA-stack layer above both: K-pass conditioning, byte-pretrained
  backbones, preserve mode (already in the plane family).

Ship rule: whichever arm dominates WikiNews-2024 becomes the
ara-diac-news successor; seq2seq line is closed (converged at 10.03).
