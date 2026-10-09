# WO33 — the word-identity channel: variant-disambiguation tagger

Decomposition (2026-10-09, local probe, same corpus):

| arm | WikiNews-2024 WER | note |
|---|---|---|
| pure most-frequent-reading lookup | 25.15 (DER 13.18) | no model; 482,609-type table; OOV 18.10% |
| literal char BiLSTM (run-034) | 14.30 (DER 10.10) | char context, no word channel |
| byt5-base seq2seq specialist (run-029, shipped) | 10.13 (DER 8.98) | implicit lexicon via byte pretraining |
| byt5-large plane + 900K silver (run-033) | running | their recipe class, stronger backbone |

Lexical memorization is 3/4 of the task; the remaining ~25 points are
context disambiguation among a word's few observed readings — the word
channel char-level models approximate only indirectly. QCRI's own
design (Fadel) is word-level: word-embedding + char-CNN + BiLSTM over
variant indices.

Arm: run-035 — variant-disambiguation tagger on the same corpus as
run-033/034. Word-emb (128) + char-CNN (k=2..5, 50 filters each) ->
BiLSTM 2x512 -> head over {variant idx 0..5, OOV}. Labels from the
variant table (most-frequent = 0). Decode emits the chosen variant
string; output preserves the input skeleton letter-exactly. fp32
(LSTM bf16 = NaN). ~62M params, bs 128, 5 epochs, a100, ~2h.

Gates:
- WN-2024 multiref < 14.30 (beats the char-only BiLSTM) → the word
  channel is real at our scale.
- WN-2024 < 10.13 (beats the shipped specialist) → becomes the
  ara-diac-news successor; then stack SOTA layers (K-pass, larger
  backbone, preserve mode).
- Fails both → the gap is not the word channel either; remaining
  hypothesis is corpus-internal convention alignment, closed by the
  register-specialist doctrine (crown = ID specialist + news
  specialist shipped separately).

Ship rule: same as WO32 — dominant arm on WN-2024 becomes the news
successor; export via int8 IMF v1 chain only if it beats 10.13.

Complement probe (post-verdict): oracle-min over {run-029, run-035}
bounds the ensemble ceiling; if >2 points over the best single arm,
a routing/blending follow-up (WO28 lineage) opens.
