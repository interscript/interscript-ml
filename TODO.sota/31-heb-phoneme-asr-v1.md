# WO31 — Hebrew-tuned phoneme ASR (the v0 lesson, spec'd)

WO30 v0 proved the pipeline but the teacher is the bottleneck: the
universal espeak-cv-ft CTC phonemizes Hebrew poorly. The fix (the
ReNikud recipe's first stage):

1. Fine-tune wav2vec2-lv-60-espeak-cv-ft on FLEURS he_il (~10h,
   ~3.6K train utterances) with espeak-ng reference phonemizations of
   the TRANSCRIPTS as targets (self-supervised bootstrap: espeak reads
   the text; the CTC learns to hear it in real audio).
2. Re-run the stage-2 alignment probe with the tuned ASR (expect PER
   well under 0.395's already-passing mark).
3. Relabel FLEURS + retrain the v1 student; gate unchanged (<0.2393).
4. Only then does ivrit.ai scale ($500) make sense.

Cost: one fine-tune (~2-4 GPU-hours) + relabel + v1 train. Owner
green-light for the block.
