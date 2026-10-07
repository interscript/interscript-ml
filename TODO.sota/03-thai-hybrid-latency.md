# WO03 — Thai hybrid fast-path + latency table

**Priority:** P1 · **Cost:** days · **Gate:** published latency × accuracy
table; hybrid within 1.2× of dict-only latency and within +0.3pp PER of
model-only on our benchmark.

## Why

FastThaiG2P (arXiv 2608.12814) owns the latency narrative (0.15 ms/utt)
but is 26.9% PER on its own set — dictionary+rules. We own quality
(2.85% PER, greedy, runtime protocol) at neural latency. The client
answer is not a race but a hybrid tier: dictionary fast-path for
in-vocab words, our neural model as the OOV fallback (their own profile:
58% of latency is fallbacks — that fallback is us).

## Design

1. **Lexicon**: word→IPA from Kaikki Thai (the same lineage as training
   data). Multiple readings per word → keep the most frequent; store
   count provenance. Build script emits `tha-lexicon.jsonl`
   (word, ipa, count) — a data artifact, versioned with the model.
2. **Fast-path**: Thai has no spaces → longest-prefix (maximal-munch)
   match over the lexicon; unmatched spans go to the neural model
   (windowed 1400-byte protocol, unchanged); spliced back.
3. **Bench** on `thai-kaikki-g2p/test.jsonl` (1,219 sentences, CPU):
   - dict-only PER + latency/utt
   - model-only PER + latency/utt
   - hybrid PER + latency/utt (+ OOV-fallback rate)
4. **RESULTS.md**: the table, protocol column explicit, one-paragraph
   positioning vs FastThaiG2P (never quote their PER against ours —
   different set; quote ours and cite theirs as self-reported).

## Deliverables

- `interscript-py`: hybrid translate path + lexicon builder script.
- Artifact: lexicon released alongside the model (GitHub Releases +
  sha256, per data-channel law).
- Bench JSON + RESULTS.md table.

## Non-goals

PyThaiNLP dependency (our maximal-munch over Kaikki covers it);
retraining anything; ts/ruby ports of the hybrid (py first, note ports).
