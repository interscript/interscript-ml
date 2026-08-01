# Architecture decision record: direct supervised training on gold corpus

## Context

The legacy rababa model used a Tacotron CBHG encoder-decoder with
attention. Tacotron is a speech synthesis architecture repurposed for
text diacritization. Two issues with it:
- Wrong inductive bias (speech features ≠ character-level text)
- Hard to ship in a browser (large, slow)

An earlier version of this ADR proposed replacing Tacotron with an
"LLM teacher + distilled student" pipeline (Qwen3.5-4B teacher → 6M
student). **The LLM teacher was wrong.** This document records the
correction and the actual chosen architecture.

## Why the LLM teacher was wrong

For diacritization and transliteration, **the corpus is the
authority, not the LLM.**

- **rababa_arabic**: Tashkeela++ (~2M verses) and OpenDiacritizer are
  scholar-annotated Arabic with full harakat. Human-authored gold.
- **rababa_hebrew**: SNA Nikud is rabbinic-school-trained annotator
  output. Authoritative.
- **secryst_thai_ipa**: Wiktionary IPA pairs are linguist-authored.
  Authoritative.

An LLM teacher fine-tuned on the same corpus cannot add information
the corpus doesn't already have. It can only:
1. Re-express existing labels (no information gain)
2. Introduce its own errors (information loss)
3. Cost ~$15-25/task for the privilege

## Chosen architecture (two tiers)

### Tier 1 — Default (MVP): direct supervised training

A moderately-sized (~30M param) character-level transformer, fine-tuned
on the gold corpus. This is what ships first.

```
   ┌──────────────────────────────────────────────┐
   │  (optional) ~10GB unlabeled Arabic text      │
   │  self-supervised MLM pretrain                │
   │  → checkpoint                                │
   └────────────────────┬─────────────────────────┘
                        │ init
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Tashkeela++ gold labels (supervised CE)     │
   │  → 30M-param char transformer                │
   └────────────────────┬─────────────────────────┘
                        │ ONNX export + int8 quant
                        ▼
   ┌──────────────────────────────────────────────┐
   │  ~8 MB browser model, ~5% DER                │
   └──────────────────────────────────────────────┘
```

Cost: ~$10-15/task on 1× A100 (3-5h wall time).
DER target: 4-6%.

### Tier 2 — Optional (quality): distill from a fine-tuned teacher

If Tier 1 quality isn't enough, distill from a teacher that is *also*
trained on the gold corpus:

```
   ByT5-base (pretrained on multilingual text)
        │
        │ supervised fine-tune on Tashkeela++ (gold corpus)
        ▼
   Teacher checkpoint  (~600 MB, ~3% DER)
        │
        │ distill into student (gold CE + teacher KL)
        ▼
   30M student (~30 MB fp32 / 8 MB q8)
   ~4% DER, browser-friendly
```

**Critical**: the teacher is fine-tuned on Tashkeela++ too. It has no
opinions beyond what the gold corpus teaches it. Distillation here
transfers *learned representations* (the teacher's embedding space,
attention patterns), not authority. That's the legitimate use of
distillation — as a *compression* strategy, not as a *labeling*
strategy.

Cost: ~$25-35/task (ByT5-base fine-tune + distill on 1× A100).
DER target: 3-4%.

## Why ~30M params (not 6M, not 300M)

| Size | DER (no pretrain) | DER (with pretrain) | Browser (q8) | Browser (q4) |
|---|---|---|---|---|
| 6M | 8-10% | 5-7% | ~2 MB | ~1 MB |
| **30M** | **6-8%** | **4-6%** | **~8 MB** | **~4 MB** |
| 100M | 4-5% | 3-4% | ~25 MB | ~13 MB |
| 300M (ByT5-small) | 3% | 2-3% | ~75 MB | ~40 MB |
| 600M (ByT5-base) | 2-3% | 2% | too big for browser | ~80 MB |

30M is the sweet spot for browser-targeted diacritization: small
enough to ship (~4-8MB quantized), large enough to hit useful DER.
ByT5-small/base stay available as Tier-2 teachers or for
server-side deployment.

## Why character-level (not BPE/subword)

- Harakat attach to single characters. BPE merges hide the per-char
  resolution that's the entire task.
- Vocab is small (~50-100 chars) so embedding tables don't dominate
  parameter count.
- Inference is character-by-character; matches how ONNX runtime
  handles dynamic shapes in the browser.

## Alternatives considered (and why rejected)

- **Quantized Tacotron CBHG** (legacy): wrong inductive bias for text.
- **LLM teacher (Qwen3.5-4B + LoRA)**: rejected above. Corpus is
  authoritative; LLM adds noise + cost.
- **mT5/BERT fine-tune directly**: BPE tokenization loses char-level
  signal. Possible workaround with vocab surgery, but messy.
- **Encoder-only with per-char classification**: simpler than
  encoder-decoder but typically 1-2pp worse DER because it can't model
  target-side dependencies. Could revisit if browser budget is tight.

## Cost comparison

| Path | Stages | Wall time (1× A100) | Est. cost | DER |
|---|---|---|---|---|
| LLM teacher + distillation (rejected) | 3 | ~16-22h | ~$20-30 | ~5% |
| **Tier 1: direct supervised** | 1-2 | ~3-5h | ~$10-15 | 4-6% |
| Tier 1 + self-pretrain | 2 | ~6-10h | ~$15-25 | 4-5% |
| Tier 2: distill from ByT5-base teacher | 3 | ~10-15h | ~$25-35 | 3-4% |

Tier 1 is both cheaper and better than the rejected LLM-teacher path.

## Risks

- **Domain mismatch.** Tashkeela++ is Quran-heavy; the model may
  produce unexpected diacritization on modern news. Mitigation: add
  OpenDiacritizer + Wikipedia Arabic to the training mix; document
  the bias in the model card.
- **30M may underfit on hard cases.** Consonant clusters in modern
  Arabic can be ambiguous. Mitigation: monitor per-category DER; bump
  to 100M if needed (one config edit, no code change).
- **Quantization loss.** int8 dynamic quant usually adds <0.3pp DER;
  int4 can add 1-2pp. Mitigation: ship int8 by default; int4 only for
  size-critical mobile.

## When the LLM teacher WOULD be justified (so we know what we're declining)

- **Noisy labels** (sentiment, emotion) where teacher agreement > single annotator
- **Generative tasks** (story, code, dialogue) where there's no correct answer
- **Domain adaptation with synthetic augmentation** — Tashkeela++ is
  Quranic-heavy; an LLM could *augment* (not replace) gold with
  news-domain synthetic labels
- **Zero labeled data** — pure zero-shot

None of these describe diacritization in 2026. The framework's
``DistillTrainer`` remains available for Tier-2 distillation from a
fine-tuned teacher (not an LLM with prior opinions).

## What this ADR replaces

The previous version proposed an LLM teacher pipeline. That proposal
is **superseded**. References to "teacher fine-tune" in legacy docs
(`TODO.rababa/02-teacher-model.md`) are kept as historical record but
no longer reflect the implementation path.
