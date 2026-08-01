# Architecture decision record: LLM teacher + distilled student

## Context

The legacy rababa model used a Tacotron CBHG encoder-decoder with
attention. Tacotron is a speech synthesis architecture repurposed for
text diacritization. Quantizing it (the original "make it small"
proposal) makes a bad architecture smaller but not better.

## Decision

Replace Tacotron with a two-stage training pipeline:

1. **Teacher**: Qwen3.5-4B-Instruct fine-tuned via LoRA on Tashkeela++.
   The LLM already has strong Arabic language understanding from
   pretraining; fine-tuning teaches it the diacritization task.
2. **Student**: A 6M-parameter character-level transformer distilled
   from the teacher. Small enough to ship in the browser (~6MB ONNX).

## Why this works

- LLMs already model Arabic morphotactics — diacritization is a
  downstream capability, not a from-scratch learning problem.
- Distillation transfers the teacher's softened output distribution
  to the student. The student learns richer signal than gold labels
  alone (the "dark knowledge" of near-misses).
- Character-level (not subword) is the right granularity for
  diacritization: harakat attach to single characters, and a 6MB model
  is fast enough for in-browser inference.

## Alternatives considered

- **Quantized Tacotron.** Rejected: makes a bad model smaller, not
  better. DER barely improves; latency barely drops.
- **ByT5-small fine-tune as teacher.** Plausible fallback if Qwen3.5
  proves too large. ByT5 is character-native, smaller, and already
  proven for text-infilling tasks. Kept as a backup in `models/`.
- **Distillation from a speech model (Nemotron, Whisper).** Rejected:
  those models output speech features, not text. We'd be converting
  between modalities unnecessarily.

## Risks

- LLM teacher may hallucinate plausible-but-wrong diacritization on
  out-of-domain text. Mitigated by training-data augmentation with
  adversarial examples.
- Student may underfit if teacher logits collapse to one-hot.
  Mitigated by KL temperature = 4.0 + alpha = 0.5 (equal CE and KL).
