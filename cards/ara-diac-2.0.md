## Benchmark

SadeedDiac-25 — full 1,200-paragraph set, Misraj's evaluator, zero
skipped examples, greedy decode (protocol-matched published numbers
only):

| system | Total DER | kind |
|---|---|---|
| Claude-3.7-Sonnet | 1.39% | frontier LLM |
| **ara-diac-2.0 (this model)** | **2.29%** | dedicated, 580M |
| GLM-5.2 | 2.69% | frontier LLM |
| Gemini-Flash-2.0 | 3.19% | frontier LLM |
| GPT-4 | 3.86% | frontier LLM |
| Sadeed-1.5B | 7.29% | leaderboard baseline |

Word-final iʿrāb endings (the hardest component): 1.33% DER.
Out-of-domain generalization (WikiNews-2024 multi-reference):
17.38 WER / 11.83 DER.

## Architecture

ByT5-large-class seq2seq transformer (580M parameters), byte-level
input/output (no tokenizer — id = byte + 3). Trained on classical and
modern Arabic running text with a news-domain adaptation stage over a
morphology-auxiliary curriculum. The canonical distribution is the
Secryst IMF artifact (sha256-verified, parity-gated against the
PyTorch reference); this repository mirrors those exact bytes.
