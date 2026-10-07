# TODO.sota — 2026 SOTA campaign (client-usable quality)

Derived from the 2026-10-05 SOTA sweep (Arabic / Hebrew / Thai).
Goal: close the measured gaps to the 2026 frontier on all three
languages while keeping every artifact client-usable (int8, sha-verified,
multi-runtime, versioned).

## Substrate: HF (NOT Modal)

All training and eval jobs run on **Hugging Face Jobs** (`hf` CLI, v1.26+):

```bash
hf jobs run --flavor a100-large -d \
  -v hf://datasets/Interscript/hebrew-v4:/data:ro \
  -v hf://buckets/Interscript/isx-training:/ckpt:rw \
  python:3.12 python train.py
```

- Datasets: `hf://datasets/Interscript/<name>` (mounted read-only).
- Checkpoints/artifacts: `hf://buckets/Interscript/isx-training` (read-write).
- Detached always (`-d`); poll via `hf jobs ls` / `hf jobs logs`.
- Modal is retired for compute. It may be used ONLY to export corpora
  off Modal volumes during the one-time migration (#44).

## Standing rules that shape this campaign

- **NO LLM TEACHERS** (memory law, 2026-08): never use Claude/GPT/
  Gemini/any LLM as a label source for haraqat/niqqud/Thai IPA.
  WO06 is therefore noisy-student/self-ensemble — NOT the
  "LLM-ensemble teacher" phrasing used in the sweep notes.
- Versions/releases/cards/index are owner decisions at ship time;
  specs define gates, not version numbers.
- Data channel for shipped artifacts stays GitHub Releases + sha256;
  HF dataset repos here are TRAINING inputs, not the shipping channel.
- PRs only; no tags without explicit instruction.

## Work orders

| WO | File | Priority | Depends | Status |
|----|------|----------|---------|--------|
| 01 | 01-wikinews-multiref-reconciliation.md | P1 | — | pending |
| 02 | 02-user-diacritics-preservation.md | P1 | — | pending |
| 03 | 03-thai-hybrid-latency.md | P1 | — | pending |
| 04 | 04-hebrew-plane-scaleup.md | P2 | #44 | pending |
| 05 | 05-external-benchmarks.md | P2 | 01, 07 | pending |
| 06 | 06-arabic-r8-noisy-student.md | P2 | #44 | pending |
| 07 | 07-nikud-ipa-rule-layer.md | P3 | — | pending |
| 08 | 08-thai-small-tier.md | P3 | #44 | pending |
| 09 | 09-hebrew-audio-pseudolabel-probe.md | P3 (parked probe) | — | pending |

## Schedule

Wave 1 (local, days): WO01, WO02, WO03, WO07 + infra #44.
Wave 2 (HF training, launched together): WO04 (a100-large), WO06
(a100-large), WO08 (l4x1) — run concurrently, watch via `hf jobs ls`.
Wave 3: WO05 tables after 01+07 land; WO09 probe only.
