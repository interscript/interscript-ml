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
| 01 | 01-wikinews-multiref-reconciliation.md | P1 | — | DONE (RESULTS #257) |
| 02 | 02-user-diacritics-preservation.md | P1 | — | DONE (py#28, ruby#801) |
| 03 | 03-thai-hybrid-latency.md | P1 | — | DONE (py#31, RESULTS #258) |
| 04 | 04-hebrew-plane-scaleup.md | P2 | #44 | gate CLEARED — artifact DER 8.18 (int8, runtime protocol); release pending owner version |
| 05 | 05-external-benchmarks.md | P2 | 01, 07 | DONE (RESULTS #259; MILIM awaiting release) |
| 06 | 06-arabic-r8-noisy-student.md | P2 | #44 | jobs LAUNCHED (label + train, HF queue) |
| 07 | 07-nikud-ipa-rule-layer.md | P3 | — | DONE (py#29, py#30) |
| 08 | 08-thai-small-tier.md | P3 | #44 | job LAUNCHED (run-025, l4x4) |
| 09 | 09-hebrew-audio-pseudolabel-probe.md | P3 (parked probe) | — | DONE as parked probe (script only) |
| 10 | 10-heb-plane-2-release.md | P1 | 04 | DONE (release, index-v9, 3 pin bumps, card, golden, HF mirror #28) |
| 11 | 11-arabic-r8-chain.md | P1 | — | jobs RUNNING (a100) |
| 12 | 12-thai-tiny-contract.md | P2 | — | NEGATIVE (caveated) — PER 2038%; sub-10MB closed |
| 13 | 13-thai-lexicon-release.md | P2 | 03 | DONE (tha-lexicon-kaikki-1.0 + sha, CC BY-SA) |
| 14 | 14-ts-plane-port.md | P2 | — | DONE (ts#99/#100/#101, CI leg models#265) |
| 15 | 15-api-edge.md | P2 | 14 | DONE (api#29: edge-first /v1/infer, kind in index) |
| 16 | 16-parking-lot.md | — | — | parked (not now) |
| 17 | 17-arabic-register-closure.md | P1 | 11 | dose-response MAPPED (RESULTS wave 2); r8d knee-seeking RUNNING; specialist option staged |
| 18 | 18-parity-runs.md | P1 | 10 | pending |
| 19 | 19-lexicon-loader.md | P2 | 13 | DONE (py#33) |
| 20 | 20-hebrew-large-arm.md | P1 | 04 | NEGATIVE — 8.48 vs base 8.18; base stays shipped |
| 21 | 21-arabic-plane-transfer.md | P2 | 17 | VERDICT: OOD −3.14 WER, ID +1.66 (Pareto) — specialist candidate staged |
| 22 | 22-runtime-releases.md | P1 | 14 | PREPARED — owner-gated versions |
| 23 | 23-ort-alignment.md | P1 | 18 | CLOSED — characterization complete; 3-leg parity GREEN at documented bound |
| 24 | 24-verdict-wave.md | P1 | 17 | r8a/b/c + run-026 + thai CLOSED; r8d (knee) RUNNING |
| 25 | 25-ar-news-specialist.md | P1 | 17 | VERDICT: OOD 10.13/8.98 (crown), ID 5.50; export staged |
| 27 | 27-ar-large-id-arm.md | P1 | — | NEGATIVE — 2.5765 parity with r7; capacity closed on AR too |
| 28 | 28-oracle-complementarity.md | P2 | — | CLOSED — ceiling 96.78 real; naive voting catastrophic |
| 30 | 30-heb-learned-ipa-v0.md | P1 | 09 | NEGATIVE — CER 1.18; ASR teacher is the bottleneck |
| 31 | 31-heb-phoneme-asr-v1.md | P1 | 30 | SPEC'D — tune the phoneme CTC on Hebrew first (owner block) |
| 32 | 32-matched-architecture-arms.md | P1 | 17 | BiLSTM 14.30 refuted; run-033 10.78/9.16 gate-MISS — specialist dominates; successor = word-channel class |
| 26 | 26-heb-noisy-student.md | P1 | 04 | NEGATIVE — 8.72 vs 8.18; all three text levers closed |
| 33 | 33-lexicon-disambiguator.md | P1 | 17/32 | NEGATIVE — 18.07/11.44; word-only can't diacritize OOV |
| 34 | 34-hybrid-word-char.md | P1 | 32/33 | CLOSED — hybrid 13.53, fastText 13.59 (density refuted), vcd ±0, oracle 0.71; specialist stands; delta = external data scale |

## Schedule

Wave 1 (local, days): WO01, WO02, WO03, WO07 + infra #44.
Wave 2 (HF training, launched together): WO04 (a100-large), WO06
(a100-large), WO08 (l4x1) — run concurrently, watch via `hf jobs ls`.
Wave 3: WO05 tables after 01+07 land; WO09 probe only.

## Path-to-success graph + SOTA matrix (2026-10-10, terminal for this campaign wave)

```
ARABIC ── WikiNews-2024 multiref (OOD news, WER/DER) ── MEASURED TERMINAL
  CROWN on comparable surface: ara-diac-news-1.0 10.13/8.98 (converged 10.03/8.95)
  full ladder: specialist 10.03 > plane-large 10.78 > hybrid 13.53 ≈ fastText 13.59
    > char-only 14.30 > word-only 18.07; oracle ceiling 9.33; vcd ±0.00
  closed by measurement: training, backbone scale, word channel (3 forms),
    lexical density, constrained decode, blending, corpus dose (202,680 units)
  remaining 2.70 delta = QCRI's FULL silver + self-consistent conventions
    → EXTERNAL: their corpus is theirs to share (owner decision, if ever)

ARABIC ── SadeedDiac-25 (ID): r7 2.2864 [BEST dedicated] · specialist 5.50 (register cost)

HEBREW ── nikud DER: 8.18 crown (heb-diac-plane-2.0, 2× runner-up)
  ALL TEXT LEVERS CLOSED (8.48/8.72/10.03) → WO31 audio program (owner block)
HEBREW ── g2p CER: rules 0.2393 · ReNikud 0.0244 (audio-supervised; WO31 line)
THAI ── PER: hybrid 0.1478 @0.73ms [crown]; tiny tier closed NEGATIVE
CLIENT SURFACE: 3-runtime parity ✓ · preserve ✓ · edge /v1/infer ✓
  index-v10 · npm 5.7.0 · 29 HF mirrors · ruby pins 802/804
```

| axis | ours | 2026 frontier | status |
|---|---|---|---|
| AR in-domain DER | 2.2864 | 1.39 (LLM API) | #1 dedicated (LLM = no-teacher law) |
| AR OOD multiref WER | 10.13 (10.03 converged) | 2.70 (full-corpus scale) | crown at our scale; delta external |
| HE nikud DER | 8.18 | — (we lead) | crown held; text exhausted; audio = WO31 |
| HE g2p CER | 0.2393 (rules) | 0.0244 (audio) | blocked on WO31 owner block |
| TH PER | 0.1478 | 26.9 (latency play) | crown held + hybrid latency tier |
| client surface | 3 runtimes + edge | none exists | moat maintained |
