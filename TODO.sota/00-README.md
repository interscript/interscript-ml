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
| 21 | 21-arabic-plane-transfer.md | P2 | 17 | arm QUEUED (run-028, silver 60K) — trainer train#118, corpora migrated |
| 22 | 22-runtime-releases.md | P1 | 14 | PREPARED — owner-gated versions |
| 23 | 23-ort-alignment.md | P1 | 18 | CLOSED — characterization complete; 3-leg parity GREEN at documented bound |
| 24 | 24-verdict-wave.md | P1 | 17 | r8a/b/c + run-026 + thai CLOSED; r8d (knee) RUNNING |
| 25 | 25-ar-news-specialist.md | P1 | 17 | RUNNING (run-029, register-pure full scale) |
| 27 | 27-ar-large-id-arm.md | P1 | — | RUNNING (run-031, byt5-large dedicated, r5 corpus) |
| 28 | 28-oracle-complementarity.md | P2 | — | CLOSED — ceiling 96.78 real; naive voting catastrophic |
| 26 | 26-heb-noisy-student.md | P1 | 04 | stage 1 RUNNING (labeling); stage 2 staged |

## Schedule

Wave 1 (local, days): WO01, WO02, WO03, WO07 + infra #44.
Wave 2 (HF training, launched together): WO04 (a100-large), WO06
(a100-large), WO08 (l4x1) — run concurrently, watch via `hf jobs ls`.
Wave 3: WO05 tables after 01+07 land; WO09 probe only.

## Path-to-success graph + SOTA matrix (2026-10-07, live)

```
ARABIC ── in-domain (SadeedDiac-25)        HEBREW ── nikud (nakdimon)
  r7 2.2864 [BEST dedicated]                 heb-plane-2.0 8.18 [BEST text-only]
  Claude-3.7 1.39 (API LLM; hallucinates)    next: us (12.48) → 16.4 → 35.6
    │                                          │
    └─ r8a (self-labeled arwiki) RUNNING       └─ byt5-large/K=4 (untried)
    └─ r8b (QCRI silver)       RUNNING       HEBREW ── g2p IPA (their board)
    └─ gate: hold ID ≤2.2864 AND              rules chain 0.2393 CER (0-train)
       move OOD 17.38 → ≤10                   ReNikud 0.0244 (audio-supervised)
                                              └─ learned IPA head: BLOCKED
ARABIC ── OOD (WikiNews-2024 multiref)          (espeak ceiling 0.4543; needs
  r7 17.38/11.83 · plane 18.69/11.35            MILIM or audio = 2027)
  QCRI 2.70 (in-domain-trained silver)
    └─ register closure = r8a/r8b above      THAI ── g2p (kaikki protocol)
                                              hybrid 0.1478 PER @0.73ms [BEST]
ARABIC plane ── 2.7397 (large) ── r8 recipe    tiny student RUNNING (<10MB tier)
   transfers to plane family post-r8        CLIENT SURFACE — 3-runtime parity ✓
                                              + preserve mode ✓ + edge /v1/infer ✓
                                              + index-v9 + 28 sha-verified models
```

| axis | ours | 2026 frontier | status |
|---|---|---|---|
| AR in-domain DER | 2.2864 | 1.39 (LLM API) | #1 dedicated; r8 arms closing |
| AR OOD multiref WER | 17.38 | 2.70 (their-domain) | register gap; r8a/r8b attacking |
| HE nikud DER | 8.18 | — (we lead) | crown held; base+K=3 |
| HE g2p CER | 0.2393 (rules) | 0.0244 (audio) | blocked on supervision (2027) |
| TH PER | 0.1478 | 26.9 (latency play) | crown held + hybrid latency tier |
| client surface | 3 runtimes + edge | none exists | moat maintained |
