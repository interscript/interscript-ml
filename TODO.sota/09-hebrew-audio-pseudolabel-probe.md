# WO09 — Hebrew audio pseudolabel POC (PARKED probe)

**Priority:** P3 · **Status: PARKED — probe only.** No full campaign,
no GPU-scale labeling, without an explicit owner decision.

## Why parked

ReNikud's real moat is thousands of hours of Hebrew audio + a phoneme
ASR pseudo-labeling pipeline — a multi-week, multi-GPU program. The
2026-10-05 sweep verdict: park as the 2027 bet; probe viability only.

## Probe (what "implement all" means here)

1. `probe_hebrew_audio_pseudolabel.py`:
   - Input: a small dir of Hebrew audio clips (any public sample set;
     the script takes `--audio-dir` and does NOT fetch anything itself).
   - Stage 1: universal phoneme CTC ASR
     (`facebook/wav2vec2-lv-60-espeak-cv-ft`) → IPA phoneme sequence.
   - Stage 2: forced alignment viability check — espeak-style reference
     phonemization of our plane-restored nikud text vs the ASR phoneme
     stream; measure token-level agreement on a small sample.
   - Output: a viability report (agreement stats, per-hour labeling
     cost estimate on HF flavors, failure modes).
2. A short probe report in RESULTS.md (or a TODO note if audio isn't
   locally available), ending with the go/no-go recommendation for a
   2027 campaign.

## Explicit non-goals

No large-scale labeling; no trainer changes; no corpus construction; no
dependency on Dicta or any external API for labels (self-hosted models
only).
