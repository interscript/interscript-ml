# 05 — RIDE-style SFT-residual extrapolation: probe-first arm

Status: CLOSED — arm measured SEPARATED-NEGATIVE (2026-10-02)
Literature basis: RIDE (arXiv 2609.36484) — extrapolate the
teacher-over-base residual directly in representation space:
student hidden states regressed toward
`h_target = h_teacher + λ·(h_teacher − h_base)`; approaches or exceeds
the teacher across four base/RL-teacher pairs.

## Scope correction (user-confirmed 2026-10-01)

The mechanism does NOT require an RL teacher. It needs any
(base, improved) checkpoint pair; the residual direction
`d = improved − base` is what is extrapolated. Our **r6→r7** SFT pair
(run-006-morph → run-007-news) qualifies. What remains forbidden is RL
*training* ([[rl-negative-diacritization]] — measured flat 3×), not
residual extrapolation of an SFT delta.

## Why probe-first

- Student-side lever (ledger: 8 negatives) — do not spend GPU on a
  training arm before the direction is shown to transfer.
- The r7 delta is small (−0.29pp ID) and domain-shaped (news mix);
  extrapolating a domain-idiosyncratic direction would amplify news
  specialization, not general diacritization competence.

## Probe (cheap: forward passes only, no training)

1. Load run-006-morph/best and run-007-news/best (580M ByT5 each).
2. Forward N=200 units from two domains: SadeedDiac val paragraphs
   (classical) + WikiNews-2024 text (news). Capture per-layer
   mean-pooled encoder hidden states.
3. Per layer: d_classical = mean(h_r7) − mean(h_r6) on classical;
   d_news likewise on news. Compute cos(d_classical, d_news).
4. **Kill criterion: max-layer cosine < 0.5 ⇒ direction is
   domain-idiosyncratic ⇒ close the arm, record in RESULTS.md.**
5. Pass ⇒ full arm: hidden-state distillation with displacement
   (λ ∈ {0.5, 1.0}) as an aux loss on the student trainer — requires
   a new feature-regression path in modal_distill (spec before code;
   TDD the loss on synthetic tensors).

## Steps

1. [x] TDD pure computation: `residual_directions(h_base, h_teacher)`
       and cosine sim on synthetic tensors (7/7 green,
       rababa/test_ride_probe_math.py).
2. [x] Modal probe script (rababa/probe_ride_direction.py; A10G,
       ~8 min, app ap-LePMpScA3RA8EN6TLDP2vN).
3. [x] Run probe; verdict + per-layer cosine table below.
4. [x] Gate decision: **TRANSFER** — training arm spec'd below.

## Result (measured 2026-10-01)

Per-layer cos(d_classical, d_news), 200 units per domain, mean-pooled
encoder hidden states, run-006-morph vs run-007-news:

```
L00 +0.9414  L01 +0.9513  L02 +0.9199  L03 +0.9002  L04 +0.8765
L05 +0.8696  L06 +0.8593  L07 +0.7734  L08 +0.5931  L09 +0.4521
L10 +0.2145  L11 -0.0015  L12 -0.1086  L13 -0.0842  L14 +0.0167
L15 +0.1624  L16 +0.1971  L17 +0.2387  L18 +0.2775
max +0.9513 >= 0.5  ->  TRANSFER
```

Reading: the r7-over-r6 SFT residual is strongly domain-general in
early/mid encoder layers (L0–L8: 0.59–0.95) and idiosyncratic deep
(L11+ ≈ noise). The news-mix fine-tune moved surface/orthographic
processing in a direction that transfers to classical text — the
RIDE displacement premise holds where representations are shared.
Artefact: rababa-checkpoints:/ride_probe_r6_r7.json.

## Training-arm spec (gated on this probe; launch = owner decision)

*(Executed 2026-10-01 under "Proceed all" — run-016-ride, PR #233.)*

- Infra: feature-regression aux loss in modal_distill — teacher/base
  hidden states must be cached per layer subset. Restrict to L0–L8
  (probe: only these transfer; deep-layer displacement would inject
  domain idiosyncrasy).
- Target: h_t' = h_teacher + λ(h_teacher − h_base), λ ∈ {0.5, 1.0}.
- Loss: L = CE(labels) + β·MSE(h_student[L0-8], h_t'[L0-8]),
  β tuned so MSE term ≈ 10% of total at start (pre-registration).
- Single-variable off the 2.1 recipe (run-007 data, teacher labels,
  Muon, seed 42); adopt gate ≥ 0.3pp DER improvement (E4-style bar).
- Est. build: teacher/base hidden-state dump (one-off Modal job,
  ~1h A100) + trainer loss path + spec; run cost ≈ one 2.1-recipe arm.

## Arm result (measured 2026-10-02) — SEPARATED-NEGATIVE, arm CLOSED

run-016-ride: the 2.1 recipe verbatim (r7 labels, Muon, 6 epochs,
seed 42) + encoder-hidden regression toward ridge-projected displaced
targets (λ=1.0, layers 0–8, β auto-calibrated 3.576e-05 = 10% of CE
at start, ridge fit on 16 batches of mask-flattened positions;
ride.pt checkpointed). Training converged normally (CE 0.57→0.29 over
13,026 steps); teacher reproduced at 2.2921 on the same eval.

| measure | value |
|---|---|
| student DER-CE (full 1,200) | **5.8627** |
| vs 2.1 rung 4.5701 | **+1.3926pp [1.156, 1.646], p=0.0** — separated |
| delta vs teacher | 3.5038 [3.222, 3.792] |

**Verdict: NEGATIVE — decisively.** The direction-transferability
probe passed (max cos 0.9513), the premise was mechanistically sound,
training was healthy — and the outcome still hurt by 1.4pp. Reading:
domain-general direction is necessary but not sufficient; regressing
a 300M byte student's encoder toward ridge-projected 580M targets
DISPLACES representations the decoder relies on, competing with the
CE objective rather than sharpening it. This is ledger row #9 and the
strongest test of the closure rule to date: the one student-side arm
with a measured mechanistic premise still failed. The student-side
residual is closed on evidence, not exhaustion. No further arms.
