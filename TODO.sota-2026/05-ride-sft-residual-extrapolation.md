# 05 — RIDE-style SFT-residual extrapolation: probe-first arm

Status: SPECIFIED (2026-10-01) — probe only; training arm gated on probe
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

1. [ ] TDD pure computation: `residual_directions(h_base, h_teacher)`
       and cosine sim on synthetic tensors (tests first, watch fail).
2. [ ] Modal probe script (two models × 200 units × 2 domains;
       A100 minutes, not hours).
3. [ ] Run probe; write verdict + per-layer cosine table here.
4. [ ] Gate decision: close, or spec the training arm separately.

## Result

(to be written only from measured numbers)
