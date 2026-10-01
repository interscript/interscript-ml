# 03 — Student-side lever family: CLOSED (record + caveat)

Status: SPECIFIED (2026-10-01)

## The record

The student–teacher residual (2.1 student 4.5701 vs r7 teacher 2.2890)
has now resisted every lever measured:

| lever | result |
|---|---|
| corpus scale / register mix (both directions) | negative / flat-negative |
| on-policy distillation (GKD) | negative (6.0036) |
| product-key memory layers | real but small (−0.70pp) |
| epochs 3→6 | −0.25pp (secondary) |
| Muon optimizer | −2.96pp (adopted — shipped in 2.0/2.1) |
| headwise Muon (671B-scale recipe) | separated-negative (+0.2267pp) |
| Sinkhorn embeddings | flat |
| lexical memory (engram) | flat |

The 2026-09-30 literature sweep independently corroborates the
closure: the OPD wave (MAESTRO 2609.37510; RIDE 2609.36484; Fisher
sparsity 2609.36262; sparse supervision 2609.04565) targets reasoning
trajectories with distribution-shift mechanisms our deterministic
dense-label task does not have. No new student-side method in the
window contradicts the verdict.

**Standing rule: no further GPU spend on student-side levers without a
pre-registered mechanism novel to the ledger.** The frontier mover on
record is teacher-side data (r5→r6→r7). Next frontier experiments:
01 (YallaMorph aux) and 05 (RIDE probe — the only student-side item
with a cheap kill-gated probe).

## Caveat to publish (methodology honesty)

Per arXiv 2608.27729: our paired between-students bootstrap measures
prediction-resampled variance, NOT training-seed variance. All arm
verdicts to date are single-seed runs. The headwise-Muon
separated-negative (+0.2267pp, p=0.017) is directionally consistent
with its size class, but the seed axis is unmeasured; future arms run
multi-seed or state the caveat. Ship decisions are unaffected (the
base recipe shipped regardless).

## Steps

1. [ ] RESULTS.md: lever-ledger closure entry + seed-variance caveat
       (same PR as 02, separate commit).
2. [ ] paper.adoc: one sentence attaching the caveat to the
       optimizer-recipe arm paragraph.

## Result

(to be written only after merge)
