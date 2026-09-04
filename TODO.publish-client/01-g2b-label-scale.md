# 01-g2b-label-scale

The domain-residual causal test: build units from the cleaned tashkeela-full corpus, relabel with the r7 teacher, distill at the G2a recipe. Gate: >=0.5pp full-set gain attributes to domain. [models+papers]

Status: DONE (2026-09-04). Full-set verdict 4.8231 [2.194,2.554] vs G2a 4.5701 [1.91,2.35] — flat-negative, gate FAILED. With E6's swap-negative, the domain-coverage attribution is rejected in both directions; residual reframes as corpus-unreachable (on-policy GKD is the live lever). RESULTS.md PR #170, Paper B cell PR #171.
