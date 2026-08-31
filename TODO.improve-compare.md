# Improve & compare — the plan (2026-09-01)

State: teacher r7 2.2864; best student 4.5701 (6ep, Muon); gap ~2.28pp
= ~0.25pp epochs (taken) + ~2.0pp domain (attribution stands, G2a
full-set) + residual optimization headroom (CE 0.0013 — thin).

## A. Improve — levers, ranked by measured attribution

| # | Lever | Mechanism | Pre-registered gate | Cost |
|---|---|---|---|---|
| A1 | **Domain-data rung (G2b)** | Tashkeela++ fetcher (rababa PR #1, open) -> more domain text -> r7 relabel -> 6ep+Muon distill | >= 0.5pp full-set gain attributes to domain; else the residual is optimization-adjacent | label+train ~1d GPU |
| A2 | **layerdrop @ 6ep** (browser-lite 2.1) | same epochs/labels/optimizer as G2a, half depth | tracks G2a's gain within 0.1pp (depth-cut shares the levers) | ~6h GPU |
| A3 | **MTP-aux rung (E5)** | multi-token-prediction auxiliary heads densify per-position supervision; candidate vs repetition pathologies | >= 0.3pp full-set OR measurably fewer decode loops | ~1d GPU + head code |
| A4 | int4-lite export + margins gate | quantize A2's artifact; E1-style margin flip gate before ship | flips < 1% at confident margins | CPU only |
| A5 | r8 multi-seed (paper A rigor) | 2 extra seeds, morph-vs-none arms | direction stable across seeds | teacher-tier GPU, expensive |
| A6 | Hebrew replication rung | layerdrop on heb teacher | law replicates cross-lingually | ~1d GPU |

Run A1/A2 first (biggest attribution pool); A3 is the novel-method
bet; A5/A6 are paper-rigor items.

## B. Compare — the apparatus (all built, mostly deployed)

1. **Full-set-only publication rule** — 4 documented overstatement
   instances (3.66->8.26, 3.81->7.44, 2.01->4.57 subset->full; the
   G2a near-overturn is the flagship exhibit). Never publish subset.
2. **Paired bootstrap CIs** — in the harness (delta/ci95/p_leq0,
   seed 42, sentence-level resampling). TODO: retrofit the headline
   specs (2.0, layerdrop, 1.0) so every paper table has brackets.
3. **Offline pairing** — final_preds.jsonl per run: any two runs on
   the same inputs pair without GPU; keep every future eval emitting
   it.
4. **Provenance** — labels sha256 + bytes in every verdict (live);
   artifact sha256 in models.yaml; protocol disclosure lines (decode
   params; LLM rows disclose reasoning_effort — GLM-5.3 memo).
5. **Subset-vs-full as a figure** — four pairs, one scatter; the
   measurement-discipline exhibit for paper B.
6. **Leaderboard protocol-matching** — dedicated vs prompted rows
   separated; GLM-5.3-Flash row (8.59/8.90) recorded; rababa PRs
   #64/#66 pending user rebase.
7. **Systems tiers** — bench E1/E2 measured (verification free;
   int8>int4 decode); E3 browser cell pending.

## C. Immediate queue

1. G2a CI rerun (in flight) -> CI table into RESULTS + paper tables
2. A2 layerdrop-6ep spec + launch
3. A1: rebase/merge rababa PR #1, fetch corpus, launch G2b
4. E3 browser bench scaffold
5. Paper B decomposition rewrite with G2a numbers + CI brackets
