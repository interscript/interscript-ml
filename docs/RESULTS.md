# interscript-ml results

Evaluated results for models produced in this repository. Each section
anchor is the provenance target referenced by model metadata
(`models/*/` metadata.yaml) and `models/metrics-sources.yaml`.

## tha-g2p-base-1.0 — Thai G2P distillation (2026-08-19)

Sequence-level KD: B-K/umt5-thai-g2p-v2-0.5k teacher -> ByT5-base
student over 48,757 beam-4 teacher-generated labels (Kaikki + epitran
Wikipedia corpus, deduplicated, degenerate outputs filtered). Harness:
beam-4, corpus-level PER (total_ed / total_gold over characters of
joined-piece decode), 1,219 held-out Kaikki Thai test sentences
(`src/gpu/modal_distill.py::evaluate_per`).

| Model | PER (beam-4) | PER (greedy) | EM (greedy) |
|---|---|---|---|
| Teacher (B-K/umt5 hub base) | 4.43% | 1.25% | 95.16% |
| **Student (ByT5-base, gate)** | **9.19%** | **3.53%** | 90.48% |

Same-protocol distillation cost: +2.28pp greedy-to-greedy (was +4.76pp
beam-vs-beam), comfortably inside the +5pp budget. Greedy measured
2026-08-26 through the same harness at num_beams=1 — see the
tha-g2p-small correction for the decode pathology.

**Tier inversion at greedy:** the client tier (ByT5-small int4, 2.85%
through the runtime ONNX path) outperforms this server-tier student
(3.53% through the torch harness); the 0.08pp ONNX parity delta cannot
account for a 0.68pp gap, so the ordering is real on this harness. The
beam-4 figures had the tiers reversed.
(docs/DISTILL-SOURCE-PROMPT.md). ByT5-small ablations for reference:
12.63% on 23K labels, 12.06% on 48.7K labels (capacity-limited, both
rejected by the gate).

Context: the secryst-published 2.32% umt5 teacher is unrecoverable
from saved artifacts (transformers 5.15 save drops the untied umt5
lm_head) and the volume's epitran augmentation corpus is tone-less;
this release distills the best verified teacher available. A repaired
2.32%-tier teacher re-enters this pipeline when secryst regenerates it.

The 1,219-sentence test set is now published as a citable benchmark:
`benchmarks/thai-kaikki-g2p/` (data, protocol, reference points). No
external Thai comparison exists to date; systems evaluating on the
published benchmark can be ranked against the reference points above.

## fas-g2p-1.0 — Persian G2P (2026-08-19)

The v1 ByT5-small teacher shipped directly (byte-level, client-tier
size — no distillation step applies). REF teacher: persian-g2p-checkpoints
`persian_g2p/run-001/best`, RELEASE-FROZEN per rababa
docs/DISTILL-SOURCE-PROMPT.md (RL variants and the v5/mapped
representation line are closed negative; v1 is final).

| Metric | Value |
|---|---|
| CER (v1 test split, greedy, editdistance) | ≈1.6% |
| SentenceBench homograph (ezafe-normalized) | 77.34% |

Published reference: Homo-GE2PE homograph 76.89% — v1 is above the
published best on this benchmark. Claim scope (2026-08-26): SentenceBench
homograph accuracy only. Concurrent Persian G2P lines report on their own
PER benchmarks — prompted LLMs with post-processing (arXiv 2409.08554,
best 8.30% PER) and intermediate-language transliteration trained on
LLM-generated data (arXiv 2505.06599) — none shares an evaluation set
with SentenceBench, so no cross-paper ranking is claimed.

## heb-diac-small-1.0 — Hebrew student distillation (2026-08-20)

Logit KD from the s43 teacher (rababa_hebrew_byt5_s43/run-001/best):
KL + CE on hebrew-v4, ByT5-small init. Harness: greedy decode, Nakdimon
IMF test split (1,864 long sentences), same harness for both models
(`src/gpu/modal_distill.py::evaluate`).

| Model | DER | CER |
|---|---|---|
| Teacher (s43, ByT5-base) | 24.79% | 22.14% |
| **Student (ByT5-small, gate)** | **30.37%** | 24.47% |

Shrink cost +5.58pp — inside the ~5.6pp budget pre-accepted for this
pair (rababa docs/DISTILL-SOURCE-PROMPT.md section 2).

## tha-g2p-small-1.0 — Thai G2P client tier (2026-08-22)

The client-tier release of the Thai G2P distillation: run-003,
ByT5-small student on the full label set (48,757 usable beam-4 labels
from the B-K/umt5-thai-g2p-v2-0.5k teacher). Same harness as
tha-g2p-base-1.0 (beam-4, corpus-level PER, 1,219 held-out Kaikki Thai
test sentences, `src/gpu/modal_distill.py::evaluate_per`; checkpoint
re-measured 2026-08-22 for this release).

| Model | PER | Exact match |
|---|---|---|
| Teacher (B-K/umt5 hub base) | 4.43% | 95.57% |
| **Student (ByT5-small, client rung)** | **12.06%** | 87.94% |

Shrink cost +7.63pp — outside the +5pp server-tier gate (that gate is
met by tha-g2p-base-1.0 at 9.19%): shipped anyway per the frontier
below, as the smallest artifact that does not collapse. Exported at
int8 (~300MB); see the frontier table for why no smaller rung exists
today.

**Correction (2026-08-24): greedy is the real decode, and it is far
better than the beam-4 harness numbers.** Re-measured on the shipped
int4 zip through the Python runtime (the exact ONNX KV decode users
get), true Levenshtein, full 1,219-sentence set:

| Decode | Teacher PER | Student PER | Student EM |
|---|---|---|---|
| beam-4 (published, torch harness) | 4.43% | 12.06% | 87.94% |
| **greedy (runtime protocol)** | **1.25%** | **2.85%** | **88.93%** |

The teacher is also affected by the beam pathology (4.43 beam-4 → 1.25
greedy, measured 2026-08-25 through the same harness at num_beams=1):
same-protocol, the client tier's true shrink cost is **+1.60pp**, not
the +7.63pp the beam-vs-beam comparison suggested.

The beam-4 numbers are inflated by length-normalized beam preferring
long garbage on this model's flat per-token distributions (top-1
logprob ≈ -4.6 vs uniform -5.6): exact-match barely moves but every
non-exact output runs long, multiplying edit distance. Beam decode is
COUNTERPRODUCTIVE for these students; the runtimes ship greedy and
that is optimal. The runtime exposes num_beams as an opt-in (verified
correct against per-beam batch-1 references); the published beam-4
figures stand as measurements under that decode, not as quality
claims. All future gates decode greedy (the Arabic harness already
does).

## Client-tier size–quality frontier (2026-08-22)

Thai G2P, same harness (beam-4 corpus PER, 1,219 Kaikki sentences; teacher
B-K umt5 4.43%):

| Student | Init | Params | Artifact (int8) | PER |
|---|---|---|---|---|
| custom 8+8 d384 | random | 33M | ~30MB | 75.80 (collapsed) |
| custom 8+8 d384 + bridges | random | 33M | ~30MB | 71.12 |
| custom 10+10 d512 + bridges | random | 70M | ~70MB | 78.51 |
| ByT5-small | pretrained | 300M | ~300MB | 12.06 |
| ByT5-base (server tier) | pretrained | 580M | 1.2GB fp32 | 9.19 |

Findings: (1) random-init byte-level seq2seq collapses regardless of
capacity at this scale — the microkimi bridges improve structure (75.8 →
71.1) but cannot rescue G2P accuracy; enlarging without pretraining does
not help (70M = 78.5). (2) ByT5-small's width (d=1472) dominates its
parameter count — depth-pruning yields no useful intermediate rung
(263M). (3) The pretrained rung is the whole quality cliff: 300M at
12.06% (run-003, full labels; 12.63% on the 23K subset) vs 70M at 78.5%.

Conclusion: G2P client tier ships at the ByT5-small rung — 246MB at
int8, 202MB at int4 (parity 0.0734pp, quality cost ~0.17pp CER; PRs
#30/#31) — today; a 30–70MB G2P tier requires byte-level pretraining
of the small model first (future work). Copy-task languages are
evaluated separately below.

## ara-diac-tiny verdict — 33MB from-scratch student collapsed (2026-08-23)

The Arabic copy-task hypothesis test: a 33M-parameter custom byte-level
student (d384, 8+8) trained CE on 11,792 r6-teacher labels for 3
epochs (train CE converged to 0.46). Gate harness: windowed DER-CE at
the 1400-byte r5 window, greedy, haraqat-projected, Misraj evaluator —
identical to rababa's eval_sadeed_windowed; validated by the teacher
reproducing its documented tier on this replication.

| Model | DER-CE (300 Sadeed paragraphs) |
|---|---|
| Teacher (r6, run-006-morph) | 1.32% |
| **Student (33M from-scratch)** | **83.08%** — REJECTED |

Gate ≤ teacher + 0.5pp: the student misses by two orders of magnitude.

**RETRACTION (2026-08-24):** this verdict is CONFOUNDED — every Arabic
label generated before the byt5 `decode_joined` fix was mojibake
(double-encoded targets); both Arabic students trained on corrupted
labels, and their identical DER scores are the bare-text constant, not
a capacity result. The numbers stand as measured but the capacity
conclusion for Arabic is UNPROVEN pending a clean-label re-run. The
Thai tiny verdict is unaffected (umt5/sentencepiece labels were
byte-exact); the pretrained-backbone law rests on Thai evidence.

## ara-diac-tiny run-004 — the retracted verdict reproduced, on poisoned data (2026-08-29)

The spec intended to rerun the tiny tier on clean labels silently
consumed the Aug-23 label snapshot (pre `decode_joined` fix). Evidence
chain:

- run-004/best decodes `كتاب` as `ÙÙØ§ØªÙØ¨` — UTF-8-as-Latin1 mojibake,
  the exact label corruption the retraction describes
- the 300-paragraph windowed gate (n=300, teacher reproduces its
  documented 1.3205) scores the student **83.0797** vs the retracted
  **83.08** — identical to four decimals: the poisoned-data constant,
  not a capacity result
- `final_eval.json` in the run dir is the durable provenance

Verdict: run-004 says nothing about from-scratch capacity; the
retraction stands. **run-005** (fresh teacher labels, no snapshot,
2026-08-29) is the actual clean-label falsification test — in flight.

## ara-diac-tiny run-005 — clean-label verdict: from-scratch collapses (2026-08-29)

The falsification test the Aug-24 retraction called for: same 33M
from-scratch student (d384, 8+8), labels regenerated live from the r6
teacher (11,793 units, no snapshot), 4,422 steps / 3 epochs, final CE
~0.9. Windowed gate, 300 SadeedDiac-25 paragraphs, teacher reproduces
1.3205:

| Model | DER-CE (300) |
|---|---|
| Teacher (r6) | 1.3205% |
| Tiny, mojibake labels (run-004) | 83.08% |
| **Tiny, clean labels (run-005)** | **74.68% — REJECTED** |

Clean labels recover ~8pp of the collapse — the student learns real
signal — but remains two orders off the <= 3.07 gate. **The
pretrained-backbone law now rests on Arabic evidence as well as
Thai**: from-scratch byte-level students at this width do not work.
The viable path to a sub-100MB browser tier is width reduction FROM a
pretrained ByT5-small (closed-form stitch across widths, microkimi
protocol), not from-scratch training.

## ara-diac-small-1.0 — Arabic client tier (2026-08-24)

Sequence-level KD from the r6 teacher (rababa_arabic_byt5/run-006-morph,
2.5793 windowed DER-CE full-protocol): 29,322 greedy labels on r5-units
(domain + replay, 1400-byte windows), ByT5-small init, 3 epochs. Same
windowed harness as the ara-diac-tiny verdict (300 SadeedDiac-25
paragraphs, Misraj evaluator, haraqat projection):

| Model | DER-CE (300-para subset) |
|---|---|
| Teacher (r6) | 1.3205% |
| **Student (ByT5-small, client rung)** | **3.6580%** |

**Full-set correction (2026-08-26): the subset was not representative.**
Re-measured on the full 1,200-paragraph SadeedDiac-25 benchmark (same
harness; teacher reproduces its documented value at 2.5815 vs 2.5793,
confirming protocol consistency):

| Model | DER-CE (full 1,200) |
|---|---|
| Teacher (r6, full-set) | 2.5815% |
| **Student (ByT5-small, full-set)** | **8.2590%** |

The first 300 paragraphs sit in the student's training-domain
neighborhood; the remaining 900 expose a domain-generalization gap the
subset hid. The student's catalog number is the full-set 8.26; the
300-para figures above stand as measurements of that subset only.

Gate discussion: against the full set the strict budget (teacher
+0.5pp) is missed by +5.68pp — the capacity-plus-domain cost of
ByT5-small trained on 29K r5-unit labels, consistent in direction with
the Thai client tier. Shipped as the Arabic client rung with that
number disclosed: the student generates real, well-voweled Arabic at a
fraction of the teacher's artifact (1.3 vs 2.6 GiB) and the strict gate
is met by the teacher release (ara-diac-1.0, 2.58 full-set). On the
SadeedDiac-25 leaderboard the student at 8.26 sits just behind
Sadeed-1.5B (7.2915 published) and ahead of nothing measured below it —
the earlier "between Gemini-Flash and GPT-4" reading was an artifact of
the unrepresentative subset and is withdrawn.

Training notes: this is the third training of run-002 — the first on
mojibake labels (byt5 decode_joined bug), the second silently resumed
from the poisoned lineage's checkpoints (now guarded by labels.sha
digest matching), this one clean end-to-end. CE plateaued at ~0.016.

### run-003-pkm — memory-layer student (2026-08-28, research run)

The qwen-next capacity experiment (EXPERIMENTS.md E2): identical to
run-002 except three product-key memory layers (+85.9M lookup params,
zero-init gates) on the ByT5-small decoder — single-variable.

| Model | DER-CE (full 1,200) |
|---|---|
| Teacher (r6, full-set) | 2.5815% |
| **Student + PKM memory (run-003-pkm)** | **7.5553%** |
| Student vanilla (run-002) | 8.2590% |

0.704pp of the 5.677pp teacher-student gap closed (12.4% relative) at
near-zero added compute — below the pre-registered ≥1.0pp win bar, so
the memory axis is real but not the dominant term of the gap. Gates
verified engaged (0.034-0.053 at completion). Not shipped; the vanilla
client rung stands.

### run-004-pkm-muon — optimizer A/B on the memory student (2026-08-28)

Identical to run-003-pkm except the optimizer (Muon on 2D hidden
matrices, AdamW group for embedding-like params; EXPERIMENTS.md E3):

| Model | DER-CE (full 1,200) |
|---|---|
| Teacher (r6, this container) | 2.5997% |
| **ByT5-small + PKM + Muon (run-004)** | **4.8287%** |
| ByT5-small + PKM + AdamW (run-003) | 7.5553% |
| ByT5-small vanilla + AdamW (run-002) | 8.2590% |

**−2.727pp from the optimizer alone** — the adopt gate (≥0.3pp)
exceeded 9x; 3.430pp of the 5.677pp canonical gap closed (60.4%)
combining memory + optimizer. Training CE ~0.007 vs ~0.02 at equal
steps; ~1.2s/step vs ~3.4s; no stability events. The teacher-student
gap at this rung decomposes: ~0.70pp capacity + ~2.73pp optimization
+ ~2.25pp residual (domain coverage). The vanilla+Muon factorial cell
(run-005-muon) completes the decomposition.

### run-005-muon — factorial cell 4: vanilla + Muon (2026-08-28)

| Model | DER-CE (full 1,200) |
|---|---|
| ByT5-small + Muon (run-005) | 5.2945% |
| ByT5-small + PKM + Muon (run-004) | 4.8287% |
| ByT5-small + PKM + AdamW (run-003) | 7.5553% |
| ByT5-small vanilla + AdamW (run-002) | 8.2590% |

The 2x2 closes cleanly: optimizer alone −2.96pp; memory alone −0.70pp
(−0.47 under Muon); combined −3.43pp (60.4% of the 5.677pp canonical
gap) — roughly additive, slightly sub-additive on memory. Residual
~2.2pp is domain coverage. Optimization is the dominant recoverable
term of the distillation gap at the ByT5-small rung.

### run-006-r7-muon — E4: the 2.0 release candidate (2026-08-29)

The two measured wins compounded on the vanilla architecture: r7
canonical teacher (fresh greedy labels) + Muon optimizer, same
corpus/limits/seed family. Pre-registered E4 gate ≤ 6.26 (prediction
4.3–5.0):

| Model | DER-CE (full 1,200) |
|---|---|
| Teacher (r7, in-run) | 2.2890% |
| **ByT5-small, r7 labels + Muon (run-006)** | **4.8218%** |
| ByT5-small, r6 labels + Muon (run-005) | 5.2945% |
| ByT5-small, r6 labels + AdamW (run-002, shipped 1.0) | 8.2590% |

**4.8218 — gate passed; −3.44pp / 42% relative vs the shipped 1.0** at
identical architecture and artifact size. Matches the PKM arm's 4.829
without the memory layers. Release: ara-diac-small-2.0 (run-006
checkpoint; strict teacher+0.5pp still missed at +2.53pp, disclosed).

Leaderboard context (SadeedDiac-25, Misraj evaluator, zero-skip,
harakat-projected DER-CE): the teacher tier (r6, 580M) at 2.5793
(reproduced at 2.5815, 2026-08-26) is the best dedicated model measured
under this protocol — second only to Claude-3.7-Sonnet's published
1.3941, ahead of GLM-5.2 zero-skip (2.6911), Gemini-Flash-2.0 (3.1926),
GPT-4 (3.8645), and Sadeed-1.5B (7.2915; source table in rababa
docs/RESULTS.md). The client student's full-set 8.26 lands behind
Sadeed-1.5B; see the correction above.

## IMF runtime benchmarks — E1 node tier (2026-08-29)

Paper-C evaluation axis (benchmarks/imf-runtime; SPEC.md defines
tiers x environments x metrics). First measurements, node tier
(Apple Silicon, node 24, interscript@4.1.0, production Release path):

| tier | cold resolve+fetch+verify | warm cache-hit | sha256 tax | session create | decode (short/long) | peak RSS |
|---|---|---|---|---|---|---|
| ara-diac-small-1.0-int8 (257MB) | 25.0s (network) | 528ms | 110ms | 13.1s | 994ms / 2.76s | 953MB |
| tha-g2p-small-1.0 (int8, 202MB) | — | 397ms | 114ms | 12.1s | 85ms / 3.18s | 983MB |
| tha-g2p-small-1.0-int4 (202MB) | — | 369ms | 87ms | 7.3s | 213ms / 6.48s | 711MB |

Headline: **the integrity discipline is free** — whole-file sha256 is
~0.1s against 7-13s session creation; the verified-index + cache-hit
path is ~0.4s. int4 halves load time but decodes ~2.5x slower than
int8; int8 is the client default. E2 (Modal 4-vCPU / 8 GiB — the production serving shape, 2026-08-29):

| tier | cold load (zip + verify + ORT) | decode short/med/long |
|---|---|---|
| tha-g2p-small-1.0 (257MB) | 3.92s | 130 / 409 / 671 ms |
| ara-diac-small-1.0-int8 (257MB) | 4.04s | 395 / 525 / 973 ms |

Server vs node-laptop tier: cold load 4s vs 13s session create, decode
~2.5-3x faster — the serving tier trades network for speed. E3 (browser
WASM/WebGPU) pending.

## ara-diac-small-layerdrop — the depth-cut rung PASSES on the subset (2026-08-30)

Encoder 12->6 from pretrained ByT5-small, layers copied VERBATIM
(no projection - the width-cut rungs failed at 74.68/82.96), Muon,
same clean r6 labels, 10,995 steps, final CE 0.016 (the scratch rung
converged near 0.9 - 50x lower train loss at the same step count).
300-paragraph subset gate (teacher reproduces 1.3205):

| rung | params | subset DER-CE |
|---|---|---|
| ByT5-small 1.0 (AdamW, full) | 300M | 3.658 |
| **layerdrop (enc 6, Muon)** | ~190M | **3.8088** |

Halving encoder depth costs 0.15pp on the subset - the pretrained
representation survives a depth cut that width surgery destroyed.
Full-set gate (1,200 paragraphs) in flight; int8 ~190MB, int4 ~95MB
(the browser-budget artifact). Survived two infra failures en route
(eviction without watchdog; a regressed d_kv derivation) - both fixed.

## ara-diac-small-layerdrop — full-set verdict: 7.44 (2026-08-31)

The 1,200-paragraph gate (teacher reproduces 2.5815):

| rung | params | full-set DER-CE | subset DER-CE |
|---|---|---|---|
| 1.0 (full depth, AdamW, r6) | 300M | 8.259 | 3.658 |
| **layerdrop (enc 6, Muon, r6)** | **~190M (63%)** | **7.4413** | 3.8088 |
| r6 + Muon (full depth) | 300M | 5.2945 | — |
| 2.0 (r7 + Muon, full depth) | 300M | 4.8218 | — |
| scratch d384 | 33M | 74.68 | 83.08 |
| SVD width-stitch d384 | 29M | 82.96 | — |

Reading: halving encoder depth + Muon BEATS full-depth AdamW (7.44 vs
8.26) at 63% of the parameters — but the depth cut costs 2.15pp against
its optimizer-matched peer (5.29). Strict gate (teacher+0.5) failed.
This is the THIRD instance of the first-300 subset overstating quality
(3.66 vs 8.26; 3.81 vs 7.44) — the subset sits in the training-domain
neighborhood; full-set-only stands as the publication rule, now with a
quantified repeat rate. The size-quality frontier is complete and
monotone: 33M/74.7 - 29M/83.0 - 190M/7.4 - 300M/5.3 - 300M/4.8
(teacher 2.28-2.58). Browser-tier decision (user): ship
layerdrop-int4 (~95MB, ~7.5 DER with int4 flip risk ungated) as the
lite rung, or hold the tier at 2.0-int8 (264MB, 4.82).
