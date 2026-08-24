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

| Model | PER | Exact match |
|---|---|---|
| Teacher (B-K/umt5 hub base) | 4.43% | 95.57% |
| **Student (ByT5-base, gate)** | **9.19%** | 90.81% |

Distillation cost: +4.76pp, inside the +5pp budget
(docs/DISTILL-SOURCE-PROMPT.md). ByT5-small ablations for reference:
12.63% on 23K labels, 12.06% on 48.7K labels (capacity-limited, both
rejected by the gate).

Context: the secryst-published 2.32% umt5 teacher is unrecoverable
from saved artifacts (transformers 5.15 save drops the untied umt5
lm_head) and the volume's epitran augmentation corpus is tone-less;
this release distills the best verified teacher available. A repaired
2.32%-tier teacher re-enters this pipeline when secryst regenerates it.

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
published SOTA on this benchmark.

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
