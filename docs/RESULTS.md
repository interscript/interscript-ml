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

## ara-diac-small-1.0 — Arabic client tier (2026-08-24)

Sequence-level KD from the r6 teacher (rababa_arabic_byt5/run-006-morph,
2.5793 windowed DER-CE full-protocol): 29,322 greedy labels on r5-units
(domain + replay, 1400-byte windows), ByT5-small init, 3 epochs. Same
windowed harness as the ara-diac-tiny verdict (300 SadeedDiac-25
paragraphs, Misraj evaluator, haraqat projection):

| Model | DER-CE |
|---|---|
| Teacher (r6) | 1.3205% |
| **Student (ByT5-small, client rung)** | **3.6580%** |

Gate discussion: the strict budget (teacher +0.5pp) is missed by
+2.34pp — the measured capacity cost of ByT5-small on Arabic
diacritization, consistent with the Thai client tier (+7.63pp beam-4 /
2.85% greedy against a 4.43% teacher). Shipped as the Arabic client
rung per that precedent: the student generates real, well-voweled
Arabic at a fraction of the teacher's artifact (1.3 vs 2.6 GiB), the
strict gate is met by the teacher release (ara-diac-1.0), and the miss
is disclosed rather than averaged away. For comparison, this student's
3.66% sits in the same league as the r3 production teacher's era
(2.68% on the older full-set protocol).

Training notes: this is the third training of run-002 — the first on
mojibake labels (byt5 decode_joined bug), the second silently resumed
from the poisoned lineage's checkpoints (now guarded by labels.sha
digest matching), this one clean end-to-end. CE plateaued at ~0.016.

Leaderboard context (SadeedDiac-25, Misraj evaluator, zero-skip,
harakat-projected DER-CE): the teacher tier (r6, 580M) at 2.5793 is the
best dedicated model measured under this protocol — second only to
Claude-3.7-Sonnet's published 1.3941, ahead of GLM-5.2 zero-skip (2.6911),
Gemini-Flash-2.0 (3.1926), GPT-4 (3.8645), and Sadeed-1.5B (7.2915;
source table in rababa docs/RESULTS.md). This student's 3.66% was
measured on the 300-paragraph subset; the full-1,200-paragraph
measurement is running (2026-08-26) and will replace the subset number
when it lands.
