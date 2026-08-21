# ml-models results

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
