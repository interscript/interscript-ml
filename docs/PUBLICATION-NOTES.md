# Publication notes — what is worth publishing, and where it lives

Status: 2026-08-28. Rule: a claim is publishable when it is
harness-measured and recorded in RESULTS.md (models/protocol numbers)
or EXPERIMENTS.md (registered experiments). Nothing below cites an
un-landed number.

## Complete, measured, citable now

### 1. The artifact contract (IMF v1) and cross-runtime parity
One zip, three ONNX graphs, fixed byte table, per-member SHA-256,
byte-identical output from Ruby/Python/TypeScript; opset-14 floor for
old consumer runtimes. Paper: sections 3–4 (section-imf). The framing
contribution — neural models under the same discipline as
deterministic transliteration maps.

### 2. The decode-protocol correction
Beam-4 with length normalization inflates flat byte-student PER 4.2×
(12.06 published vs 2.85 real, same artifact, greedy). Language-
dependent at the teacher tier (Hebrew gains ~12 DER points from beam,
Arabic nothing). Paper: section-decode. This is the paper's most
quotable calibration result.

### 3. Pretrained-or-collapse (the client-tier frontier law)
Random-init byte students collapse at every capacity tested (33M
75.80 PER, 70M 78.51); the pretrained rung is the whole cliff (300M
→ 2.85). The Arabic replica was label-corrupted in transit —
disclosed, retracted as evidence, kept as a reproducibility lesson.
Paper: section-frontier + section-repro.

### 4. Controlled aux-representation ablation (r8, 2026-08-27)
Identical teacher/corpus/seed/init; only the aux stream's output
representation varies. Morphology 2.5793 < phonemic IPA 2.6588 < none
2.6775 (full 1,200-paragraph windowed zero-skip); the IPA projection
itself was learned (2.3% CER probe). **The strong form of "phonemes
help diacritization" fails; lexical-morphological knowledge is the
active ingredient.** Paper: leaderboard section. RESULTS.md (rababa):
r8 section.

### 5. Margin-aware parity + the head-fp32 quantization fix (E1)
Teacher-forced margin analysis (flip rate / KLD / near-tie share)
measured across the entire catalog — invisible to the CER gate.
Headline diagnosis: the shipped Hebrew int8 artifact flipped 9.34% of
positions, 80% at confident margins. Controlled probe matrix isolated
the quantized *head* (not weight granularity: per-channel recovered
almost nothing at +25% size); keeping the 1.2M-parameter head in fp32
cut flips 36× (0.26%, all near-tie) at +0.4% size. Now the export
default + a release gate + a regression test. Paper: IMF section
"Margin-aware parity" bullet. EXPERIMENTS.md E1.

### 6. The memory-layer capacity experiment (E2, 2026-08-28)
Single-variable: shipped ByT5-small student vs +3 product-key memory
layers (+85.9M params, near-zero compute; gates verified engaged).
8.259 → 7.555 full-set DER — 0.704pp of the 5.677pp teacher-student
gap (12.4% relative), **below the pre-registered ≥1.0pp bar**. Honest
conclusion: the distillation gap is dominated by optimization and
domain coverage, not parameter capacity. First controlled test of the
LongCat/Qwen "embedding scaling" axis on a byte-level seq2seq student.
Paper: frontier section (fourth question). RESULTS.md run-003-pkm.

### 7. Measurement-discipline findings
- Subset-selection artifact: first-300-paragraphs 3.66 vs full-set
  8.26 (teacher reproduces 2.5815/2.5793 — harness soundness proven).
  Paper: leaderboard section. Standing rule: full-set-only publication.
- Leaderboard positioning: r6 2.5793 is the best dedicated model under
  the protocol (behind Claude-3.7-Sonnet's 1.3941, ahead of GLM-5.2
  2.6911, Gemini-Flash 3.1926, GPT-4 3.8645, Sadeed-1.5B 7.2915).

### 8. Negative results (honesty assets, all in the log)
RL teacher polishing flat/negative ×3; microkimi bridges improve
structure but not accuracy; teacher beam-search unnecessary for
Arabic; per-channel int8 rejected on measurement; the 30 MiB tier
closed as infeasible without pretraining; **E5 MTP-aux (2026-09-01):
5.0853 vs the 4.8218 control — multi-token-prediction as a training
auxiliary HURT at this scale (+0.26pp), with a disclosed preemption
confound (fresh aux head for the final 23% of steps); E6
constant-budget register swap (2026-09-02): 5.8057 — replacing news
units with classical Tashkeela at constant total HURT (−0.98pp vs
control), the domain-shaped-residual hypothesis's causal test
failing in the swap direction; the add direction (G2b) remains
open.** The E5/E6 pair is the paper's data-vs-architecture exhibit:
two levers from the frontier-LLM literature (MTP, register
diversification at constant budget) both regressed on byte-level
student distillation — the levers that moved the rung were optimizer
(E3), fresher teacher labels (E4), and epochs (G2a).

### 9. Muon optimizer A/B on the memory student (E3) — LANDED 2026-08-28
**4.8287 vs 7.5553 full-set (−2.727pp from the optimizer alone); adopt
gate (≥0.3pp) exceeded 9×. ADOPTED.** Training CE ~0.007 vs ~0.02 at
equal steps, ~1.2s/step vs ~3.4s, no stability events. With E2 this
completes a controlled decomposition of the distillation gap at the
ByT5-small rung: ≈0.70pp capacity + 2.73pp optimization + 2.25pp
residual (domain coverage). The strongest training-methods result in
the paper — first controlled Muon measurement on byte-level seq2seq
distillation. Paper: frontier section. EXPERIMENTS.md E3, RESULTS.md
run-004. **Factorial cell 4 (vanilla+Muon, run-005): 5.2945 — the 2×2 closes cleanly** (optimizer alone −2.96pp, memory alone −0.70pp, combined −3.43pp, roughly additive). Paper carries the full table.

### 10. Arabic news-domain adaptation (r7) — COMPLETE 2026-08-28, NEW CANONICAL TEACHER
**ID 2.2864/1.3343 (−0.29pp over r6) AND OOD WikiNews-2024 multi-ref
17.38/11.83 (vs r6's 19.82/12.46)** — r7 sweeps both surfaces; the
teacher-lineage domain trade-off is gone. Best dedicated model under
the protocol, behind only Claude-3.7-Sonnet's published 1.3941, now
well clear of GLM-5.2 (2.6911). r7 replaces r6 as the canonical
Arabic teacher for future distillations. Paper leaderboard table
updated; rababa RESULTS.md carries both tables. Follow-on (user
decision): ara-diac-2.0 teacher release; students re-distilled from
r7 with Muon (E3) are the natural ara-diac-small-2.0.

## Venue decision (2026-08-29): TWO papers + a benchmark-backed third

The material outgrew one paper. Split by **teacher vs student**, not
training vs distillation — the 2x2 factorial gap decomposition is a
distillation-training result and must stay with the student paper.

**Paper A — teacher** (facts 1, 2, 8): "Morphological supervision,
not phonemic: controlled evidence and a dual-surface teacher for
Arabic diacritization." r8 single-variable aux ablation (morph
2.5793 < IPA 2.6588 < none 2.6775 - the strong form of "phonemes
help" fails), r7 dual-surface domain adaptation (ID 2.2864 / OOD
17.38-11.83, trade-off gone), protocol-matched leaderboard with the
full-set-only rule. 100% measured, writable now. ACL/EMNLP main or
SEMITIC.

**Paper B — student** (facts 3-7, 9-11, 13-14): "Closing the
teacher-student gap for browser-tier byte-level seq2seq." Spine: the
2x2 factorial (optimizer -2.96pp, memory -0.70pp, combined -3.43pp,
~additive, residual 2.25pp = domain). Pretrained-or-collapse law in
two languages with the label-provenance forensics arc (83.08 ->
83.0797 -> 74.68) as the reproducibility section; decode-protocol
correction (beam inflates 4.2x) and margins/head-fp32 (36x flip
reduction at +0.4% size) as calibration sections; the SVD-stitch
rung as the browser-budget ending (if it fails, the shipped
ByT5-small int8 tiers end the paper instead). IMF v1 as the artifact
appendix. Efficient-NLP/distillation venue with artifact badge.

**Paper C — systems** (fact 11 + the runtime stack): gated on a real
evaluation axis. Benchmarks to build (benchmarks/imf-runtime):
decode latency x tier x runtime backend (onnxruntime-node, Modal
4-vCPU serving shape, browser WASM/WebGPU), cold load vs cache-hit,
zip+member verification overhead, session-create cost, peak memory,
index-resolution latency. Without these it stays an appendix.

Fact allocation table (14 facts -> papers): A: 1, 2, 8. B: 3, 4, 5,
6, 7, 9, 10, 13, 14. C: 11, 12 + new benchmark results.

## ara-diac-tiny: the "clean-label re-run" wasn't (2026-08-29)

Narrative caution for the paper: the Aug-24 retraction flagged that the
83.08 tiny-collapse number was measured on mojibake labels. The
attempted clean re-run (run-004) silently reused the same Aug-23
snapshot via a baked-in `labels_file` path — and reproduced 83.0797
exactly, which is itself the tell: identical-to-4-decimals means same
data, not same capacity law. Two lessons for the experiments section:
(1) label-provenance must be content-hashed into the run record, not
inferred from filenames; (2) a reproduced number is only evidence of
reproducibility when the input pipeline is versioned. run-005
(regenerating labels live from the r6 teacher) is the honest test.

## ara-diac-tiny run-005: the law holds (2026-08-29)

The clean-label rerun scores 74.68 vs the poisoned run's 83.08 (gate
3.07): corruption explained 8pp, capacity explains the rest. The
"pretrained backbone or nothing" claim now has Arabic evidence
matching the Thai ablation, closing the retraction arc
(verdict -> retraction -> poisoned rerun -> clean rerun). For the
paper: report the pair (83.08, 74.68) as the label-quality and
capacity bounds of the same from-scratch rung, with the teacher
reproducing 1.3205 across all three measurements as the harness
control. Next rung on the frontier: stitch-down from ByT5-small
pretraining (ridge-fit width bridge), which changes init, not data or
capacity alone.

## layerdrop full-set (2026-08-31): the frontier is complete

7.4413 at 63% of parameters - better than the shipped 1.0, 2.15pp
behind its optimizer-matched full-depth peer. For paper B this closes
the capacity section: a monotone five-point size-quality frontier
(scratch, width-stitch, depth-cut, optimizer-matched, teacher-matched)
with every rung measured full-set under one harness. The subset
overstatement now has three quantified instances - it is a finding,
not a nuisance: domain-neighborhood evaluation subsets inflate student
quality by 2-4x in this regime. Frame it in the measurement section as
a generalizable warning with the three pairs as evidence.

## GLM-5.3-Flash on SadeedDiac-25 (2026-08-31): frontier generalist regression

First measurement of GLM-5.3-Flash (320B/18B active): **8.5721/6.5335
raw, 8.7978/6.6368 zero-skip** at reasoning_effort=low — ~3.4x worse
DER than GLM-5.2 (2.5060/2.6911), behind Sadeed-1.5B (7.2915). Two
protocol facts discovered live: the API rejects disabled thinking
outright (400 code 1210; valid efforts exactly low/high/max), so the
plain-completion protocol is inexpressible for this model. ATTRIBUTION
(2026-09-01, corrects the first orthography-shaped reading): the gap
is wrong haraqat — 10.05% of positions vs GLM-5.2's 2.64% (matching
our r7 teacher's 2.62% to 0.02pp); missing 1.01%, extra 0.20%, and
the whole dagger-alif U+0670 convention effect is 0.125pp (rules
derived from aligned positions; controls <=0.009pp). Paper: leaderboard
row + protocol note (PR #100, attribution corrected in PR after #69
in rababa). This strengthens the dedicated-model thesis: the
frontier's newest generalist lost classical-Arabic mark knowledge its
predecessor had, while the dedicated 580M teacher improved. The
per-position decomposition method (missing/wrong/extra +
convention-normalization with controls) is reusable — apply to any
future LLM row before interpreting its DER.

Sibling measurement finding (same day): resumable LLM-eval
checkpoints silently resume error sentinels — the first pass read
15.96 DER because 140 exhausted-retry empties were resumed as done.
Belongs in the paper's measurement-discipline paragraph with the
subset-inflation instances; the resume path now drops empty rows
(rababa PR #65).

## The distillation-gap decomposition, revised by measurement (2026-09-01)

The E2/E3 factorial (3-epoch students) attributed the ~5.7pp gap as
0.70pp capacity + 2.73pp optimizer + ~2.25pp residual "domain
coverage." The 6-epoch rung (G2a) and the CI-carrying harness revise
this: doubling epochs alone recovered 0.25pp full-set (4.8218 ->
4.5701, CIs non-overlapping) — the residual was not purely domain.
The decomposition for paper B, every line full-set with brackets:

| lever | full-set DER | paired CI of delta |
|---|---|---|
| teacher r7 | 2.2864/2.2921 | — |
| 1.0: AdamW, 3ep, r6 | 8.259 | retrofit in flight |
| + Muon (E3) | 5.2945 | — |
| + r7 teacher | 4.8218 | — |
| + 6 epochs (G2a) | 4.5701 | delta 2.12 [1.91, 2.35] |
| depth halved (lite, 6ep) | 5.784 | delta 3.25 [3.03, 3.49] |

Paper-B framing: levers compose roughly additively (optimizer >>
teacher > epochs), and the E6 causal test came back NEGATIVE —
swapping 8k news units for classical-register Tashkeela at constant
30k total scored 5.8057 (−0.98pp vs control), so the residual is
not fixed by register mix at constant budget; the add direction
(G2b, 48k total) is the live test, and capacity appears ONLY as
depth — width is load-bearing (both stitch ratios collapsed) while
depth trades 1.21pp for 37% of the artifact. The subset-overstatement
phenomenon (five instances, up to 3.2x inflation) is the
measurement-discipline exhibit.
