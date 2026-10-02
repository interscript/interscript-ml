# 01 — r8 teacher: run-009-yallamorph (YallaMorph/CamelMorph aux stream)

Status: CLOSED — VERIFIED NEGATIVE on both surfaces (2026-10-02); r7 stays canonical
Literature basis: YallaMorph (arXiv 2609.10153, EMNLP 2026) — 663,804
controlled morphological-generation instances over 4,795 lemmas,
constructed from CamelMorph MSA via CAMeL Tools. The GitHub repo ships
**samples only**; the full benchmark is not downloadable as a corpus.
Since the underlying resource (CamelMorph MSA + CAMeL Tools) is public,
we regenerate the paradigm pairs ourselves — a deterministic
dictionary/morphology resource, NOT LLM-generated labels (guardrail
compliant, see [[no-llm-teacher-distillation]]).

## Thesis

Every frontier move on the Arabic teacher came from data-side levers
(r5 2.6775 → r6-morph-aux 2.5793 → r7-news 2.2864). Morph coverage was
the r6 lever; YallaMorph/CamelMorph gives systematic,
feature-complete morphological coverage (incl. cliticized forms —
exactly the iʿrāb/clitic band where the residual concentrates) on top
of r7's news mix.

Naming: the `run-008` slot was consumed by the IPA-ablation
(train_arabic_r8.py → run-008-ipa). This teacher is
**run-009-yallamorph**; prose alias "r8-generation teacher".

## Design

Recipe = r7 verbatim (train_arabic_r7.py) + one new aux stream:

- Stream A (plain): r7's mix unchanged — cached r5-units (anchor) +
  news×3 + WikiNews-2014-gold×4.
- Stream M (aux, NEW): `MORPH: ` ASCII prefix (byte-distinct from
  Arabic, mirrors the proven `TAG: ` mechanism), input =
  `MORPH: <features-as-ascii> | <undiacritized form>`, target =
  `<fully diacritized form>`. Features rendered as ASCII
  key:value tokens (pos, aspect, state, case, ...). Multi-form
  instances: one line per valid form. IMPOSSIBLE configs: excluded
  (no target).
- Aux upsampled to ~25% of the mix (r6-proven dose).
- Init: run-007-news/best. Batch 2/accum 15, A100-80GB, bf16, 1 epoch,
  save every 300 steps, volume commit on save (r5/r6/r7-proven).
- Inference contract unchanged: no prefix at inference = plain
  diacritization.

## Steps

1. [x] Clone CAMeL-Lab/YallaMorph; extract xlsx samples as the
       validation set for our generated forms.
2. [x] Data build: camel-tools 1.5.7 + **Camel Morph MSA v1.0**
       (LREC-COLING 2024, CC BY 4.0 — the resource YallaMorph was
       constructed from; camel_data's calima-msa-r13 CANNOT generate
       mood/command forms — use the camel_morph repo DB). Key
       interface lesson: generation must be UNDERSPECIFIED (pos +
       proclitic variant only); fully-specified requests silently
       reject cells with unmarked features (1st person gen='u').
       13,000 lemmas (YallaMorph sample in-DB: v 513 / n 1,862 /
       adj 594 + inventory top-up, seed 42) → 4,735,166 raw pairs →
       300,000 lines (60/40 verb/nominal), 29MB, on volume
       /datasets/yallamorph-aux/ (lines.txt + DONE + README).
       Validation vs YallaMorph few-shot gold: **38/52 exact**;
       residual divergence = proclitic-chain conventions
       (hamzat-istifham أَلِـ, sin/lam stacking) + DB-version lemma
       gaps — not incorrect forms.
3. [x] TDD the line builder (build_yallamorph_aux.py, 12 tests
       green; rababa PR #104).
4. [x] train_arabic_r9_yallamorph.py (r7 copy + MORPH stream + gates;
       rababa PR #104). Mix verified in launch logs: anchor=586,505,
       news-mix=50,003, morph-used=197,169, **aux-share 25.00%**.
5. [x] Launch `modal run --detach` (+ supervisor; one double-launch
       incident from `modal app list` name truncation — grep prefix
       "rababa-ara", dupes stopped, volume verified clean).
6. [x] ID gate: windowed zero-skip SadeedDiac-25 full 1,200-para DER
       ≤ 2.389 (r7 2.2864 + 0.1 tolerance) — **FAILED: 2.4895**.
7. [x] OOD gate: WikiNews-2024 multi-ref — **FAILED: 17.4265/12.1093**
       vs r7's 17.3794/11.8273 (worse on both axes; no trade).
8. [x] Canonical call: **r7 stays canonical.** run-009 recorded as
       ablation. No student distillation from r9.

## Result (measured 2026-10-02) — NEGATIVE, first negative teacher-side result

| surface | r9 | r7 | verdict |
|---|---|---|---|
| SadeedDiac-25 Total DER (CE) | 2.4895 | 2.2864 | +0.20pp worse — gate fail |
| Morph DER | 1.5054 | 1.3343 | worse |
| WikiNews-2024 WER / DER | 17.4265 / 12.1093 | 17.3794 / 11.8273 | worse on both |

**Mechanism (data-backed):** the aux corpus's vocalization convention
is far denser than the benchmark's — marks per 100 letters: fatha
43.0 vs 28.4 (1.5×), damma 11.9 vs 7.1 (1.7×), shadda 8.6 vs 4.3
(2.0×), tanwīn 0.3 vs 2.0 (nearly absent). CamelMorph paradigm forms
are fully-explicit isolated words; SadeedDiac-style text is partially
vocalized running text. At 25% dose, the aux stream taught convention
drift, not morphology: the teacher over-marks the plain stream.

**Lesson (refines the knowledge-injection template):** r6's qalsadi
aux worked because it was running text IN benchmark convention; r9's
paradigm tables were out-of-context AND differently-conventioned.
Knowledge injection is delivery-vehicle-sensitive. A future teacher
lever from lexical resources must be rendered into benchmark-
convention running text (e.g. lexically-guided text selection from
the corpus, not paradigm tables).

The lineage: r5→r6→r7 all-positive; r9 first negative. Teacher-side
levers are NOT closed (r7's news mix remains the proven axis), but
this instantiation is dead: no re-run, no dose tuning, no convention
normalization retry — recorded and closed.

## Gates / risks

- Aux format interference with the plain stream: mitigated by the
  ASCII prefix mechanism (r6 evidence: deterministic at inference).
- CamelMorph license: record the underlying resource license in the
  aux data dir README before any redistribution (training-internal use
  only for now).
- Kill criterion: if ID gate fails after a clean run, close the lever
  and keep r7 canonical (news mix remains the last mover).

## Result

(to be written only from measured numbers)
