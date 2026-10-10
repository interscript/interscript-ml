# WO35 — the alternate-format scoring artifact + the re-opened 1.4-point program

## Discovery (2026-10-10)

Error-typography of the specialist's 1,065 wrong WikiNews words: 74.2%
length mismatches, 1.4% vowel-class — inverted from expectation. Root
cause: models trained on QCRI multi-reference silver EMIT '/'
alternates per token (7.44% of tokens; Sadeed rows too: 12.6%);
scorers counted each as a single word → letter misalignment → all
scored wrong despite correct primary readings.

Corrected protocol (first-alternate decode — deterministic,
shippable, same benchmark):

| arm | prior | first-alt |
|---|---|---|
| run-029 specialist (shipped ara-diac-news-1.0) | 10.03/8.95 | **4.10/1.56** |
| run-036 hybrid | 13.53/9.92 | 8.85/2.84 |
| symmetric any-vs-any (specialist) | — | 3.18 WER |

Rankings from WO32-34 hold (dominance unchanged); absolute numbers
shift. models.yaml metrics corrected. Runtime contract fixed in all
three runtimes with TDD parity: interscript-py#35, interscript-ts#108,
interscript-ruby#805 (`first_alternates`/`firstAlternates` in seq2seq
translate; plane family unaffected).

## Answer to "how do we get their silver + conventions"

We already HAVE the conventions — the corpus taught our model to emit
them (that was the artifact). Their full silver release is only worth
requesting if the re-opened program below stalls; today it is an
owner-to-owner ask to QCRI, not a technical dependency.

## Remaining work

1. Sadeed rescore under first-alt (preds saved; DER 5.50 inflated).
2. Re-open the OOD program at 4.10 vs 2.70 (1.4 WER):
   - hybrid-2 stack: run-036/037 class + the specialist's corpus mix
     (joint seq2seq + word channel), sweep lr/hid/epochs (knobs wired).
   - oracle probe re-run under corrected protocol once new arms land.
3. Card text for ara-diac-news-1.0 mirror (HF repo 29) + release
   notes: decode-contract change → runtime version bumps are the
   OWNER's decision (artifact bytes unchanged; sha valid).
4. Derive the 1.56 DER vs published frontier DER comparisons for the
   external-benchmark table (WO05) once Sadeed rescore lands.
