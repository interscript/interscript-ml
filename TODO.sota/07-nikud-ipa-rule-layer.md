# WO07 — nikud→IPA rule layer

**Priority:** P3 · **Cost:** days · **Gate:** deterministic, documented,
tested rule table over the v4 corpus cluster inventory; composes with
the plane runtime (`heb plane → nikud text → IPA`).

## Why

ReNikud's target surface is IPA for TTS (lexical stress, spoken norms).
We are not ReNikud — but a deterministic nikud→IPA layer gives TTS
clients a working Hebrew phonemizer TODAY, on top of our shipped plane
model, and unlocks WO05's comparability with heb-g2p-benchmark.

## Design

1. **Rules over the actual corpus inventory** (not theoretical Hebrew):
   enumerate the nikud clusters that occur in hebrew-v4 (the 135-class
   inventory); write letter×cluster→IPA per standard reading.
   - Special cases handled explicitly: שׁ/שׂ dots; dagesh lene in
     בג"ד כפ"ת; dagesh chazaq (doubling) conservative rule; sheva
     (initial → ə; otherwise ə/e per position heuristics, documented as
     heuristic); matres lectionis (vav/yod with holam/hiriq/male);
     final gutturals; qamats qatan (documented as lexicon-resolved).
2. **Lexicon overrides**: a table for the highest-frequency exceptions
   (qamats qatan words, proper names); rules lose to lexicon.
3. **Composition API**: `nikud_to_ipa(text)` in
   `interscript-py` + a `PlaneModel.translate_to_ipa()` convenience
   that chains plane restoration → rules. Stateless, deterministic,
   no ONNX involved.
4. **Tests**: cluster-inventory coverage table (every class → explicit
   rule or explicit lexicon fallback — no silent falls-through);
   round-trip spot checks against our labeled corpus pairs; golden
   examples for each special case.

## Deliverables

- `nikud_ipa.py` + rule doc (the table itself is the doc).
- Tests (TDD, real corpus strings, no doubles).
- Documented limitation: no lexical stress, no spoken-norm variation —
  that is ReNikud's audio-supervised territory; stated in the card/README.

## Non-goals

Learning anything (rules only); stress assignment; teamim.
