# WO02 — User-diacritics preservation in plane runtimes

**Priority:** P1 · **Cost:** hours–days · **Gate:** `preserve` mode with
byte-identical user diacritics, all tests green, no behavior change when
off.

## Why

QCRI's EMNLP 2025 paper ships "a model that preserves user-specified
diacritics" as a flagship client feature. For us it is nearly free: the
plane's K-pass conditioning already takes a plane tensor as input, so a
partially-vocalized input is just a partially-filled plane — pin user
classes, mask only the empty positions.

## Contract

`PlaneModel.translate(text, preserve_diacritics=True)`:

1. `split_planes(text)` on the INPUT (skeleton + user classes; Hebrew:
   `nikud_planes`, Arabic: `haraqat_planes`).
2. Initial plane tensor = user class ids where non-empty, MASK_ID where
   empty. K-pass decode runs as usual.
3. Positions the user supplied are NEVER overwritten by the model —
   final classes = user classes at those positions, model predictions
   elsewhere.
4. Byte-exact guarantee: every diacritic the user typed round-trips
   verbatim (cluster order preserved as written — the Hebrew canon
   lesson applies to user input too).
5. Default `preserve_diacritics=False` — behavior and parity goldens
   UNCHANGED.

## Deliverables

- `interscript-py`: `PlaneModel.translate(..., preserve_diacritics=...)`
  (TDD: fully-labeled input → identity output; partial input → user
  classes intact, empty positions filled; unlabeled → identical to
  current path).
- `interscript-ruby-ml`: same flag on `Interscript::ML::PlaneModel#translate`
  (same test matrix against the tiny-plane fixture).
- ts port: blocked on the documented ts plane loader gap — note it, do
  not unblock here.
- RESULTS.md / model card note: preservation mode + the byte-exact
  guarantee.

## Explicit non-goals

No seq2seq (IMF) preservation mode — that decode cannot pin positions
without a new contract; out of scope.
