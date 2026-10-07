# heb-diac-plane-2.0

Hebrew nikud diacritization, plane-factorized (run-022): ByT5-**base**
encoder + per-character nikud-class head, K=3 mask-predict decode,
int8 (dynamic QInt8) ONNX, 416 MB. Trained on hebrew-v4 (nakdimon +
sefaria + distilled + expanded) on HF Jobs.

## Metrics (runtime protocol)

| benchmark | value |
|---|---|
| nakdimon test DER (greedy, int8, py runtime) | **8.18** (1.0: 12.48) |
| phonikud heb-g2p-benchmark CER / no-stress (via nikud_ipa rules) | 0.2393 / 0.1389 |

## Features

- `translate(text, preserve_diacritics=True)` — user-supplied nikud is
  pinned inside the decode and round-trips byte-exactly (leading marks
  included).
- Language-generic plane contract (`kind: plane`); same runtime as the
  Arabic plane family.
- nikud→IPA composition: `interscript.ml.nikud_ipa.plane_to_ipa`.

## Provenance

- Training: TODO.sota/04 (run-022, HF Jobs a100-large); corpus
  `Interscript/hebrew-v4` (private); recipe mirrors run-021 at
  byt5-base/4ep/K=3.
- Artifact sha256: `9c8f0432294ebae6abc847ddb79572c1ab979f4851f03d44a9bb6495ba767a9e`.
- License: BSD-3-Clause (code + weights).
