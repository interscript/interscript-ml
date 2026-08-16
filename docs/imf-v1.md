# Interscript Model Format (IMF) v1

IMF v1 is the versioned, portable artifact of **interscript-ml** — the
phonological layer of Interscript. A model.zip is adoptable on its own
terms, like ONNX itself: any runtime that can open a zip, sha256 a file,
and run two ONNX sessions can serve the model. Adopting the artifact does
not require adopting our training code.

The format exists to make three guarantees:

1. **One runtime everywhere.** v1 supports exactly one tokenizer: raw
   UTF-8 bytes (pad=0, EOS=1, ByT5 convention). No vocab files, no
   sentencepiece, no per-model tokenization code in Ruby/TS/Python.
   Non-byte models enter via distillation (TODO.runtime-arch/07), never
   via a second tokenizer system.
2. **Old-runtimes load it.** Opset is pinned to 14 because the Ruby
   `onnxruntime` gem bundles an old ORT that cannot load higher opsets
   (verified the hard way — secryst PR #44). The validator enforces
   graph opset == metadata opset <= 14.
3. **Every number is traceable.** Metrics in metadata must cite a
   `RESULTS.md` anchor; the parity block records ONNX-vs-reference
   agreement. Numbers that cannot be traced do not ship.

## Zip layout

```
model.zip
├── metadata.yaml      # manifest (schema below) — never hand-write the
│                      # sha256 block; `imf pack` computes it
├── encoder.onnx       # required. inputs: [input_ids], dynamic batch/seq
├── decoder.onnx       # required (fallback path). inputs: [input_ids,
│                      #   encoder_hidden_states] -> [logits]
├── decoder-kv.onnx    # optional (default artifact when present).
│                      #   inputs add past_*; outputs add present_*
└── README.md          # required. Usage in all three APIs
```

## metadata.yaml schema

| Field | Type | Constraint |
|---|---|---|
| `format` | str | must be `imf-v1` |
| `id` | str | `<name>-<major>.<minor>`, lowercase segments, e.g. `khm-latn-1.0` |
| `task` | enum | `g2p` \| `diacritization` \| `translit` |
| `source_script` | str | ISO 15924 script code (e.g. `Khmr`) |
| `target` | str | target script or scheme (e.g. `Latn`) |
| `tokenizer` | enum | `bytes` (the only v1 value) |
| `opset` | int | 7..14; must equal the graphs' opset |
| `decoder` | enum | `plain` \| `kv` (`kv` requires decoder-kv.onnx) |
| `precision` | enum | `fp32` \| `fp16` \| `int8` (fp16 = torch-native half export: float16 graph IO, int64 ids unchanged; runtimes read dtypes from the session — the ORT float16 converter produces all-zero hiddens on real ByT5 and must not be used) |
| `license` | str | non-empty (strict gate) |
| `trained_from` | str | repo + run/checkpoint id |
| `metrics` | list | `{name, value, protocol, source}`; `source` must be a `RESULTS.md#anchor` (strict gate) |
| `parity` | map? | `{samples, cer_delta}`; strict gate: samples >= 500, cer_delta <= 0.2pp |
| `sha256` | map | every `*.onnx` member -> hex digest; no dangling entries |

The `id` does not encode precision: `khm-latn-1.0-fp16.zip` and
`khm-latn-1.0-int8.zip` share id `khm-latn-1.0`; the model index
(TODO.runtime-arch/08) resolves channel and precision.

Metrics blocks are generated from `docs/RESULTS.md`, never hand-written
(TODO.runtime-arch/10). Parity is produced by the WO03 gate.

## Example

```yaml
format: imf-v1
id: khm-latn-1.0
task: translit
source_script: Khmr
target: Latn
tokenizer: bytes
opset: 14
decoder: plain
precision: fp16
license: BSD-3-Clause
trained_from: secryst train_khmer_byt5.py run-001 (secryst-checkpoints:/khmer_byt5/run-001/best)
metrics:
  - name: cer
    value: 27.42
    protocol: "greedy decode; 895 held-out pairs; split 16,120/895/895 seed 42"
    source: secryst/docs/RESULTS.md#khmer-transliteration-2026-08-14
  - name: em
    value: 59.66
    protocol: "greedy decode; 895 held-out pairs; split 16,120/895/895 seed 42"
    source: secryst/docs/RESULTS.md#khmer-transliteration-2026-08-14
parity:
  samples: 500
  cer_delta: 0.03
sha256:
  encoder.onnx: a4a4eb...
  decoder.onnx: b54b5c...
```

## Validation

Two levels (`src/imf/validator.py`):

- **Base** — what every runtime does on load: zip integrity (CRC),
  required members, metadata parses, every `.onnx` sha256-verified,
  graph opset matches metadata and stays <= 14, decoder contract
  (`input_ids` / `encoder_hidden_states` / `past_*`-`present_*` names).
- **Strict** — the release gate: base + non-empty anchored metrics,
  parity within thresholds, license present. No zip ships without it.

CLI (also the CI entry point):

```
PYTHONPATH=src python -m imf validate models/khm-latn/khm-latn-1.0-fp16.zip
PYTHONPATH=src python -m imf validate <zip> --strict     # release gate
PYTHONPATH=src python -m imf info <zip>                  # print manifest
PYTHONPATH=src python -m imf pack --source <dir-or-legacy-zip> \
    --metadata <yaml> [--readme <file>] --out <zip>       # sha256 computed
PYTHONPATH=src python -m imf parity <zip> --checkpoint <hf-dir> \
    --test-data <jsonl>            # WO03 gate; writes parity into the zip
PYTHONPATH=src python -m imf golden <zip> --inputs <jsonl> --out <jsonl> \
    # cross-runtime golden set: 100 fixed strings, Python = reference
```

The parity gate compares ONNX KV greedy decode against the transformers
decoder loop (the exact math the export wraps — not `generate`, whose
config-dependent behavior no runtime implements) over >= 500 test pairs;
it writes `{samples, cer_delta}` into metadata and refuses to leave the
zip non-strict. On Modal the same gate runs headless:

```
modal run --detach src/gpu/modal_export.py::main --model urd-g2p
modal run --detach src/gpu/modal_export.py::parity --model urd-g2p
```

CI runs the full gate on the fixture model (export -> parity ->
strict-validate) in the `export-fixture` job.

Legacy notes:

- Old secryst zips (`vocabs.yaml` + single `transformer.onnx`) and the
  PR #44 byte-level zips (`metadata.yaml: name: byt5`) predate IMF.
  They fail base validation with a pointer to re-export/upgrade.
- The fp32 Khmer zip on `secryst-checkpoints:/khmer_byt5/` has a CRC
  error in `encoder.onnx` (found 2026-08-16, exactly the corrupt-download
  class of failure the sha256-on-load rule exists for). It must be
  re-exported by the WO02 pipeline; the fp16 zip is intact and was
  upgraded to IMF v1 via `imf pack`.

## Versioning

Format changes bump the `format` field (`imf-v2`, ...). Model versioning
lives in `id` (`khm-latn-1.1`). Adding an optional member or metadata
field is v1-compatible; anything a v1 runtime would misinterpret is a
new format version.
