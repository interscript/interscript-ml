# interscript-ml (Python runtime)

The reference Python runtime for **IMF v1** model zips — the phonological
layer of Interscript. The Ruby (secryst gem) and TypeScript
(@interscript/ml) runtimes are diffed against this one on shared golden
sets.

```python
from interscript_ml import Model

model = Model.load("khm-latn-1.0")        # id: index resolve -> download
                                          # -> sha256-verify -> cache -> load
model.translate("ភាសា")                   # -> "pheasaea"
model.id                                   # "khm-latn-1.0"

model = Model.load("khm-latn-1.0.zip")    # or: a local zip path directly
```

- Byte-level only: the canonical ByT5 table (byte `b` → id `b+3`,
  trailing EOS) — no vocab files, no per-model tokenization code.
- Greedy KV-cache decode when the zip ships `decoder-kv.onnx`
  (default), plain full-recompute fallback otherwise.
- Every `.onnx` member is sha256-verified against `metadata.yaml`
  before the session is created; corrupt downloads fail loudly.
- Dynamic fetch per the `models.yaml` contract (shared with the Ruby and
  TypeScript runtimes): resolve id -> channel URL, download to temp,
  verify whole-file sha256 against the index, atomically install into
  `~/.cache/interscript/models/<id>/`. Overrides:
  `INTERSCRIPT_ML_INDEX` (URL or path), `INTERSCRIPT_ML_CACHE`.

Install: `pip install ./runtime` (from the ml-models checkout) or
`pip install -e "./runtime[dev]"` for development.

Tests: `python -m pytest runtime/tests` — tiny-graph zips, no torch
needed. The end-to-end golden test runs when `INTERSCRIPT_ML_E2E_ZIP`
points at a real zip (e.g. `models/khm-latn/khm-latn-1.0-fp32.zip`)
and asserts byte-identical outputs against `golden/khm-latn-100.jsonl`.

License: BSD-3-Clause.
