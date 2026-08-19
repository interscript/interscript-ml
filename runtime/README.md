# secryst (Python crystal)

The Python crystal — reference implementation of **IMF v1** model zips
and the `models.yaml` index (the **interscript-ml** contract), the
phonological layer of Interscript. The Ruby (`secryst` gem) and
TypeScript (`@secryst/ml`) crystals are diffed against this one on
shared golden sets. This crystal owns golden generation and numerical
adjudication for the family.

Home repo: https://github.com/secryst/secryst-py (this copy in
ml-models/runtime is the frozen origin; the package now lives there).

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
  `~/.cache/secryst/models/<id>/`. Overrides:
  `SECRYST_INDEX` (URL or path), `SECRYST_CACHE`.

Install: `pip install ./runtime` (from the ml-models checkout) or
`pip install -e "./runtime[dev]"` for development.

Tests: `python -m pytest runtime/tests` — tiny-graph zips, no torch
needed. The end-to-end golden test runs when `SECRYST_E2E_ZIP`
points at a real zip (e.g. `models/khm-latn/khm-latn-1.0-fp32.zip`)
and asserts byte-identical outputs against `golden/khm-latn-100.jsonl`.

License: BSD-3-Clause.
