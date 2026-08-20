# interscript-ml

The **contract** for Interscript's phonological layer — the normative
definition of what a "hidden reading" model is, and the zoo that
publishes models conforming to it.

This repo owns three things and nothing else:

1. **The `models.yaml` index** — the stable URL every runtime resolves
   model ids against (with per-artifact sha256s and split-part support).
2. **The IMF v1 model-zip format** — the artifact contract:
   `metadata.yaml` + ONNX graphs + member sha256 manifest. The normative
   text is [SPEC.md](SPEC.md); the reference loader now lives in the
   [Python crystal](https://github.com/secryst/secryst-py).
3. **The model zoo + publish pipeline** — teachers from
   [secryst-train](https://github.com/secryst/secryst-train)
   are distilled, gated (parity written into the artifact), and released
   as index entries here.

## The system

```
interscript            deterministic transliteration maps + engines
(ruby · js · py)         │ maps that need vocalization dispatch to a
                        │ crystal through stdlib adapters (optional)
                        ▼
secryst crystals       Ruby gem · pip install secryst · npm i secryst
(secryst org)          implement IMF v1 + models.yaml — nothing else
                        │
                        ▼
interscript-ml  ◄────── models/zips resolve through this index
(THIS repo)     ──────► golden sets: crystals diffed against each other

secryst-train          teachers (arabic · persian · urdu + hebrew docs);
(secryst org)          secryst's own product; students ship through the zoo
```

Dependency directions, stated once:

- **interscript-ml depends on nothing.** It is the contract: an index,
  a format, golden sets, and release tooling.
- **Crystals depend only on the contract.** A crystal has zero
  interscript-core dependency — a TTS front-end can phonemize Khmer
  with `pip install secryst` and nothing else.
- **Engines depend on crystals only optionally.** An engine without a
  crystal simply cannot execute maps that declare a vocalization step.
- **Training is owned by secryst.** Teachers live in
  [secryst/secryst-train](https://github.com/secryst/secryst-train) —
  the crystal family's own product. This repo publishes their distilled
  students through the export gate; it does not train anything.

## Repositories

| repo | role |
|---|---|
| [interscript/interscript-ml](https://github.com/interscript/interscript-ml) | this — contract + zoo |
| [secryst/secryst](https://github.com/secryst/secryst) | Ruby crystal (the original, est. 2020) |
| [secryst/secryst-py](https://github.com/secryst/secryst-py) | Python crystal — reference, owns golden generation |
| [secryst/secryst-ts](https://github.com/secryst/secryst-ts) | TypeScript crystal (npm `secryst`) |
| [secryst/secryst.github.io](https://www.secryst.org) | the crystals' documentation site |
| [secryst/secryst-train](https://github.com/secryst/secryst-train) | training monorepo — **secryst-owned teachers** |
| [interscript/rababa](https://github.com/interscript/rababa) · [rababa-farsi](https://github.com/interscript/rababa-farsi) · [rababa-urdu](https://github.com/interscript/rababa-urdu) | archived origins of the train monorepo (full history merged there) |

`runtime/` in this repo is the **frozen origin** of the Python crystal —
kept for provenance; live code and releases are in secryst-py.

## Environment (as implemented by every crystal)

`SECRYST_INDEX` (index URL or path; default: `models.yaml` on this
repo's main) · `SECRYST_CACHE` (default `~/.cache/secryst`). Cache hits
are re-verified against the index on every load.

License: BSD-3-Clause.
