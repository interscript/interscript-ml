# 04 — npm packages for JS consumers

**Status:** SPECIFICATION
**Priority:** P4

## Goal

JavaScript/TypeScript developers install one package and get every ML
model working out of the box. No model management. Sensible defaults.
Self-hostable for air-gapped networks.

## Package layout

Three npm packages, scoped under `@interscript`:

### 1. `interscript-ts` (existing, bumped)

The runtime. Already exists at v0.1.x. Bump to v0.2.0 when ML maps
go live. Adds:
- MLModel.transform() interface
- ONNX runtime integration (onnxruntime-node + onnxruntime-web)
- `transliterateAsync()` for ML funcalls
- Model provisioning layer

### 2. `npm `secryst` (manifest now the models.yaml index)` (new, tiny)

A pure-JSON manifest package. No JS code. Just a version index:

```json
{
  "schema_version": 1,
  "models": {
    "rababa_arabic": {
      "version": "1.0.0",
      "size_bytes": 6197600,
      "sha256": "3a7f2b...",
      "url": "https://cdn.jsdelivr.net/gh/interscript/interscript-ml@rababa_arabic-v1.0.0/rababa_arabic.onnx",
      "vocab_url": "https://cdn.jsdelivr.net/gh/interscript/interscript-ml@rababa_arabic-v1.0.0/rababa_arabic-vocab.json"
    },
    "rababa_hebrew": { ... },
    "secryst_thai_ipa": { ... }
  }
}
```

Size: ~1KB. Updated on every release. Used by `interscript-ts` to
resolve "default" → specific version + URL.

### 3. `@interscript/model-rababa-arabic` (new, optional)

Bundles the ONNX directly for offline-first JS apps:

```
@interscript/model-rababa-arabic/
├── package.json
├── rababa_arabic.onnx        (6MB)
├── rababa_arabic-vocab.json
└── index.js                  (exports { modelPath, vocabPath })
```

`import { modelPath } from "@interscript/model-rababa-arabic"` gives
a path to the local file. Useful for:
- Electron apps
- Node CLIs
- Air-gapped enterprise networks
- Reproducible CI

Same pattern for `@interscript/model-rababa-hebrew`,
`@interscript/model-secryst-thai-ipa`.

## Update flow

```
1. CI builds ONNX, attaches to GH release
2. CI updates npm `secryst` (manifest now the models.yaml index) manifest
3. CI runs: npm version patch && npm publish for npm `secryst` (manifest now the models.yaml index)
4. Users running `interscript-ts@0.2.x` see the new version on next
   app restart (model cache invalidates by URL change)
5. Users running `@interscript/model-rababa-arabic` opt-in via
   `npm update`
```

## Why three packages (not one)

- **MECE.** Runtime code (interscript-ts) vs version manifest (npm `secryst` (manifest now the models.yaml index)) vs bundled binary (@interscript/model-*). Three concerns, three packages.
- **Bundle size.** Browser apps want interscript-ts + npm `secryst` (manifest now the models.yaml index) (small). They fetch ONNX at runtime from CDN. Bundle stays small.
- **Air-gap friendliness.** Enterprises install @interscript/model-* via private npm mirror. No CDN calls.
- **Versioning independence.** Runtime API breaks != model version bumps != model retrain.

## Why `npm `secryst` (manifest now the models.yaml index)` not embedded in `interscript-ts`

If manifest lives in `interscript-ts`, every model release forces a
runtime bump. Decouples release cadence:
- Model retrain: bump `npm `secryst` (manifest now the models.yaml index)` only.
- Runtime API change: bump `interscript-ts` only.

## Install ergonomics

```bash
# Smallest install, runtime CDN fetch:
npm install interscript-ts

# Self-contained install (no CDN calls):
npm install interscript-ts @interscript/model-rababa-arabic

# Pin to specific model version:
npm install npm `secryst` (manifest now the models.yaml index)@1.2.0
```

## Versioning

`npm `secryst` (manifest now the models.yaml index)` uses CalVer-style: `1.<month>.0` so users can
see at a glance when the manifest was last updated. (e.g. `1.8.0` =
August 2026 release.)

`@interscript/model-<task>` versions track the model itself:
`1.0.0`, `1.0.1`, `1.1.0`, etc.

## Acceptance

- [ ] `npm `secryst` (manifest now the models.yaml index)` published (placeholder manifest)
- [ ] `@interscript/model-rababa-arabic` published (placeholder)
- [ ] `interscript-ts@0.2.0` released depending on `npm `secryst` (manifest now the models.yaml index)`
- [ ] README documents install patterns for all three combos
