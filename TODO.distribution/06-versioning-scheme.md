# 06 — Versioning scheme

**Status:** SPECIFICATION
**Priority:** P2

## Goal

Independent version per task. Predictable impact of bumping. Clear
compatibility contract for consumers.

## Per-task SemVer

Each task has its own version: `<task>-v<MAJOR>.<MINOR>.<PATCH>`.

| Bump | When | Impact |
|---|---|---|
| MAJOR | Input/output vocab changes; ONNX I/O shape changes; map config incompatible | Consumers must update |
| MINOR | Retrain with more data; small architecture tweak (more layers); quantized variant added | Backward compatible; DER improves |
| PATCH | Bug fix; metadata fix; ONNX re-export with same weights | Drop-in replacement |

Examples:
- `rababa_arabic-v1.0.0` → `v1.0.1`: re-export with cleaner ONNX metadata, same weights.
- `rababa_arabic-v1.0.0` → `v1.1.0`: retrained on 2M more verses, DER drops 5.1% → 4.8%.
- `rababa_arabic-v1.0.0` → `v2.0.0`: vocab merges characters; interscript-ts needs new vocab handling.

## Compatibility matrix

Each release declares what it's compatible with:

```yaml
# rababa_arabic-v1.0.0/provenance.json
schema_version: 1
compatible_with:
  interscript_ts: ">=0.2.0 <0.3.0"
  interscript_ruby: ">=0.4.0 <0.5.0"
  onnxruntime: ">=1.17"
replaces:
  rababa_arabic: "0.x"
```

`interscript-ts` checks compatibility before loading; raises a clear
error on mismatch (not a silent fallback).

## Quantization variants

Quantized variants don't get their own version. They're a property of
the release:

```
rababa_arabic-v1.0.0/rababa_arabic.onnx       (fp32)
rababa_arabic-v1.0.0/rababa_arabic-q8.onnx    (int8)
rababa_arabic-v1.0.0/rababa_arabic-q4.onnx    (int4)
```

Same version, same weights, different precision. Same DER (modulo
quantization loss, which is documented in benchmarks.json).

## Pre-release tags

Following SemVer pre-release syntax:
- `rababa_arabic-v1.0.0-alpha.1`
- `rababa_arabic-v1.0.0-beta.2`
- `rababa_arabic-v1.0.0-rc.1`

Pre-releases are:
- Uploaded to GH Releases as drafts
- Published to HF Hub with `stage: prerelease` tag
- **NOT** added to `npm `secryst` (manifest now the models.yaml index)` manifest until promoted
- Available via explicit URL: `cdn.jsdelivr.net/gh/interscript/interscript-ml@rababa_arabic-v1.0.0-rc.1/`

## Versioning the framework itself

Framework version (in `pyproject.toml`) is independent of any task
version. The framework is a tool; tasks are products.

Framework: `0.1.0` → `0.2.0` → ... → `1.0.0` (after first production task).

## Deprecation flow

When a task version is superseded:
1. New release tag pushes normally.
2. `npm `secryst` (manifest now the models.yaml index)` manifest points `default` → new version.
3. Old version stays in GH Releases (immutable).
4. Old version's HF model card gets a banner: "Superseded by v1.1.0".
5. `Interscript.clear_cache!(older_than: 90.days)` eventually prunes.

No "yanked" versions unless integrity bug (then: revoke + alert + bump major).

## Acceptance

- [ ] First stable release: `rababa_arabic-v1.0.0`
- [ ] `provenance.json` includes `compatible_with` matrix
- [ ] Pre-release flow documented
- [ ] Deprecation banner appears on superseded HF model cards
