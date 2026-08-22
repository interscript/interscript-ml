# ML Model Distribution — Unified Plan

**Goal:** Every Interscript ML model is trivially installable, auto-updating,
reproducible, and verifiable. One source-of-truth, three delivery channels,
zero manual steps after a `git tag`.

## Audiences (real, identified)

| ID | Audience | Job to be done | Touchpoint |
|---|---|---|---|
| A1 | Web end-user | Type Arabic, get harakat | interscript.org, SW-cached ONNX |
| A2 | JS dev | `npm i`, call API, done | `interscript-ts` auto-fetches from CDN |
| A3 | Ruby dev | `gem install`, call API | `~/.cache/interscript/` lazy download |
| A4 | Researcher | Read card, fine-tune, reproduce DER | HuggingFace Hub + this repo |
| A5 | Mobile dev | Tiny ONNX, bundle in app | GH Releases (quantized variant) |
| A6 | Other-lang author | Implement IR + fetch model | JSON IR spec + stable release URLs |

## Channel architecture (hybrid, MECE)

```
Source code           → github.com/interscript/interscript-ml
                         (this repo: framework, configs, tests, CI/CD,
                          release changelog, issue tracking)

Trained weights       → huggingface.co/interscript/<task_name>
   (PyTorch, full)      (model card, datasets, transformers.js hook,
                          free Cloudflare-backed CDN, researcher entry)

Browser-ready ONNX    → github.com/interscript/interscript-ml/releases/tag/<task>-v<X.Y.Z>
                         (one .onnx asset per release; checksums attached;
                          stable immutable URLs; programmatic via `gh`)

CDN mirror            → jsdelivr.net/gh/interscript/interscript-ml@<tag>/...
                         (global edge cache of the GH release assets;
                          CORS-friendly; no rate limits for end users)

JS glue package       → npm: npm `secryst` (manifest now the models.yaml index)
                         (tiny manifest of current versions per task;
                          consumed by interscript-ts to resolve URLs)

Ruby model cache      → ~/.cache/interscript/<task>/<version>/model.onnx
                         (downloaded lazily from GH releases on first call;
                          SHA256-verified; atomic rename on success)
```

**Why this split (not "pick one"):**

- HuggingFace is the only channel researchers trust. Without it we're invisible to A4.
- GH Releases is the only channel with version-pinned immutable URLs tied to source tags. Without it we can't guarantee reproducibility for A5/A6.
- jsdelivr is the only channel with edge cache + CORS for browsers. Without it A1 pays full latency.
- npm manifest lets A2 self-host or air-gap by pinning the manifest.
- Ruby cache lets A3 work without bundling 6MB in a gem.

Each channel serves a real audience. None is redundant.

## File index

- `00-overview.md` — this file
- `01-github-releases.md` — release asset layout, tag conventions, asset naming
- `02-huggingface-hub.md` — HF org, model cards, datasets, auto-conversion
- `03-cdn-strategy.md` — jsdelivr + GH releases URL convention, fallback chain
- `04-npm-packages.md` — `npm `secryst` (manifest now the models.yaml index)` manifest package
- `05-ruby-model-cache.md` — gem-side download + verify + atomic write
- `06-versioning-scheme.md` — per-task semver, breaking-change rules
- `07-ci-cd.md` — `.github/workflows/release.yml` full design
- `08-supply-chain.md` — SHA256 checksums, SLSA provenance, Sigstore signing
- `09-model-cards.md` — per-task card template + automation
- `10-benchmarks.md` — DER/PER publication, regression alerts
- `11-offline-install.md` — air-gapped / private registry / mirror
- `12-mobile-edge.md` — quantized variants, format matrix
- `13-telemetry.md` — opt-in usage signals, privacy posture
- `14-sunset-kill-switch.md` — what happens when a model is bad

## Phase schedule (each phase delivers standalone value)

| Phase | Outcome | Audience unlocked |
|---|---|---|
| **P1: Foundation** (done) | Framework + tests committed | A4 (read-only) |
| **P2: Release infra** | Tag → ONNX → GH release auto | A5/A6 (binary) |
| **P3: HF integration** | `hf.co/interscript/<task>` live | A4 (full) |
| **P4: CDN + JS** | interscript-ts auto-loads from CDN | A1/A2 |
| **P5: Ruby cache** | `gem install` works out-of-box | A3 |
| **P6: First trained model** | rababa_arabic-v1.0.0 released | All |
| **P7: Quantized variants** | Mobile-ready assets | A5 (mobile) |
| **P8: Benchmark bot** | DER regression alerts on PR | A4 (internal) |

## Why GH Releases is the right primary channel (for our model sizes)

**The case for GH Releases as primary:**

1. **Size fit.** Our student ONNX is ~6MB. GH Releases supports up to 2GB
   per asset. HuggingFace LFS also fine. But GH Releases is the only one
   that gives us **release notes UI** tied to a source tag — that's the
   unit of "we shipped a thing".

2. **Reproducibility.** A `git tag rababa_arabic-v1.0.0` is a permanent
   pointer to a specific source state. The release attached to it is a
   permanent pointer to a specific binary. Researchers and CI pipelines
   can pin to that tag forever. HF Hub versions are similar but the
   release-notes UX is missing.

3. **Decoupling.** If HuggingFace changes ToS, gets acquired, or rate-
   limits, our CDN consumers (interscript-ts, interscript-ruby) keep
   working because they hit GH Releases / jsdelivr.

4. **Checksums first-class.** GH Releases API exposes asset digests;
   we add SHA256 sidecar files for explicit verification.

5. **Discovery via npm.** The npm manifest package points to GH release
   URLs. JS consumers never see HF. Researchers never see npm. Each
   audience gets the channel they expect.

**HuggingFace stays canonical for:**
- Model cards (structured metadata, search, demo API)
- Datasets (Tashkeela++ processed version)
- Auto-conversion to ONNX/TF-Flax/GGUF via HF Hub
- Programmatic inference API (free tier)

**Hierarchy:** HF = authoring surface. GH Releases = delivery surface.
jsdelivr = edge cache. npm = discovery.

## Design principles applied

1. **OCP.** Adding a new task = new directory under `src/tasks/` + new
   release tag pattern + new HF model repo. Zero edits to release infra.
2. **MECE.** HF owns the researcher surface. GH owns the binary surface.
   npm owns the JS discovery surface. No overlap.
3. **DRY.** The release workflow runs the same export code locally and
   in CI. The CDN URL convention is computed from the tag, not stored.
4. **Performance.** jsdelivr edge cache means a request from Tokyo hits
   a Tokyo POP, not GitHub's servers. First-load latency < 200ms.
5. **Reversibility.** Tags are immutable but new releases can supersede.
   Bad releases get a `retracted` flag in the manifest; consumers
   auto-upgrade.
6. **No AI attribution in release notes.** Per project conventions.
