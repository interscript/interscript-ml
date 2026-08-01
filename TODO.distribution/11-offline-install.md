# 11 — Offline / air-gapped install

**Status:** SPECIFICATION
**Priority:** P5

## Goal

Enterprises, governments, and researchers in low-connectivity regions
can install Interscript + models without ever touching the public
internet.

## Three patterns, MECE

### Pattern A: Cache prefetch + tarball transfer

For environments that have *some* internet access (just not for production servers).

```bash
# On dev machine (has internet)
gem install interscript
interscript prefetch --task rababa_arabic --version 1.0.0
# → downloads to ~/.cache/interscript/rababa_arabic/1.0.0/

tar czf interscript-models.tar.gz ~/.cache/interscript/

# Transfer to air-gapped server
scp interscript-models.tar.gz prod-server:/tmp/

# On prod server (no internet)
ssh prod-server
gem install interscript  # works without internet if gem is mirrored
tar xzf /tmp/interscript-models.tar.gz -C /
export INTERSCRIPT_CACHE_DIR=~/.cache/interscript
ruby -e "require 'interscript'; puts Interscript.transliterate('var-ara-Arab-Arab-rababa', 'كتب')"
```

### Pattern B: Vendor models in language-specific packages

For environments with private npm/RubyGems mirror.

**npm:**
```bash
# Has internet
npm pack @interscript/model-rababa-arabic
# Produces interscript-model-rababa-arabic-1.0.0.tgz

# Upload to private npm mirror (Verdaccio, Artifactory, npm Enterprise)
npm publish --registry https://npm.internal/ interscript-model-rababa-arabic-1.0.0.tgz

# Air-gapped dev
npm install --registry https://npm.internal/ @interscript/model-rababa-arabic
```

**RubyGems:**
```bash
gem build interscript-models-rababa-arabic.gemspec
gem push --host https://gems.internal/ interscript-models-rababa-arabic-1.0.0.gem

# Air-gapped
gem install --source https://gems.internal/ interscript-models-rababa-arabic
```

The gem's `lib/interscript/models/rababa_arabic.rb` exposes:
```ruby
module Interscript::Models
  module RababaArabic
    def self.path = Pathname(__dir__).join("../../../vendor/rababa_arabic.onnx")
    def self.version = "1.0.0"
  end
end
```

The interscript-ruby runtime checks `Interscript::Models::<Task>` first
before downloading. If present, no CDN call.

### Pattern C: Local CDN mirror

For environments that can't reach jsdelivr / GH but have a local server.

```bash
# Set base URLs at app startup
Interscript.configure do |c|
  c.model_versions_url = "https://internal.mirror/interscript/models.yml"
  c.cdn_base = "https://internal.mirror/interscript/"
end
```

```typescript
import { setModelBase, setVersionsUrl } from "interscript-ts"
setVersionsUrl("https://internal.mirror/interscript/models.json")
setModelBase("https://internal.mirror/interscript/")
```

Mirror contents are static files (just `rsync` from a public mirror).

## Prefetch CLI

`interscript prefetch` (Ruby) and `interscript-ts prefetch` (CLI):

```bash
interscript prefetch                              # all latest
interscript prefetch --task rababa_arabic         # one task
interscript prefetch --task rababa_arabic:1.0.0   # specific version
interscript prefetch --include-quantized           # all variants
interscript prefetch --output /path/to/dir         # custom location
```

Verifies SHA256 + Sigstore (if `--strict`) before adding to cache.

## Manifest for offline install

`interscript prefetch` writes a manifest at `~/.cache/interscript/manifest.json`:

```json
{
  "prefetched_at": "2026-08-01T12:00:00Z",
  "models": {
    "rababa_arabic": {
      "version": "1.0.0",
      "variants": ["fp32", "q8"],
      "total_size_bytes": 7500000,
      "sha256": { "fp32": "...", "q8": "..." }
    }
  }
}
```

This manifest can be inspected by sysadmins to plan capacity.

## Update flow (offline)

To update models on an air-gapped install:
1. Re-run prefetch on dev machine.
2. Diff old vs new manifest.
3. rsync changed files to internal mirror (or tarball transfer).
4. Bump `model_versions.yml` on internal mirror.
5. Servers auto-pick up new version on next call (cache miss → fetch from internal mirror).

## Why three patterns (not one)

- **Pattern A** is for one-off / low-frequency installs (researcher, gov lab).
- **Pattern B** is for enterprise dev workflows that already have a private package mirror.
- **Pattern C** is for production fleets that want centralized caching.

Three deployment scenarios, three patterns. MECE.

## Acceptance

- [ ] `interscript prefetch` CLI implemented
- [ ] `@interscript/model-*` npm packages produce `.tgz`
- [ ] `interscript-models-*` RubyGems produce `.gem`
- [ ] Documentation page: "Installing offline"
- [ ] Test: full offline install works in Docker with no network
