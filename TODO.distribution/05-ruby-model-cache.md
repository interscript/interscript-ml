# 05 — Ruby model cache

**Status:** SPECIFICATION
**Priority:** P5

## Goal

`gem install interscript` and `Interscript::Stdlib::Functions.rababa(...)`
"just works". Model file is downloaded lazily on first call, cached
per-version, SHA256-verified, atomic rename.

## Where the cache lives

`~/.cache/interscript/<task>/<version>/<filename>` on Unix.
`%LOCALAPPDATA%\interscript\<task>\<version>\<filename>` on Windows.

Cache layout:

```
~/.cache/interscript/
├── rababa_arabic/
│   └── 1.0.0/
│       ├── model.onnx
│       ├── model.onnx.sha256
│       ├── vocab.json
│       └── config.json
├── rababa_hebrew/
│   └── 1.0.0/
│       └── ...
└── secryst_thai_ipa/
    └── 1.0.0/
        └── ...
```

## Download flow

```ruby
def fetch_model(task:, version:, filename:)
  cache_dir = Interscript.cache_dir.join(task, version)
  cache_dir.mkpath
  target = cache_dir.join(filename)
  return target if target.exist?

  expected_sha = fetch_expected_sha256(task:, version:)
  url = cdn_url(task:, version:, filename:)
  tmp = cache_dir.join("#{filename}.tmp.#{Process.pid}")

  URI.open(url) do |response|
    File.open(tmp, "wb") { |f| IO.copy_stream(response, f) }
  end

  actual_sha = Digest::SHA256.file(tmp).hexdigest
  raise IntegrityError, "checksum mismatch for #{url}" unless actual_sha == expected_sha

  File.rename(tmp, target)
  target
end
```

Properties:
- **Atomic.** Temp file + rename = no partial state on crash.
- **Verified.** SHA256 mismatch raises; bad bytes never reach disk as the canonical filename.
- **Idempotent.** Re-downloading is a no-op if cached.
- **Concurrent-safe.** Temp filename includes PID; multiple processes don't collide.

## Where the URL + checksum come from

The cache layer reads `Interscript.model_versions` (configurable, defaults
to bundled manifest). The manifest is a YAML file shipped with the gem:

```yaml
# lib/interscript/model_versions.yml (bundled)
rababa_arabic:
  version: "1.0.0"
  sha256: "3a7f2b..."
  size_bytes: 6197600
  url_base: "https://cdn.jsdelivr.net/gh/interscript/interscript-ml@rababa_arabic-v1.0.0"
```

Override at runtime:
```ruby
Interscript.configure do |c|
  c.model_versions_url = "https://internal.mirror/interscript/models.yml"
  c.cache_dir = Pathname("/var/cache/interscript")
end
```

## Cache size management

```ruby
Interscript.clear_cache!(older_than: 90.days)  # cleanup old versions
Interscript.cache_size  # => 18.4 MB
```

LRU eviction: when total cache > 100MB, oldest unused version is pruned.

## Offline install

For air-gapped networks:
1. On a network-connected machine, `bundle exec rake interscript:prefetch` downloads all current models to `~/.cache/interscript/`.
2. Tar + transfer to air-gapped machine.
3. Set `Interscript.cache_dir` to the extracted location.

OR: vendor the ONNX in a Ruby gem:

```ruby
# Gemfile
gem "interscript-models-rababa-arabic", "~> 1.0"
```

That gem bundles the ONNX (~6MB) and exposes `Interscript::Models::RababaArabic.path` returning the local file path. The cache layer checks `Interscript::Models::<Task>` first, falls back to download.

## Integration with existing Rababa/Secryst adapters

Existing code in `lib/interscript/stdlib/functions/rababa_adapter.rb`:
```ruby
model_path = Interscript.rababa_provision(config, model_uri)
```

`rababa_provision` is the legacy hook. We update it to:
1. Check if a newer model exists at `Interscript.model_versions[:rababa_arabic]`
2. Call `fetch_model(task: "rababa_arabic", version: ..., filename: "model.onnx")`
3. Return the cached path

Legacy configs (`Interscript.rababa_configs[config_key]`) keep working — they just supply a custom `model_uri` that overrides the manifest URL.

## Acceptance

- [ ] `Interscript.cache_dir`, `Interscript.model_versions` APIs added
- [ ] `lib/interscript/model_provision.rb` implements download+verify+atomic rename
- [ ] SHA256 mismatch raises `Interscript::IntegrityError`
- [ ] Legacy `rababa_provision` / `secryst_provision` route through new layer
- [ ] Bundled `lib/interscript/model_versions.yml` updated per release
- [ ] Optional `interscript-models-*` gems provide offline path
