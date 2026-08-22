# 03 — CDN strategy (jsdelivr + GH releases)

**Status:** SPECIFICATION
**Priority:** P4

## Goal

A browser visiting interscript.org downloads the ONNX model with
<200ms latency from anywhere on earth. Zero origin load on GitHub.
Zero cost to us.

## URL convention

Primary (CDN-cached):
```
https://cdn.jsdelivr.net/gh/interscript/interscript-ml@<task>-v<x.y.z>/<task>.onnx
```

Fallback (direct GH):
```
https://github.com/interscript/interscript-ml/releases/download/<task>-v<x.y.z>/<task>.onnx
```

jsdelivr mirrors GitHub Releases content via the `@<tag>` syntax. The
URL is permanent, immutable, edge-cached. Free for any traffic volume.

## Why jsdelivr (not Cloudflare R2, not CloudFront, not raw GH)

- **Free, no account, no setup.** No credentials to manage.
- **CORS-friendly.** `Access-Control-Allow-Origin: *` by default. ONNX
  fetches from interscript.org work without proxy.
- **Edge cache.** 200+ POPs globally. A user in São Paulo hits a São
  Paulo POP, not GitHub San Francisco.
- **Stable URLs.** `@<tag>` is immutable. Cache headers are correct.
- **Version pinning.** `@rababa_arabic-v1.0.0` is forever. New releases
  get new URLs; old URLs never change.
- **No rate limits.** jsdelivr absorbs traffic; GitHub doesn't see it.
- **Auto-mirrors npm.** Same `jsdelivr.net/npm/<pkg>@<ver>/` pattern.

## Fallback chain (defensive)

```typescript
async function fetchWithFallback(url: string): Promise<ArrayBuffer> {
  // 1. Try jsdelivr (CDN, fast)
  try {
    const r = await fetch(`https://cdn.jsdelivr.net${url}`, { cache: "force-cache" })
    if (r.ok) return await r.arrayBuffer()
  } catch {}
  // 2. Try GitHub Releases directly (origin)
  try {
    const r = await fetch(`https://github.com${url}`)
    if (r.ok) return await r.arrayBuffer()
  } catch {}
  // 3. Try HuggingFace mirror
  const r = await fetch(`https://huggingface.co/interscript${url}/resolve/main/model.onnx`)
  return await r.arrayBuffer()
}
```

Triple redundancy. If any one of jsdelivr / GitHub / HuggingFace is
down, the other two pick up.

## Service Worker cache (browser)

```js
// sw.js
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url)
  if (url.pathname.endsWith(".onnx")) {
    event.respondWith(
      caches.open("models-v1").then(async (cache) => {
        const cached = await cache.match(event.request)
        if (cached) return cached
        const fresh = await fetch(event.request)
        if (fresh.ok) cache.put(event.request, fresh.clone())
        return fresh
      })
    )
  }
})
```

Result: second visit loads instantly. Models cached indefinitely
(versioned URLs = immutable = safe to cache forever).

## interscript-ts integration

`src/ml/provision/index.ts` already supports `setModelBase()`. Default
becomes:

```typescript
const DEFAULT_MODEL_BASE = "https://cdn.jsdelivr.net/gh/interscript/interscript-ml@"
```

`loadModel({ kind: "rababa", id: "default" })` resolves to:
1. Look up `<task>` from id (e.g. `"default"` → `"rababa_arabic"`)
2. Look up `<version>` from `npm `secryst` (manifest now the models.yaml index)` npm package
3. Build URL: `${DEFAULT_MODEL_BASE}<task>-v<version>/<task>.onnx`
4. Fetch with fallback chain
5. Cache in IndexedDB

## Version manifest (npm)

The `npm `secryst` (manifest now the models.yaml index)` npm package exposes a tiny JSON manifest:

```json
{
  "rababa_arabic": { "version": "1.0.0", "size_bytes": 6197600 },
  "rababa_hebrew": { "version": "1.0.0", "size_bytes": 5240800 },
  "secryst_thai_ipa": { "version": "1.0.0", "size_bytes": 4816000 }
}
```

`interscript-ts` reads this at startup. Users can pin a version via:

```typescript
setModelBase("https://cdn.jsdelivr.net/gh/interscript/interscript-ml@rababa_arabic-v1.0.0/")
```

## Acceptance

- [ ] interscript-ts default model base points to jsdelivr
- [ ] Service Worker cache registered on interscript.org
- [ ] Triple-fallback chain tested
- [ ] First-load p95 < 500ms globally (jsdelivr edge)
- [ ] Subsequent-load p95 < 50ms (SW cache)
