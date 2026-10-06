# 13 — API edge on the unified runtime

**Status: pending.** The Cloudflare worker (interscript-api) predates
the unified ts surface.

## Steps
1. Worker consumes interscript-ts (one transliterate contract: maps +
   models via index).
2. Bundled-maps artifact unchanged (data channel stays Releases);
   models resolve on demand or pre-warm per region (cost decision).
3. Conformance: worker outputs byte-match local runtime on the shared
   golden set.
