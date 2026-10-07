# WO15 — API edge on the unified runtime (TODO.final 13)

1. interscript-api worker consumes interscript-ts; maps resolve from
   bundled assets, models resolve via the index (kind-aware dispatch).
2. /v1/transliterate surface: {id, text} -> map engine OR neural
   runtime by index kind; /v1/models lists index entries.
3. Conformance: worker output byte-matches local runtime on the shared
   golden set (bounded: tiny-plane fixture scale).
