# IMF runtime benchmarks — the evaluation axis for the systems paper (C)

Measures whether verified, content-addressed neural artifacts are
usable at each serving tier, and what the verification discipline
costs.

## Tiers (live Release index, production resolution path)
- ara-diac-small-1.0-int8  (257 MB int8, diacritization)
- tha-g2p-small-1.0-int8   (202 MB int8, g2p)
- tha-g2p-small-1.0-int4   (202 MB int4, g2p)
- (browser-budget rung: ara-diac-tiny-stitched when shipped)

## Environments
- E1 node: onnxruntime-node via `interscript` npm imf registry (Apple
  Silicon M-series, node 24; hardware recorded per run)
- E2 server: Modal 4-vCPU / 8 GiB — the production inference shape
  (models pre-staged on the secryst-models volume: measures serving
  path, not network)
- E3 browser: onnxruntime-web WASM (+ WebGPU where available) via
  headless Chromium — scaffolded, first runs pending

## Metrics
- M1 resolve+fetch+verify (cold, network) vs cache-hit load (warm)
- M2 zip open + member sha256 verify (the integrity-tax line item)
- M3 session create (ORT init)
- M4 decode latency by input length (short 16B / medium 128B /
  long 512B), tokens/s where applicable
- M5 peak RSS

Every run records: model ids, artifact sha256s, hardware, runtime
versions. Numbers land in RESULTS.md (## IMF runtime benchmarks).
