# WO10 — heb-diac-plane-2.0 release chain (WO04 remainder)

**Gate: PASSED** — artifact DER 8.18 (int8, runtime protocol; 1.0=12.48).

1. GitHub Release `heb-diac-plane-2.0`: zip (416MB) + sha256 sidecar.
2. `models.yaml` entry (kind: plane, k_passes: 3, int8) with runtime-protocol metrics.
3. Card `cards/heb-diac-plane-2.0.md` (nakdimon 8.18; heb-g2p-benchmark CER 0.2393/0.1389 no-stress; preserve mode).
4. Golden rows for neural-parity (50-row py-reference jsonl).
5. Cut `index-v9` (models-index.yaml release asset) with the new entry.
6. Pin bumps: interscript-py registry, interscript-ruby-ml, interscript-ts -> index-v9 (pin tests updated).
7. HF mirror: Interscript/heb-diac-plane-2.0 (zip + card README).
