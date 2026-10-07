# WO12 — Thai tiny contract (WO08 remainder)

run-025 running (a100-large): wiki slice -> umt5 teacher KD -> ~12M
char encoder-decoder. Gates: PER <= 3.5 AND int8 < 10MB.

1. Read verdict.json from the bucket.
2. PASS -> design kind=tiny-g2p zip contract (graph + joint vocab +
   decode params) + py runtime (greedy) + parity leg; ship.
3. FAIL -> RESULTS.md negative closes the sub-10MB idea cleanly.
