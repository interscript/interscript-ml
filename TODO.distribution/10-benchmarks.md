# 10 — Benchmark publication

**Status:** SPECIFICATION
**Priority:** P2

## Goal

Every release carries a machine-readable benchmark file. A live
leaderboard at `interscript.org/ml/benchmarks` shows historical DER/PER
across versions. A regression alert fires on PR if DER worsens.

## What we measure

### Quality metrics

| Task | Metric | Definition | Target |
|---|---|---|---|
| rababa_arabic | DER | Diacritic Error Rate (Levenshtein) | ≤ 5% |
| rababa_hebrew | DER | Nikud Error Rate | ≤ 8% |
| secryst_thai_ipa | PER | Phoneme Error Rate | ≤ 10% |

### Performance metrics

| Metric | Definition | Target |
|---|---|---|
| p50 latency | Median ms per word on CPU | ≤ 10ms |
| p95 latency | 95th percentile ms/word on CPU | ≤ 30ms |
| p99 latency | 99th percentile ms/word on CPU | ≤ 100ms |
| Cold start | ms until first output | ≤ 2000ms |
| Memory peak | MB RSS during inference | ≤ 100MB |

### Across variants

Each variant (fp32, q8, q4) gets its own benchmark file. Quantization
trade-off is explicit:

```
rababa_arabic-v1.0.0/
├── rababa_arabic.onnx                 (fp32)
├── rababa_arabic-q8.onnx              (int8)
├── rababa_arabic-q4.onnx              (int4)
├── rababa_arabic-benchmarks-fp32.json
├── rababa_arabic-benchmarks-q8.json
└── rababa_arabic-benchmarks-q4.json
```

## Benchmarks file format

`rababa_arabic-benchmarks-fp32.json`:

```json
{
  "schema_version": 1,
  "task": "rababa_arabic",
  "variant": "fp32",
  "version": "1.0.0",
  "evaluated_at": "2026-08-01T12:00:00Z",
  "test_set": {
    "source": "tashkeela_plus_plus/test",
    "size": 10000,
    "seed": 42
  },
  "metrics": {
    "der": 0.048,
    "der_breakdown": {
      "substitutions": 320,
      "deletions": 80,
      "insertions": 80,
      "total_chars": 10000
    }
  },
  "performance": {
    "p50_ms": 8.2,
    "p95_ms": 22.1,
    "p99_ms": 78.3,
    "cold_start_ms": 1850,
    "peak_rss_mb": 72
  },
  "hardware": {
    "cpu": "Apple M2 Pro",
    "ram_gb": 16,
    "node_version": "20.10.0",
    "onnxruntime_version": "1.17.1"
  },
  "compared_to": {
    "previous_version": "0.9.0",
    "der_delta": -0.003,
    "latency_delta_ms": -1.5
  }
}
```

## Regression alert

`.github/workflows/regression-check.yml` (on PRs touching model code):

```yaml
- name: Run benchmark
  run: python -m src.cli evaluate --task rababa_arabic > pr_bench.json
- name: Compare to main
  run: python scripts/compare_benchmarks.py main_bench.json pr_bench.json
```

`compare_benchmarks.py` posts a PR comment:

> **Quality regression detected**
>
> DER: 4.8% → 5.4% (worse by 0.6pp)
> Latency: 22ms → 24ms (worse by 2ms)
>
> This PR fails the regression check. Re-train or document the trade-off.

## Live leaderboard

`interscript.org/ml/benchmarks` page reads the manifest:

```
| Task | Version | DER | p95 latency | Released |
|---|---|---|---|---|
| rababa_arabic | v1.0.0 | 4.8% | 22ms | 2026-08-01 |
| rababa_arabic | v0.9.0 | 5.1% | 24ms | 2026-06-15 |
| rababa_hebrew | v1.0.0 | 7.2% | 19ms | 2026-07-20 |
| secryst_thai_ipa | v1.0.0 | 8.9% | 26ms | 2026-07-20 |
```

Sparkline charts show DER improvement over versions. Builds researcher
trust + competitive pressure.

## Reproducibility

`scripts/benchmark.sh` reproduces the published numbers:

```bash
./scripts/benchmark.sh rababa_arabic fp32
# → downloads model + test set
# → runs inference
# → prints DER, p50/p95/p99
# → diffs against published benchmarks-fp32.json
```

Discrepancy > 0.5pp DER is flagged for investigation.

## Acceptance

- [ ] `benchmarks-*.json` attached to every release
- [ ] Regression check on PRs
- [ ] Leaderboard page on interscript.org
- [ ] `scripts/benchmark.sh` reproduces published numbers
