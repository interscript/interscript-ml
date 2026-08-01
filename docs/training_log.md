# Training log

Append new entries at the bottom. Format:

```
## YYYY-MM-DD — task_name — notes
- params: {key: value}
- DER/PER: x.xx
- wall time: Xh
```

## 2026-08-01 — framework skeleton committed

- Initial framework: config, registry, data, model, trainer,
  evaluator, exporter, pipeline.
- Task packages: rababa_arabic, rababa_hebrew, secryst_thai_ipa.
- CPU-only unit tests pass. No GPU training yet.
