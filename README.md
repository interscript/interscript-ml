# interscript/ml-models

Unified training framework for Interscript ML-powered maps.

**Status:** skeleton (P0 in TODO.rababa/10). Framework abstractions
implemented and tested. Task data modules + configs ready; full
training requires GPU + dataset fetch.

## What this is

One training repo for every ML map in Interscript:

- **rababa_arabic** — Arabic diacritization (adds harakat)
- **rababa_hebrew** — Hebrew diacritization (adds nikud)
- **secryst_thai_ipa** — Thai → IPA transliteration

Each task is a **config + data module**. The training framework is
shared. Adding a new transliteration pair (Khmer → IPA, Japanese →
Romaji) is one new directory under `src/tasks/` — zero edits to
framework code.

## Architecture

```
src/
├── framework/          # SHARED abstractions (MECE)
│   ├── config.py       # TaskConfig loaded from YAML
│   ├── registry.py     # Plugin registry (OCP)
│   ├── data.py         # DataModule ABC
│   ├── model.py        # ModelModule ABC (teacher + student)
│   ├── trainer.py      # BaseTrainer + FineTune + Distill (DRY)
│   ├── evaluator.py    # BaseEvaluator + edit_distance + DER/PER utils
│   ├── exporter.py     # OnnxExporter ABC
│   └── pipeline.py     # TrainingPipeline orchestrator
├── tasks/
│   ├── rababa_arabic/  # config.yaml + data.py + student.py + metrics.py
│   ├── rababa_hebrew/
│   └── secryst_thai_ipa/
└── cli.py              # python -m src.cli train --task rababa_arabic
```

## Design principles (project conventions)

- **OCP** — adding a task = one new directory. Adding a metric, model
  architecture, or data source = one new subclass + `@register_*`
  decorator. Framework code is never edited.
- **MECE** — each module owns one concern. Data has no knowledge of
  model architecture. Model has no knowledge of trainer. Trainer has
  no knowledge of evaluator.
- **DRY** — the epoch loop, edit-distance math, and ONNX export
  wrapper are written once.
- **Model-driven, semantically-driven** — class names mirror domain
  concepts (`RababaArabicData`, `DEREvaluator`, `SecrystThaiIpaStudent`).
- **Performance** — frozen dataclasses for config; lazy imports for
  torch so framework tests run without GPU deps.

## Quick start

```bash
scripts/setup_env.sh               # creates .venv, installs deps
scripts/fetch_data.sh              # fetch raw datasets (set env vars first)
scripts/train.sh rababa_arabic     # full training pipeline
scripts/export.sh rababa_arabic    # export student to ONNX
scripts/publish.sh rababa_arabic   # upload to HuggingFace Hub
```

Or via the CLI directly:

```bash
python -m src.cli list
python -m src.cli train --task rababa_arabic --data-root data --out-root models
python -m src.cli evaluate --task secryst_thai_ipa
python -m src.cli export --task rababa_hebrew
```

## Adding a new task

1. Create `src/tasks/<name>/config.yaml` (copy from an existing task).
2. Create `src/tasks/<name>/data.py` extending `DataModule`, decorated
   with `@register_data_module("<name>_data")`.
3. Create `src/tasks/<name>/student.py` extending `ModelModule`,
   decorated with `@register_model_module("<name>_student")`.
4. Create `src/tasks/<name>/metrics.py` extending `BaseEvaluator`,
   decorated with `@register_evaluator("<metric>")`.
5. Run `python -m src.cli train --task <name>`.

That's it. No framework edits.

## Tests

```bash
pytest -v
```

Framework tests run without torch (CPU-only, fast). Training and ONNX
export tests are gated behind `@pytest.mark.gpu` and require the
`[train]` and `[export]` extras.

## Distribution

Models reach end users through three channels (full plan in
[`TODO.distribution/`](./TODO.distribution/)):

| Channel | Audience | Why |
|---|---|---|
| **GitHub Releases** (primary) | All consumers | Versioned, immutable, checksums, tied to source tags |
| **HuggingFace Hub** (canonical) | Researchers | Model cards, datasets, auto-conversion, inference API |
| **jsdelivr CDN** (edge) | Browser | Edge-cached, CORS-friendly, no rate limits |

Per-task versioning: `rababa_arabic-v1.0.0`, `secryst_thai_ipa-v1.2.0`,
etc. Each release ships fp32 + int8 + int4 variants with SHA256
sidecars, SLSA provenance, and Sigstore signatures.

Distribution phases (P2–P8) are tracked in `TODO.distribution/`. The
first production release lands when phase P6 (first trained model)
completes.

## License

MIT. Model weights are released under their own licenses (see
`docs/model_card.md` per task).
