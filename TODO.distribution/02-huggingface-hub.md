# 02 — HuggingFace Hub integration

**Status:** SPECIFICATION
**Priority:** P3

## Goal

`hf.co/interscript/<task>` is the canonical researcher-facing home for
every Interscript ML model. Model card, datasets, weights, auto-ONNX
conversion, free inference API. All mirrored from this repo via CI on
every release tag.

## Org setup (one-time, manual)

1. Create HuggingFace org: `hf.co/interscript`
   - Owner: `ronaldtse` (or ribose org admin)
   - Visibility: public
   - Billing: free tier sufficient
2. Create service account token: `hf_interscript_ci`
   - Scope: `write` to models + datasets under `interscript/`
   - Stored as GitHub Actions secret: `HF_TOKEN`
3. Add `interscript` to HF org membership (public visibility)

## Per-task model repo

For each task `<name>` (rababa_arabic, rababa_hebrew, secryst_thai_ipa):

```
hf.co/interscript/<name>/
├── README.md            # Model card (auto-generated from src/tasks/<name>/card.md)
├── config.json          # HuggingFace transformers config
├── pytorch_model.bin    # Full-precision teacher (Git-LFS)
├── model.onnx           # Exported ONNX (auto-converted)
├── model-q8.onnx        # Quantized variant
├── tokenizer.json       # Character vocab
├── eval_results.json    # Latest benchmark numbers
└── training_args.json   # Hyperparameters used
```

Model card template (`src/tasks/<name>/card.md`):

```markdown
---
language: [ar]
license: mit
library_name: transformers
tags:
  - arabic
  - diacritization
  - interscript
  - character-level
base_model: Qwen/Qwen3.5-4B-Instruct
metrics:
  - der
pipeline_tag: text2text-generation
---

# Interscript rababa_arabic v<version>

Adds harakat (diacritics) to undiacritized Arabic text.

## Description
...

## Intended use
...

## Training data
...

## Evaluation
| Metric | Value |
|---|---|
| DER | 4.8% |

## How to use
\`\`\`python
from transformers import AutoModel, AutoTokenizer
...
\`\`\`
```

## Datasets

For each task, the processed dataset is published under
`hf.co/datasets/interscript/<source>`:

- `hf.co/datasets/interscript/tashkeela_plus_plus`
- `hf.co/datasets/interscript/wiktionary_thai_ipa`
- `hf.co/datasets/interscript/sna_nikud`

Datasets use HF Datasets library format (Parquet shards). Loadable via:

```python
from datasets import load_dataset
ds = load_dataset("interscript/tashkeela_plus_plus")
```

## Auto-conversion

When PyTorch weights are pushed, HF Hub auto-converts to:
- ONNX (for onnxruntime-web / transformers.js)
- TensorFlow (rarely needed)
- GGUF (for llama.cpp / Ollama — overkill for our sizes, but free)

We rely on the auto-ONNX. The manually exported ONNX in GH Releases
is the canonical browser-targeted variant (smaller, dynamic axes set
the way onnxruntime-web expects).

## Sync workflow

`.github/workflows/release.yml` (after main release step):

```yaml
- name: Upload to HuggingFace Hub
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    python -m src.cli publish \
      --task $TASK \
      --repo interscript/$TASK \
      --out-root models/$TASK
```

`scripts/publish.sh` calls `huggingface_hub` to upload:
1. PyTorch checkpoint
2. Model card
3. Eval results
4. Vocab files

## Why HF as authoring surface

| Concern | GH Releases | HuggingFace |
|---|---|---|
| Model card rendering | plain markdown | structured metadata, search |
| Researcher discovery | none | first-class |
| Dataset hosting | poor | first-class |
| Auto-conversion | none | PyTorch → ONNX/TF/GGUF |
| Inference API | none | free tier |
| Citation tracking | none | BibTeX auto-display |

## Acceptance

- [ ] HF org `interscript` created
- [ ] First model pushed: `hf.co/interscript/rababa_arabic`
- [ ] First dataset pushed: `hf.co/datasets/interscript/tashkeela_plus_plus`
- [ ] CI auto-publishes on tag push
- [ ] Model card template lives at `src/tasks/<name>/card.md`
