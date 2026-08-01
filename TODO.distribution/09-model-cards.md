# 09 — Model cards per task

**Status:** SPECIFICATION
**Priority:** P3

## Goal

Every model has a complete, accurate, HuggingFace-rendered model card.
The card is authored next to the code (in `src/tasks/<name>/card.md`),
auto-filled with metrics from the latest benchmark, and synced to HF
on every release.

## Template

`src/tasks/<name>/card.md`:

```markdown
---
language: [<iso-639-1>]
license: mit
library_name: transformers
base_model: Qwen/Qwen3.5-4B-Instruct
tags:
  - interscript
  - <task_type>
  - <script>
pipeline_tag: text2text-generation
metrics:
  - <metric>
---

# Interscript <task_full_name> v<VERSION>

<one-sentence description>

## Description

<longer description: what this does, who it's for, example input/output>

## Intended uses

- ✅ <use case 1>
- ✅ <use case 2>

## Out-of-scope uses

- ❌ <use case 1> — reason
- ❌ <use case 2> — reason

## Training data

| Source | Size | License |
|---|---|---|
| <dataset_1> | <X> samples | <license> |
| <dataset_2> | <Y> samples | <license> |

## Training procedure

### Teacher

- Base: `<base_model>`
- Method: LoRA fine-tune (r=<>, alpha=<>)
- Hardware: 1× NVIDIA A100 80GB
- Wall time: <X> hours
- Hyperparameters: lr=<>, batch=<>, epochs=<>

### Student

- Architecture: char-level transformer (<layers>L, <dim>d, <heads>h)
- Method: distillation (KL temperature=<>, alpha=<>)
- Parameters: <X>M
- Hardware: 1× NVIDIA A100 80GB
- Wall time: <X> hours

## Evaluation

### Results

| Metric | Value |
|---|---|
| <METRIC> | <X>% |
| Latency p95 | <X>ms/word |
| Size (fp32) | <X> MB |
| Size (q8) | <X> MB |

### Test set

<description: held-out verses, source, license>

## Bias + risks

<domain-specific risks. E.g. for Arabic: heavy Quranic bias in Tashkeela++ may produce unexpected diacritization on secular text>

## How to use

### In JavaScript (browser/Node)

\`\`\`js
import { transliterateAsync } from "interscript-ts"
const result = await transliterateAsync("<map_code>", "<input>")
\`\`\`

### In Ruby

\`\`\`ruby
require "interscript"
Interscript.transliterate("<map_code>", "<input>")
\`\`\`

### Direct ONNX (any language)

\`\`\`bash
curl -LO https://github.com/interscript/ml-models/releases/download/<task>-v<VERSION>/<task>.onnx
# Use onnxruntime to load + run
\`\`\`

## Citation

\`\`\`bibtex
@software{interscript_<task>,
  title = {Interscript <task_full_name>},
  url = {https://github.com/interscript/ml-models},
  version = {<VERSION>}
}
\`\`\`
```

## Substitution variables

The CI release job fills in:
- `<VERSION>` — from tag
- `<METRIC>` + value — from `benchmarks.json`
- `<X> samples` — from data module's `len(train)` + `len(val)`
- `<X>M parameters` — from model class's `count_parameters()`
- `<X> hours` — from training log

Static values (description, datasets, intended uses) are authored by hand once per task.

## Where the rendered card lives

- HF Hub: `hf.co/interscript/<task>/blob/main/README.md` (rendered with HF metadata)
- GH Release: attached as `README.md` asset
- npm `@interscript/model-<task>`: included in package
- interscript.org: rendered as HTML on `/ml/<task>` page

## Internationalization

Cards are in English by default. Each task can carry translated cards:

```
src/tasks/rababa_arabic/
├── card.md             # English (canonical)
├── card.ar.md          # Arabic
├── card.zh.md          # Chinese
└── card.he.md          # Hebrew (for Hebrew task)
```

HF Hub shows the language matching `Accept-Language` header.

## Acceptance

- [ ] `card.md` template lives in `src/tasks/<name>/`
- [ ] CI substitutes `<VARIABLES>` at release time
- [ ] HF card rendered for first task
- [ ] Citation auto-generated with BibTeX
