# heb-diac-1.1

Hebrew diacritization (adds nikud). Byte-level seq2seq (ByT5-base):
the tokenizer is raw UTF-8 bytes (pad=0, EOS=1) — no vocab files.
IMF v1 artifact; format spec: interscript/interscript-ml docs/imf-v1.md.

- decoder: kv greedy (plain fallback included in the zip)
- metrics: greedy DER 16.44% — the v1 runtime path — and beam=4 DER
  16.43% (reference quality): greedy ≈ beam-4 for this model, unlike
  s43 —
  rababa/docs/RESULTS.md#hebrew-diacritization
- replaces heb-diac-1.0 (s43: greedy 29.0 / beam-4 17.46) — the
  runtime path improves from 29.0% to 16.44% DER
- trained from: rababa train_hebrew_s46.py — phonikud curriculum
  (1.5M machine-labeled knesset weak-pretrain + 73.8K hewiki garnish,
  then gold fine-tune; weak stage supplies a nikud prior, gold sets
  the ceiling)
  (rababa-checkpoints:/rababa_hebrew/run-s46-phonikud-plus/run-002-gold-ft/best)
- license: BSD-3-Clause

## Usage

Ruby (secryst gem, the Ruby binding of interscript-ml):

```ruby
require "secryst"
translator = Secryst::Translator.new(model: "heb-diac-1.1")
translator.translate("שלום")
```

TypeScript (@interscript/ml):

```ts
import { loadModel } from "@interscript/ml";
const model = await loadModel("heb-diac-1.1");
await model.translate("שלום");
```

Python (interscript_ml):

```python
from interscript_ml import load_model
model = load_model("heb-diac-1.1")
model.translate("שלום")
```
