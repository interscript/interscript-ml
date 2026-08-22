# heb-diac-1.0

Hebrew diacritization (adds nikud). Byte-level seq2seq (ByT5-base):
the tokenizer is raw UTF-8 bytes (pad=0, EOS=1) — no vocab files.
IMF v1 artifact; format spec: interscript/interscript-ml docs/imf-v1.md.

- decoder: kv greedy (plain fallback included in the zip)
- metrics: greedy DER 29.0% (the v1 runtime path); beam=4 DER 17.46%
  (reference quality — beam search is not in v1 runtimes) —
  rababa/docs/RESULTS.md#hebrew-diacritization
- trained from: rababa train_hebrew_seeds.py s43 run-001
  (rababa-checkpoints:/rababa_hebrew_byt5_s43/run-001/best)
- license: BSD-3-Clause

## Usage

Ruby (secryst gem, the Ruby binding of interscript-ml):

```ruby
require "secryst"
translator = Secryst::Translator.new(model: "heb-diac-1.0")
translator.translate("שלום")
```

TypeScript (@interscript/ml):

```ts
import { loadModel } from "@interscript/ml";
const model = await loadModel("heb-diac-1.0");
await model.translate("שלום");
```

Python (interscript-ml):

```python
from interscript_ml import Model
model = Model.load("heb-diac-1.0")
model.translate("שלום")
```

All three runtimes verify the sha256 of every ONNX member in this zip
against metadata.yaml before loading.
