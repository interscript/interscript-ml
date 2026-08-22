# urd-diac-1.0

Urdu diacritization (adds haraqat). Byte-level seq2seq (ByT5-small):
the tokenizer is raw UTF-8 bytes (pad=0, EOS=1) — no vocab files.
IMF v1 artifact; format spec: interscript/interscript-ml docs/imf-v1.md.

- decoder: kv greedy (plain fallback included in the zip)
- metrics: CER 3.74 on 11,940 held-out —
  rababa-urdu/docs/RESULTS.md#diacritization-urdu-text--text--haraqat
- trained from: rababa-urdu modal_app_diacrit.py run-001
  (urdu-diacrit-checkpoints:/urdu_diacrit/run-001/best)
- license: BSD-3-Clause

## Usage

Ruby (secryst gem, the Ruby binding of interscript-ml):

```ruby
require "secryst"
translator = Secryst::Translator.new(model: "urd-diac-1.0")
translator.translate("اردو")
```

TypeScript (@interscript/ml):

```ts
import { loadModel } from "@interscript/ml";
const model = await loadModel("urd-diac-1.0");
await model.translate("اردو");
```

Python (interscript-ml):

```python
from interscript_ml import Model
model = Model.load("urd-diac-1.0")
model.translate("اردو")
```

All three runtimes verify the sha256 of every ONNX member in this zip
against metadata.yaml before loading.
