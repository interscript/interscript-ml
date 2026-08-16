# khm-latn-1.0 (fp16)

Khmer → Latin transliteration. Byte-level seq2seq (ByT5-small):
the tokenizer is raw UTF-8 bytes (pad=0, EOS=1) — no vocab files.
IMF v1 artifact; format spec: interscript/ml-models docs/imf-v1.md.

- precision: fp16 (mixed: LayerNorm parameters in fp32)
- decoder: kv greedy (plain fallback included in the zip)
- metrics: CER 27.42 / EM 59.66 on 895 held-out pairs —
  secryst/docs/RESULTS.md#khmer-transliteration-2026-08-14
- trained from: secryst train_khmer_byt5.py run-001
  (secryst-checkpoints:/khmer_byt5/run-001/best)
- license: BSD-3-Clause

## Usage

Ruby (secryst gem, the Ruby binding of interscript-ml):

```ruby
require "secryst"
translator = Secryst::Translator.new(model: "khm-latn-1.0")
translator.translate("ភាសា")
```

TypeScript (@interscript/ml):

```ts
import { loadModel } from "@interscript/ml";
const model = await loadModel("khm-latn-1.0");
await model.translate("ភាសា");
```

Python (interscript-ml):

```python
from interscript_ml import Model
model = Model.load("khm-latn-1.0")
model.translate("ភាសា")
```

All three runtimes verify the sha256 of every ONNX member in this zip
against metadata.yaml before loading.
