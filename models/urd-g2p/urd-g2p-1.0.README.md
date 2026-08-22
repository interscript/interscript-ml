# urd-g2p-1.0

Urdu → IPA grapheme-to-phoneme conversion. Byte-level seq2seq
(ByT5-small): the tokenizer is raw UTF-8 bytes (pad=0, EOS=1) — no vocab
files. IMF v1 artifact; format spec: interscript/interscript-ml docs/imf-v1.md.

- decoder: kv greedy (plain fallback included in the zip)
- metrics: CER 14.77 / EM 33.6 on 12,699 held-out words —
  rababa-urdu/docs/RESULTS.md#g2p-urdu-text--ipa
- trained from: rababa-urdu modal_app.py run-001
  (urdu-g2p-checkpoints:/urdu_g2p/run-001/best)
- license: BSD-3-Clause

## Usage

Ruby (secryst gem, the Ruby binding of interscript-ml):

```ruby
require "secryst"
translator = Secryst::Translator.new(model: "urd-g2p-1.0")
translator.translate("اردو")
```

TypeScript (@interscript/ml):

```ts
import { loadModel } from "@interscript/ml";
const model = await loadModel("urd-g2p-1.0");
await model.translate("اردو");
```

Python (interscript-ml):

```python
from interscript_ml import Model
model = Model.load("urd-g2p-1.0")
model.translate("اردو")
```

All three runtimes verify the sha256 of every ONNX member in this zip
against metadata.yaml before loading.
