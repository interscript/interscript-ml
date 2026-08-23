# ara-diac-1.0

Arabic diacritization (adds haraqat / tashkeel). Byte-level seq2seq
(ByT5-base): the tokenizer is raw UTF-8 bytes (pad=0, EOS=1) — no
vocab files. IMF v1 artifact; format spec:
interscript/interscript-ml docs/imf-v1.md.

- decoder: kv greedy (plain fallback included in the zip)
- metrics: Total DER 2.5793% / Morphological DER 1.5317% — greedy
  decode, SadeedDiac-25, windowed zero-skip at 1400 bytes, 1,200
  paragraphs —
  rababa/docs/RESULTS.md#r6-verdict-table-sadeeddiac-25-2026-08-21
- out-of-domain (WikiNews-2024, multi-ref): WER 19.82 / DER 12.46
- trained from: rababa train_arabic_r6.py run-006-morph —
  morphological aux-task (plain + "TAG: "-prefixed iʿrāb stream from
  qalsadi labels), init from r5
  (rababa-checkpoints:/rababa_arabic_byt5/run-006-morph/best)
- license: BSD-3-Clause

## Inference contract

- Plain input → diacritized output. The morph "TAG: " stream NEVER
  appears at inference (it is a training-time supervision format only).
- Greedy decoding is the reference path: beam-4 was probed and is flat
  (Total DER 2.5588 vs 2.5793) — beam buys nothing here.
- For long inputs, split at word boundaries into ≤1400-byte windows
  (zero-skip) and stitch; generation cap 2x window size.

## Usage

Ruby (secryst gem, the Ruby binding of interscript-ml):

```ruby
require "secryst"
translator = Secryst::Translator.new(model: "ara-diac-1.0")
translator.translate("مكتبة")
```

TypeScript (@interscript/ml):

```ts
import { loadModel } from "@interscript/ml";
const model = await loadModel("ara-diac-1.0");
await model.translate("مكتبة");
```

Python (interscript_ml):

```python
from interscript_ml import load_model
model = load_model("ara-diac-1.0")
model.translate("مكتبة")
```
