# tha-g2p-small-1.0

Thai grapheme-to-phoneme (IPA). Client-tier ByT5-small (300M) student
distilled from the secryst umt5 Thai teacher (2.32% PER, public baseline
6.37%) via sequence-level KD: the teacher generated 23,295 labels with
its own sentencepiece tokenizer and the student trained CE on them with
the canonical byte table.

First model of the distillation campaign
(docs/DISTILL-SOURCE-PROMPT.md); identical IMF v1 contract to the
server-tier models — dynamic fetch, sha256-verified, KV decode.

```python
from interscript_ml import Model
model = Model.load("tha-g2p-small-1.0")
model.translate("สวัสดี")
```
