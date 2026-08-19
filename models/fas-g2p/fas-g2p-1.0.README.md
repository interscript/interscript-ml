# fas-g2p-1.0

Persian grapheme-to-phoneme. Raw Persian sentence (no prefix) in →
space-separated Latin phonemes out. The v1 ByT5-small teacher shipped
directly as the client-tier model — it is already byte-level at
client-tier size, so no distillation step applies (the +5pp gate is a
distillation contract; here the teacher IS the artifact).

```python
from interscript_ml import Model
model = Model.load("fas-g2p-1.0")
model.translate("سلام")
```
