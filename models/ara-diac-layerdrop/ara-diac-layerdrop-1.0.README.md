# ara-diac-layerdrop-1.0

The depth-cut client rung: ByT5-small with the encoder halved
(layers copied verbatim — width survives, depth is the compressible
axis). Muon + r7 labels + 6 epochs, the full lever set.

- Full-set DER: **5.78** (delta CI [3.03, 3.49]; full-depth peer 4.57)
- ~190M parameters; int8 ~190MB; **int4 ~95MB — the browser budget**
- The capacity law in one artifact: halving depth costs 1.21pp where
  every width surgery collapsed

```python
from interscript_ml import Model
model = Model.load("ara-diac-layerdrop-1.0")
model.translate("قوله فحكمها في الوفاة")
```
