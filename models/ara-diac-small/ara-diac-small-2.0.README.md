# ara-diac-small-2.0

Arabic diacritization (haraqat restoration). Client-tier ByT5-small
student — the two measured wins of the campaign compounded on the same
architecture and artifact size as 1.0: the **r7 canonical teacher**
(2.2864; fresh greedy labels) and the **Muon optimizer** (E3-adopted).

- 1.0 (r6 labels, AdamW): 8.26 full-set windowed DER-CE
- 2.0 (r7 labels, Muon): **4.82** (teacher r7 in-run: 2.289)

A 42% error reduction, pre-registered as E4 (gate ≤ 6.26; prediction
4.3–5.0 — landed at 4.82). The strict teacher+0.5pp gate is still
missed (+2.53pp, disclosed); the E2/E3 factorial attributes the
residual to domain coverage. Identical IMF v1 contract: dynamic fetch,
sha256-verified, KV decode, margins JSON alongside.

```python
from interscript_ml import Model
model = Model.load("ara-diac-small-2.0")
model.translate("قوله فحكمها في الوفاة")
```
