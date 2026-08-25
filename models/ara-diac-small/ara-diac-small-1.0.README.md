# ara-diac-small-1.0

Arabic diacritization (haraqat restoration). Client-tier ByT5-small
student distilled from the r6 teacher via sequence-level KD: 29,322
greedy teacher labels over 1,400-byte windows of the r5-units corpus,
3 epochs.

Capacity-limited at +2.34pp windowed DER-CE over the teacher (3.66% vs
1.32% on the same 300-paragraph Sadeed harness) — the strict +0.5pp
gate is met by the teacher release (ara-diac-1.0). Shipped as the
Arabic client rung per the documented Thai client-tier precedent: a
working student at a fraction of the artifact size, with the miss
measured and disclosed. Identical IMF v1 contract: dynamic fetch,
sha256-verified, KV decode.

```python
from interscript_ml import Model
model = Model.load("ara-diac-small-1.0")
model.translate("قوله فحكمها في الوفاة")
```
