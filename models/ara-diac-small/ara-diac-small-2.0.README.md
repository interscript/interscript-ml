# ara-diac-small-2.0

Arabic diacritization (haraqat restoration). Client-tier ByT5-small
student, identical corpus/labels/epochs to ara-diac-small-1.0 with one
variable changed: **the Muon optimizer** (E3 factorial). That single
change closes 2.96pp of the 5.68pp teacher-student gap:

- 1.0 (AdamW): 8.26 full-set windowed DER-CE
- **2.0 (Muon): 5.29** (teacher r6: 2.58; in-run reproduction 2.60)

The strict teacher+0.5pp gate is still missed; the residual decomposes
as ~0.70pp capacity + ~2.25pp domain coverage (E2/E3 factorial), and an
r7-teacher re-distillation is in flight. Identical IMF v1 contract:
dynamic fetch, sha256-verified, KV decode, margins JSON alongside.

```python
from interscript_ml import Model
model = Model.load("ara-diac-small-2.0")
model.translate("قوله فحكمها في الوفاة")
```
