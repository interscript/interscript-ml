# heb-diac-small-1.0

Hebrew diacritization (adds nikud), client tier. ByT5-small student
(300M) distilled from the s43 teacher via logit KD — same vocab
(byte-level), so the teacher's soft distributions transfer directly.

Gate: student 30.37% DER vs teacher 24.79% (greedy, Nakdimon IMF test
split, 1,864 long sentences) = +5.58pp, inside the ~5.6pp shrink budget
pre-accepted for this pair (docs/DISTILL-SOURCE-PROMPT.md section 2).

```python
from interscript_ml import Model
model = Model.load("heb-diac-small-1.0")
model.translate("שלום")
```
