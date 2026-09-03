# ara-diac-small-2.1

Arabic diacritization (haraqat restoration). Client-tier ByT5-small
student — the 2.0 recipe (r7 canonical teacher, Muon) held for
**double the epochs** (6 instead of 3, 39,018 steps): the epochs lever
moves the rung 4.82 → **4.5701** full-set windowed DER-CE.

The paired-bootstrap delta vs the teacher (same harness, seed 42,
1,000 resamples) is [1.91, 2.35] — disjoint from the 2.0 rung's
[2.36, 2.82]: the improvement is statistically real, and the frontier
separations hold end to end (1.0 8.26 → lite 5.78 → 2.0 4.82 → 2.1
4.57 → teacher 2.29).

The epochs result reframes the residual: doubling training buys
−0.25pp, so undertraining is not the dominant term — the E2/E3 domain
attribution substantially stands. The strict teacher+0.5pp gate
remains missed (disclosed); the causal follow-ups (E6 register swap —
negative; G2b 5× classical corpus — the add-direction test) are in
docs/RESULTS.md.

```python
from interscript_ml import Model
model = Model.load("ara-diac-small-2.1")
model.translate("قوله فحكمها في الوفاة")
```

Same IMF v1 contract as 2.0: dynamic fetch, sha256-verified, KV
decode, margin report alongside the parity block. Checkpoint:
`rababa_arabic_distill_small/run-007-r7-muon-6ep/best`.
