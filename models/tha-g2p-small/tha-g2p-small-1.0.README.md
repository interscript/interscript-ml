# tha-g2p-small-1.0

Thai grapheme-to-phoneme (IPA). Client-tier ByT5-small (300M) student
for the int8 release (~300MB): sequence-level KD from the verified
B-K/umt5-thai-g2p-v2-0.5k teacher over 48,757 beam-4 labels, trained
with the canonical byte table.

Capacity-limited at +7.63pp over the teacher (12.06% vs 4.43% PER) —
inside the +5pp server-tier gate is tha-g2p-base-1.0; this is the
smallest artifact that does not collapse (from-scratch 33M/70M students
and pruned rungs all fail — see the frontier table in
docs/RESULTS.md). Identical IMF v1 contract to every other model:
dynamic fetch, sha256-verified, KV decode.

```python
from interscript_ml import Model
model = Model.load("tha-g2p-small-1.0")
model.translate("สวัสดี")
```
