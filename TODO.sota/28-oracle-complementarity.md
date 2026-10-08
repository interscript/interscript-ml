# WO28 — cross-system complementarity: oracle ceiling real, naive voting catastrophic

Measured on SadeedDiac-25 word-exact over 52,906 words (r7, r8b,
plane-large stitched predictions):

| selection | exact-word |
|---|---|
| best solo (r8b) | 93.36% |
| **oracle any-of-3** | **96.78%** |
| naive 2-of-3 word vote | (word-level 93.81%) → **DER 28.77** |

Findings: (1) +3.42 points of words are covered by at least one system
— the complementarity ceiling is REAL and large. (2) Naive voting is
CATASTROPHIC (DER 28.77, 74.6% under-diacritized): the systems'
diacritization CONVENTIONS conflict at sentence level — mixing words
from different conventions produces incoherent output under DER even
when each word is individually "exact". (3) Harvesting the ceiling
requires a LEARNED, convention-aware selection model (train a router
on the agreement features against Sadeed gold) — candidate future WO,
not free. This empirically closes generalist mixing in OUTPUT space
exactly as WO17 closed it in TRAINING space.
