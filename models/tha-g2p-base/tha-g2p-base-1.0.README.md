# tha-g2p-base-1.0

Thai grapheme-to-phoneme (IPA). Client-tier ByT5-base (580M) student
distilled from the B-K/umt5-thai-g2p teacher via sequence-level KD:
48,757 beam-4 teacher-generated labels over the Kaikki + epitran-Wikipedia
corpus (deduplicated, degenerate outputs filtered).

Gate: student 9.19% PER vs teacher 4.43% on the same harness
(1,219 Kaikki test sentences, beam-4, corpus-level PER) — +4.76pp,
inside the +5pp distillation budget (docs/DISTILL-SOURCE-PROMPT.md).

Note on the teacher: the secryst 2.32%-PER umt5 artifacts are
unrecoverable (transformers 5.15 save drops the untied umt5 lm_head;
the volume's epitran corpus is tone-less). The 2.32% tier re-enters
this pipeline when secryst ships repaired artifacts; this model
distills the best verified teacher available (4.43%).

```python
from interscript_ml import Model
model = Model.load("tha-g2p-base-1.0")
model.translate("สวัสดี")
```
