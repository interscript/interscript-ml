# 04 — rababa-* corpora family consolidation

**Status: pending.** 11 corpus/data repos in the interscript org
(rababa-farsi, -urdu, -urdu-corpus, -persian-corpus, -tashkeela-full,
-hebrew-distilled, -hewiki, -arwiki, -tashkeela, -sefaria) plus
rababa-models and data-* in the secryst org.

## Steps
1. Inventory each: size, last activity, which training runs consume it.
2. Corpora with no code become directories under interscript-train
   (data/), preserving history via git subtree or archived mirror +
   pointer. Keep any repo that external processes push to.
3. Migrate secryst org data repos (data-arabic-pointing,
   data-thai-interscript, data-khmer-translit, rababa-models) into the
   same structure.
4. Archive emptied repos (never delete).
