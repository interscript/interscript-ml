# 02 — Website collapse: one site, one model page

**Status: DONE** (2026-10-06) — /ml current (plane rows, HF links, index-v8 counts, PR #207 on interscript.github.io); secryst.github.io redirects to it (PR #8).
family and flagship leaderboard; secryst.github.io still serves its own
zoo — two model pages exist (split-brain confirmed).

## Steps
1. Port the secryst.github.io/models content that /ml lacks: flagship
   section (ara-diac-2.0, SadeedDiac-25 leaderboard), plane rows
   (large 2.74 / small 3.59 + on-device numbers), subset-retraction
   disclosure, HF org + collection links.
2. Verify /ml catalog completeness against models.yaml index-v7 (26
   entries; script the diff, never hand-count).
3. secryst.github.io becomes a redirect stub to interscript.org (repo
   kept, never deleted).
4. Cross-link runtimes' READMEs to /ml.
