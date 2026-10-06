# 08 — Ruby ML provisioning path migration

**Status: DONE** (2026-10-06, PR #795) — user-local interscript-ml write paths, legacy fallback, TDD.
writes secryst-named paths (/var/lib/secryst, ~/.local/share/secryst).

## Steps (TDD)
1. Failing spec: provisioning writes under interscript-ml names.
2. Write paths become interscript-ml/*; legacy secryst paths remain
   READ-ONLY fallbacks for one transition period.
3. INTERSCRIPT_ML_DATA primary, SECRYST_DATA alias (already done —
   verify coverage in specs).
