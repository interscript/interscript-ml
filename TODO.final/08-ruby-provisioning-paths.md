# 08 — Ruby ML provisioning path migration

**Status: pending.** The ported Interscript::ML::Provisioning still
writes secryst-named paths (/var/lib/secryst, ~/.local/share/secryst).

## Steps (TDD)
1. Failing spec: provisioning writes under interscript-ml names.
2. Write paths become interscript-ml/*; legacy secryst paths remain
   READ-ONLY fallbacks for one transition period.
3. INTERSCRIPT_ML_DATA primary, SECRYST_DATA alias (already done —
   verify coverage in specs).
