# 07 — Release hub cleanup (interscript/interscript)

**Status: pending.** The hub now works (gem 3.0.0 via OIDC) but clones
dead repos and carries campaign TODO dirs.

## Steps
1. bootstrap.rb: drop interscript-js (dead) from the package set; keep
   ruby + maps (+ ts if the JS leg is ever revived — it should consume
   interscript-ts instead).
2. Move TODO.complete/TODO.rababa/TODO.secryst into docs/archive/ (they
   are campaign records, not live plans).
3. Confirm the OIDC gem path stays the single release entry point.
