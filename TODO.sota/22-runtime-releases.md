# WO22 — runtime release queue (OWNER-GATED, prepared)

Everything staged; version numbers are owner decisions (standing rule):

1. interscript-ts 5.7.0 (plane kind + preserve + parity leg) — unblocks
   the api worker's real dependency (vitest alias is dev-only today).
2. interscript-ruby gem next (plane preserve + index-v9 + bound knob).
3. interscript-py (pip) — still blocked on the previous developer's PyPI
   trusted-publisher fix; everything staged per TODO.final/01.
4. api-worker deploy: after (1), drop the vitest alias + type
   augmentation and deploy with the real package.
