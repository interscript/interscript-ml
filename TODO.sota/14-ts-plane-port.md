# WO14 — ts plane runtime (TODO.final 11 remainder)

1. loader.ts: accept kind=plane zips (metadata contract: members
   sha256s, classes.json, k_passes) alongside imf-v1.
2. registry: registerModel plane kind -> PlaneModel (K-pass, per-char
   majority vote, render).
3. Tests: tiny-plane fixture parity with py/ruby; pin test.
4. Add the ts leg to neural-parity.yml.
