# 05 — secryst package deprecation shims (final releases)

**Status: DONE** (2026-10-06) — npm 0.3.2 / PyPI 0.2.2 / gem 1.1.2 published with moved-to-interscript notices; all secryst repos archived.
releases (npm 0.3.1, PyPI 0.2.1, gem 1.1.1) with index-v7 pins. Old
users keep working; nobody is told where the line continues.

## Steps
1. Final shim releases that only change metadata/message to point at
   `interscript` (+[ml] where applicable): npm secryst 0.3.2, PyPI
   secryst 0.2.2, gem secryst 1.1.2. Owner confirms numbers.
2. Mark npm package deprecated (`npm deprecate` message), PyPI via
   the shim description, RubyGems via the gemspec description.
3. Archive secryst/{secryst,secryst-py,secryst-ts,secryst-train} after.
