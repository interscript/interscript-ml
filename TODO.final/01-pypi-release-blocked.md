# 01 — pip `interscript` release (BLOCKED: external)

**Status: BLOCKED** on the previous developer fixing the PyPI trusted
publisher for the `interscript` project (the workflow's OIDC claim is
rejected `environment: MISSING` vs the registered publisher; pypi.org
side only). Everything else is ready.

## Already done
- interscript-py main carries the unified surface: `interscript.ml`
  ([ml] extra), transliterate dispatch by index kind, index-v7 pin,
  INTERSCRIPT_ML_* env vars (PR #24).
- v0.2.0 tag is burned (a failed release run, never published) — the
  debut MUST NOT reuse it.

## Steps when unblocked
1. Owner states the debut version (0.3.0 per plan, or 1.0.0 to signal
   maturity — owner decision).
2. If the TP entry expects a GitHub environment: either the pypi.org
   entry drops it, or the release job declares `environment: <name>`.
3. Tag/ship via the repo release path; verify PyPI listing + `pip
   install interscript[ml]` smoke.
4. Close with the deprecation shim (TODO 05) once shipped.
