# 14 — Sunset / kill switch

**Status:** SPECIFICATION
**Priority:** P3

## Goal

When something goes wrong (bad model, compromised signing key, dataset
license dispute), we can revoke a release within minutes — globally,
across every consumer — without breaking anyone's app.

## Threat model

| Scenario | Detection | Recovery |
|---|---|---|
| Bad weights (DER > 20%) | User report, regression alert | Retract, republish |
| Compromised signing key | Sigstore Rekor anomaly | Rotate key, re-sign all releases |
| License dispute (dataset) | Legal report | Pull affected version, migrate users |
| Security vuln (model exploit) | CVE report | Force upgrade via manifest bump |
| HF outage | Uptime monitor | Already mitigated: GH releases + jsdelivr fallback |
| GH outage | Uptime monitor | HF + jsdelivr continue serving |

## Manifest as the kill switch

`@interscript/models` manifest is the source of truth for "what's the current version". Pull the entry, every consumer sees the retraction.

```json
{
  "rababa_arabic": {
    "version": "1.0.1",
    "status": "active",
    "supersedes": [
      { "version": "1.0.0", "status": "retracted", "reason": "DER regression from new training data", "retracted_at": "2026-08-15T12:00:00Z" }
    ]
  }
}
```

`interscript-ts` startup:
1. Fetches manifest from npm (cached up to 24h).
2. For each loaded model, checks `status`.
3. If `retracted`: emit warning + auto-upgrade to current active version.
4. If retraction is critical (security vuln): refuse to load old version, force upgrade.

## Yanking from distribution points

A retraction script `scripts/retract.sh <task> <version>` performs:

1. **Manifest bump** — publish new `@interscript/models` with `status: retracted`.
2. **HF Hub** — add `retracted: true` to model card metadata, pin a banner.
3. **GH Release** — convert to "draft" (preserves asset URLs but hides from listing).
4. **Slack / mailing list** — automated announcement.

Reverse with `scripts/restore.sh` if the retraction was a mistake.

## Critical security response

If a release is exploited in the wild:

1. **Within 1 hour**: retract via manifest. Most consumers auto-upgrade within 24h.
2. **Within 4 hours**: revoke Sigstore identity via Rekor (public log entry).
3. **Within 24 hours**: publish CVE, write post-mortem on `/blog/incident-<id>`.
4. **Within 72 hours**: cut new major release with fix.

## Cached copies

Retraction can't recall copies already on disk (`~/.cache/interscript/`).
Mitigations:
- **Cache TTL**: 7 days. After that, revalidate against manifest.
- **Forced purge**: `interscript purge --task rababa_arabic --version 1.0.0` removes the local cache. CI can run this at scale.
- **Critical alerts**: interscript-ts prints to console on startup if a cached version is retracted (visible in browser DevTools, Node console, Rails logs).

## Why this matters

Every distribution system needs a kill switch. Otherwise:
- A bug in v1.0.0 propagates to every consumer before we notice.
- A license dispute forces us to keep serving a bad file because we can't recall it.
- A security vuln sits unfixed in users' caches for years.

Building this on day one (even if we never need it) is far cheaper than retrofitting after an incident.

## Acceptance

- [ ] `@interscript/models` manifest carries `status` field
- [ ] `scripts/retract.sh` and `scripts/restore.sh` work end-to-end
- [ ] interscript-ts warns on startup if cached version is retracted
- [ ] Game-day exercise: simulate a retraction, verify consumers respond
