# 13 — Telemetry + privacy

**Status:** SPECIFICATION
**Priority:** P6

## Goal

Know how models are used in the wild (so we can prioritize), without
collecting anything users would object to.

## Privacy posture: opt-in, anonymous, useful

- **No user-identifying data.** No IPs, no user agents, no cookies.
- **No input content.** Never log what users transliterate.
- **Aggregate only.** Counts, not sequences.
- **Opt-in.** Off by default. Explicit "Share anonymous usage" toggle.

## What we collect (when opt-in)

```json
{
  "schema_version": 1,
  "reported_at": "2026-08-01T12:00:00Z",
  "client": {
    "runtime": "interscript-ts",
    "runtime_version": "0.2.0",
    "platform": "browser",
    "onnxruntime_version": "1.17.1"
  },
  "models_used": [
    {
      "task": "rababa_arabic",
      "version": "1.0.0",
      "calls_last_week": 423,
      "avg_input_chars": 18,
      "p95_latency_ms": 24
    }
  ],
  "errors": [
    { "type": "load_failed", "count": 0 },
    { "type": "checksum_mismatch", "count": 0 }
  ]
}
```

That's it. No content, no user IDs, no IPs.

## What we use it for

- **Prioritization.** If rababa_arabic has 50K calls/week and rababa_hebrew has 200, we invest in Arabic.
- **Regression detection.** If p95 latency jumps 30% on a new version, we investigate.
- **Variant sizing.** If most users are on mobile, we ship smaller variants.
- **Error rate tracking.** If checksum mismatches spike, the CDN is corrupting files.

## How it's sent

Browser → POST to `telemetry.interscript.org` (Cloudflare Worker, 24h retention, then aggregate).

```typescript
// interscript-ts
if (userConsentStore.hasConsentedToTelemetry()) {
  await fetch("https://telemetry.interscript.org/v1/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(weeklyReport()),
  })
}
```

Ruby → same endpoint, same payload shape:
```ruby
Interscript::Telemetry.report_weekly if Interscript.config.telemetry_opt_in
```

## Consent UX

interscript.org shows a one-time banner:

> Help improve Interscript
>
> Share anonymous usage stats (call counts, latency, errors)? No content is ever sent.
>
> [Yes, share] [No thanks] [Learn more]

`Learn more` opens `/privacy` explaining exactly what's collected, with
the JSON schema above as the source of truth.

## Server side

Cloudflare Worker (free tier) receives reports:
1. Validates schema.
2. Strips any unexpected fields (defensive).
3. Adds to an aggregate counter (per task / version / runtime).
4. Drops individual reports after aggregation.

Daily export to a public dashboard at `interscript.org/stats`:

```
Active installs (last 30 days): 1,234
Top tasks: rababa_arabic (89%), secryst_thai_ipa (11%)
Average latency p95: 23ms
Error rate: 0.02%
```

## Why bother

Without telemetry, we're flying blind. The first 6 months of post-release will involve:
- "Is anyone using Hebrew?" — we'd have no idea.
- "Did v1.1.0 regress latency?" — we'd have to ask users to benchmark.
- "What's the mobile vs desktop split?" — pure guess.

With opt-in telemetry, we have data to back decisions. Crucial for prioritizing the next year of work.

## Acceptance

- [ ] Opt-in toggle in interscript-ts + interscript-ruby
- [ ] Cloudflare Worker endpoint live
- [ ] Schema documented publicly
- [ ] `/privacy` page on interscript.org
- [ ] Aggregate dashboard on interscript.org/stats
