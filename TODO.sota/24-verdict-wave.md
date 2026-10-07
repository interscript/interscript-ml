# WO24 — r8 verdict processing + the dose-response question

In flight: r8b (QCRI 33% dose) and r8c (both) — dual-surface gates
running; thai tiny3 resumed from student.pt (decode-only, ~20 min).

WO17's r8a result reframes the arms: news-register silver MOVES OOD
(-0.67 WER at a 2.7% dose) but costs ID (+0.62 DER). The remaining
question is the DOSE-RESPONSE: if r8b/r8c also regress ID, the
follow-up (r8d) is a gentle arm — silver capped at 5% of gold tokens,
lr 3e-5, r7-init — targeting OOD improvement with ID held. Ship rule
unchanged: dominance on BOTH surfaces or documented negative.

Also pending: run-026 negative recorded (base stays shipped);
thai tiny3 verdict → WO12 contract or closure.

**The register-specialist option (owner decision, everything stageable):**
r8a is better on OOD (16.71/10.83) while r7 holds ID — mirror of the
two-ISC-families doctrine (bibliographic vs phonological: ship both,
never conflate). ara-diac-news-1.0 (r8a export) as a news-domain
specialist alongside ara-diac-2.0; clients pick by domain. Dose knobs
(R8_PSEUDO_CAP/R8_LR, train#115) make the gentle arm launchable the
moment r8b/r8c confirm the dose-response direction.
