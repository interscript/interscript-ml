# 10 — Version-generation policy across runtimes

**Status: owner decision pending.** The same runtime generation ships
as npm 5.6.0 / gem 3.0.0 / pip 0.1.0(stale) — three major lines for one
product.

## Options
A. Independent per-registry versions; publish the policy that cross-
   runtime conformance (goldens + index pin) is the real contract.
B. Aligned generation majors (e.g. all 3.x = unified-dispatch era) with
   per-registry minors.

Recommend A (documented) unless the owner wants the marketing clarity
of B. Either way: /ml and READMEs state which index each release pins.
