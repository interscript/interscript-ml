# 09 — interscript-ml → interscript-models (+ index-v8)

**Status: owner direction agreed (plural form recommended to mirror
interscript-maps).** The repo is the model catalog/contract layer; the
"ml" name misleads (it is the STABLE side, not the ML code).

## Why v8 must ride along
Three shipped runtimes pin index URLs at
github.com/interscript/interscript-ml/releases/download/... — after a
repo rename those depend on redirects. The clean fix:

## Steps
1. Rename repo → interscript-models.
2. Cut index-v8 with canonical interscript-models URLs (models.yaml
   unchanged in content).
3. Runtime patch bumps pinning v8: npm 5.6.x, gem 3.0.x, pip with its
   debut. Exact-pin specs updated test-first.
4. pip tools package naming (interscript-ml 0.1.1 / interscript-ml-tools)
   follows the owner's call in the same window.
5. HF workflows and READMEs re-pointed; historical RESULTS/anchors stay
   as written.
