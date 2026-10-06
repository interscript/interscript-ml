# Versioning policy — the three runtimes

The three runtime packages (npm `interscript`, RubyGems `interscript`,
PyPI `interscript`) version **independently**. There is no cross-registry
version alignment, and none is planned: the compatibility contract is
NOT the package version — it is

1. the **pinned index release** (`models-index.yaml` at tag index-vN,
   printed by each release's notes and asserted by an exact-pin test in
   every runtime), and
2. the **golden corpus** (cross-runtime byte-parity vectors; a runtime
   whose output diverges does not release).

Package versions move on their own registries' conventions (the npm
line is long and mature; the gem entered its 3.x unified generation;
PyPI debuts after the trusted-publisher fix). Users who need a specific
behavior pin the index (`SECRYST_INDEX`/`INTERSCRIPT_ML_INDEX` env var
or the explicit `index_url`), not the package version.

Rule of thumb for releases: an index bump (new models, URL changes) is
a patch/minor per runtime; a dispatch-surface or contract change is a
minor per runtime; only user-facing breaking API changes take majors.
