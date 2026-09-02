# 03-ara-diac-small-21

Export/publish G2a best (4.5701) as ara-diac-small-2.1: parity, release, models.yaml, index-v3, Modal serving. [models+project]

Status: pending (2026-09-02).

Progress log (2026-09-03): zips exported (fp32/fp16/int8 on
imf/ara-diac-small-21/). Parity: first pass hit the 18,000s task
timeout mid reference-decode; relaunched under the retry watchdog
(resumable reference). Next: parity pass -> publish_model.py -> GH
release + models.yaml -> index-v3 -> Modal redeploy + API regen.
