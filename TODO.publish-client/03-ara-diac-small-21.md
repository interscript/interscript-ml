# 03-ara-diac-small-21

Export/publish G2a best (4.5701) as ara-diac-small-2.1: parity, release, models.yaml, index-v3, Modal serving. [models+project]

Status: in flight (2026-09-03). fp32/fp16 parity passed (delta 0.0714pp, margin flips 0.0000%); int8 stage running after the stage-skip fix (PRs #152/#153). Publish chain follows: publish_model.py -> models.yaml -> index-v3 -> Modal deploy -> API regen.
