# WO30 — Hebrew learned-IPA v0 (first trained Hebrew G2P we own)

RUNNING (run-032, l4x1): FLEURS he_il (~5K utterances, CC-BY-4.0)
phoneme-labeled by the universal CTC (the stage-2-GO supervision) →
byt5-small seq2seq text→IPA student, small-data regime.

- Gate: phonikud heb-g2p-benchmark CER < our rules layer (0.2393).
- Stretch: Nakdimon-class (0.1102).
- If v0 clears the gate: scale decision = ivrit.ai (~1,700h at
  $0.28/audio-hour ≈ $500) for the ReNikud-class run.
- NO LLM labels anywhere in the chain.
