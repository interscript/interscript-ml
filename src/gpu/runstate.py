"""Run-state markers — the durable protocol every arm, orchestrator, and
supervisor agrees on.

The contract (learned from incidents, now an interface):
- training is done when ``best/config.json`` exists (save_pretrained)
- evaluation is done when ``final_eval.json`` exists (evaluate_der)
- progress is observable as ``step-N`` checkpoint dirs; a stall is no
  new step for STALL_SECS while training is incomplete
- ``chain_log.jsonl`` is the append-only audit trail

The mkdir bug that killed the orchestrator's arm 4 (log() opened a file
inside a run dir only training creates) is unrepresentable through this
interface: every writer mkdirs.
"""

from __future__ import annotations

from pathlib import Path

TRAINING_DONE = "best/config.json"
EVAL_DONE = "final_eval.json"
CHAIN_LOG = "chain_log.jsonl"


class RunState:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def training_done(self) -> bool:
        return (self.root / TRAINING_DONE).exists()

    def eval_done(self) -> bool:
        return (self.root / EVAL_DONE).exists()

    def latest_step(self) -> int:
        steps = [int(p.name.split("-")[1]) for p in self.root.glob("step-*")]
        return max(steps) if steps else -1

    def log(self, event: str, commit=None) -> None:
        import json
        import time

        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / CHAIN_LOG).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time()), "event": event}) + "\n")
        if commit is not None:
            commit()

    def read_eval(self) -> dict | None:
        import json

        path = self.root / EVAL_DONE
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
