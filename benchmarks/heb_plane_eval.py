"""WO04: artifact-level runtime eval for a Hebrew plane zip.

Protocol = run-021's: hebrew-v4 test, 1400-byte windows, K-pass greedy
via the py PlaneModel runtime, original-separator stitching, seq2seq_der.
Requires: PYTHONPATH pointing at interscript-py/src (runtime) and
interscript-train (nikud_planes + src/rababa for the metric).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

ZIP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/run022/heb-diac-plane-2.0.zip")
TEST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/heb-v4-test.jsonl")
UNIT_BYTES = 1400


def split_windows(text: str, budget: int = UNIT_BYTES) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words, cur, n, wins = text.split(), [], 0, []
    for w in words:
        c = len(w.encode("utf-8")) + 1
        if cur and n + c > budget:
            wins.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += c
    if cur:
        wins.append(" ".join(cur))
    return wins


def main() -> None:
    from interscript.ml.plane import PlaneModel
    from rababa.evaluate import seq2seq_der
    import nikud_planes as NP

    data = ZIP.read_bytes()
    print(f"zip={ZIP.name} sha256={hashlib.sha256(data).hexdigest()[:12]}", flush=True)
    model = PlaneModel.from_zip(data)

    rows = [json.loads(l) for l in TEST.read_text().splitlines() if l.strip()]
    targets = [r["tgt"].strip() for r in rows]
    # decode the SKELETON (the runtime contract): strip nikud first —
    # passing diacritized text treats marks as base chars (the 55% trap)
    skeletons = [NP.split_planes(t)[0] for t in targets]
    all_windows, counts = [], []
    for skel in skeletons:
        ws = split_windows(skel)
        counts.append(len(ws))
        all_windows.extend(ws)
    print(f"examples={len(targets)} windows={len(all_windows)}", flush=True)

    preds_w = []
    t0 = time.time()
    for i, w in enumerate(all_windows, 1):
        preds_w.append(model.translate(w))
        if i % 200 == 0:
            r = i / (time.time() - t0)
            print(f"[gen] {i}/{len(all_windows)} ({r:.2f} win/s)", flush=True)

    k = 0
    wrong = 0.0
    total = 0
    for tgt, c in zip(targets, counts):
        text = tgt
        words = text.split()
        seps = re.findall(r"\s+", text)
        sep_for = {i: (seps[i] if i < len(seps) else "") for i in range(len(words))}
        rebuilt = []
        for _ in range(c):
            pred = preds_w[k]; k += 1
            for w in pred.split():
                wi = sum(len(part.split()) for part in rebuilt)
                rebuilt.append(w + sep_for.get(wi, " "))
        pred = "".join(rebuilt)
        d, n = seq2seq_der(pred, tgt)
        wrong += d * n
        total += n
    der = wrong / max(1, total)
    print(json.dumps({"zip": ZIP.name, "der": round(der, 4),
                      "positions": total, "wall_s": round(time.time() - t0, 1)}),
          flush=True)


if __name__ == "__main__":
    main()
