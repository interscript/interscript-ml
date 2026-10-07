"""WO05: phonikud/heb-g2p-benchmark — chain our plane + nikud_ipa rules.

Chain: undiacritized sentence -> heb plane artifact (nikud) -> nikud_ipa
rules -> convention adapter (their gold uses chi for het, ts for tsadi,
e for sheva) -> WER/CER vs gt.tsv with jiwer-equivalent inline scoring.
No stress is produced (documented limitation); stress_wer not comparable.

Usage: PYTHONPATH=<interscript-py>/src:<interscript-train>/src
       python3 heb_g2p_bench.py <plane.zip>
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

GT = Path(__file__).parent / "data/heb-g2p/gt.tsv"
GT_SHA256 = "b7cd3db93cbad54d915f5423ecc1ff504e0e69b82b2493c7918c72b228ea2f3d"


def adapt(ipa: str) -> str:
    return ipa.replace("x", "χ").replace("t͡s", "ts").replace("ə", "e")


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    from interscript.ml.nikud_ipa import nikud_to_ipa, plane_to_ipa
    from interscript.ml.plane import PlaneModel

    got = hashlib.sha256(GT.read_bytes()).hexdigest()
    assert got == GT_SHA256, f"gt.tsv sha mismatch: {got}"
    gt = []
    for line in GT.read_text().splitlines():
        s, _, p = line.partition("\t")
        if s and p:
            gt.append((s, p))
    print(f"gt sentences={len(gt)}", flush=True)

    data = Path(sys.argv[1]).read_bytes()
    print(f"zip sha256={hashlib.sha256(data).hexdigest()[:12]}", flush=True)
    model = PlaneModel.from_zip(data)

    preds = []
    t0 = time.time()
    for i, (sent, _) in enumerate(gt, 1):
        nikud_text = model.translate(sent)
        preds.append(adapt(nikud_to_ipa(nikud_text)))
        if i % 50 == 0:
            print(f"[gen] {i}/{len(gt)} ({i / (time.time() - t0):.2f} sent/s)", flush=True)

    wers, cers, cers_ns = [], [], []
    for (sent, gold), pred in zip(gt, preds):
        gold_ns = gold.replace("ˈ", "")  # stress-insensitive CER
        pred_ns = pred.replace("ˈ", "")
        gw, pw = gold.split(), pred.split()
        prev = list(range(len(pw) + 1))
        for i, ga in enumerate(gw, 1):
            cur = [i]
            for j, pb in enumerate(pw, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ga != pb)))
            prev = cur
        wers.append(prev[-1] / max(1, len(gw)))
        cers.append(lev(pred, gold) / max(1, len(gold)))
        cers_ns.append(lev(pred_ns, gold_ns) / max(1, len(gold_ns)))
    out = {"wer": round(sum(wers) / len(wers), 4),
           "cer": round(sum(cers) / len(cers), 4),
           "cer_no_stress": round(sum(cers_ns) / len(cers_ns), 4),
           "n": len(gt),
           "note": "no stress produced; WER=1.0 is structural (every gold word carries stress)"}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    for (sent, gold), pred in list(zip(gt, preds))[:3]:
        print(f"  src: {sent[:40]}\n  pred: {pred[:60]}\n  gold: {gold[:60]}", flush=True)


if __name__ == "__main__":
    main()
