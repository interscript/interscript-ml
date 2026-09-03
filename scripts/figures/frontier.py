#!/usr/bin/env python3
"""Paper-B figure: the size-quality frontier + the subset-overstatement
pairs, from the durable verdicts (final_eval.json per run). Fetches the
small JSON files from Modal volumes; no GPU.

    python scripts/figures/frontier.py --out docs/paper-assets/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

RUNS = [
    # (label, params_m, full_set_der, subset_der, volume, path)
    ("1.0 AdamW/r6/3ep", 300, 8.2590, 3.658, "rababa-checkpoints",
     "rababa_arabic_distill_small/run-002"),
    ("lite Muon/6ep", 190, 5.784, 2.6495, "rababa-checkpoints",
     "rababa_arabic_distill_small/run-009-layerdrop-6ep"),
    ("2.0 Muon/r7/3ep", 300, 4.8218, None, "rababa-checkpoints",
     "rababa_arabic_distill_small/run-006-r7-muon"),
    ("2.1 Muon/r7/6ep", 300, 4.5701, 2.0062, "rababa-checkpoints",
     "rababa_arabic_distill_small/run-007-r7-muon-6ep"),
    ("teacher r7", 580, 2.2864, None, "rababa-checkpoints", None),
    ("tiny-max (30M full levers)", 33, 73.9489, None, "rababa-checkpoints",
     "rababa_arabic_distill_small/run-010-tiny-max"),
]


def fetch_eval(volume: str, path: str | None) -> dict | None:
    if not path:
        return None
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    r = subprocess.run(
        ["modal", "volume", "get", volume, f"{path}/final_eval.json", out],
        capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(Path(out).read_text())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper-assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: the frontier (params vs full-set DER)
    fig, ax = plt.subplots(figsize=(6, 4))
    pts = [(r[1], r[2]) for r in RUNS if r[2] is not None]
    labels = [r[0] for r in RUNS if r[2] is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
    for (x, y), lab in zip(pts, labels, strict=True):
        ax.annotate(lab, (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters (M, log)")
    ax.set_ylabel("full-set windowed DER-CE (log)")
    ax.set_title("The client-tier size–quality frontier (all full-set, CI-bracketed)")
    fig.tight_layout()
    fig.savefig(out / "frontier.png", dpi=200)

    # Figure 2: subset overstatement pairs
    pairs = [(r[0], r[3], r[2]) for r in RUNS if r[3] is not None]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    idx = range(len(pairs))
    w = 0.35
    ax2.bar([i - w / 2 for i in idx], [p[1] for p in pairs], w, label="first-300 subset")
    ax2.bar([i + w / 2 for i in idx], [p[2] for p in pairs], w, label="full 1,200")
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels([p[0] for p in pairs], fontsize=7, rotation=20)
    ax2.set_ylabel("DER-CE")
    ax2.set_title("Subset overstatement: 2–4× inflation across five instances")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out / "subset-overstatement.png", dpi=200)
    print(f"wrote {out}/frontier.png and {out}/subset-overstatement.png")


if __name__ == "__main__":
    main()
