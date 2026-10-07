"""WO03: Thai hybrid bench — dict-only vs model-only vs hybrid.

Protocol: kaikki test (1,219 sentences), corpus-level character PER
(true Levenshtein over the concatenated corpus), greedy, int8 runtime,
CPU latency wall-clock per utterance.

Usage: PYTHONPATH=<interscript-py>/src python3 thai_hybrid_bench.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TEST = Path.home() / "src/interscript/ml-qwen-feat/benchmarks/thai-kaikki-g2p/test.jsonl"
ART = Path("/tmp/tha-g2p-small-1.0-int8.zip")
LEX = Path("/tmp/tha-lexicon.json")


def levenshtein(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 4000:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def corpus_per(pairs: list[tuple[str, str]]) -> float:
    dist = sum(levenshtein(p, t) for p, t in pairs)
    total = sum(len(t) for _, t in pairs)
    return 100.0 * dist / total


def main() -> None:
    from interscript.ml.model import Model
    from interscript.ml.thai_hybrid import hybrid_translate, segment

    lex = json.loads(LEX.read_text())
    model = Model.load(str(ART))

    rows = [json.loads(l) for l in TEST.read_text().splitlines() if l.strip()]
    print(f"sentences={len(rows)} lexicon={len(lex)}", flush=True)

    def dict_only(text: str) -> str:
        return "".join(ipa if ipa is not None else chunk
                       for chunk, ipa in segment(text, lex))

    def hybrid(text: str) -> str:
        return hybrid_translate(text, lex, model)

    for name, fn in (("dict-only", dict_only),
                     ("hybrid", hybrid),
                     ("model-only", lambda t: model.translate(t))):
        t0 = time.time()
        pairs = [(fn(r["src"]), r["tgt"]) for r in rows]
        wall = time.time() - t0
        per = corpus_per(pairs)
        print(json.dumps({"mode": name, "per": round(per, 4),
                          "ms_per_utt": round(1000 * wall / len(rows), 3)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
