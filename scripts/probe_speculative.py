"""Speculative-decoding acceptance probe (TODO.qwen-next/10 §1).

Measures whether ara-diac-layerdrop-1.0-int4 (drafter) can draft for
ara-diac-small-2.1-int8 (verifier) on the golden-v1 Arabic rows: block
acceptance rate, tokens per verifier pass, and exactness of the
speculative loop against verifier-only greedy (must be identical —
greedy verification is output-preserving by construction, so a
mismatch means a probe bug, not a model property).

CPU-only, no training. Results land beside the log in ~/ml-logs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "src"))
from interscript_ml.model import Model  # noqa: E402
from interscript_ml.tokens import EOS_ID, PAD_ID, encode  # noqa: E402

DRAFTER = "ara-diac-layerdrop-1.0-int4"
VERIFIER = "ara-diac-small-2.1-int8"
GOLDEN = Path.home() / "ml-logs" / "golden" / f"{DRAFTER}.jsonl"
OUT = Path.home() / "ml-logs" / "spec_probe"
K = 8
# The plain-path exactness reference is O(T^2); cap it to short rows so
# the probe finishes. Longer rows still report acceptance statistics.
EXACT_LIMIT_BYTES = 600


def plain_logits(model: Model, hidden, feed: list[int]) -> np.ndarray:
    """Full-sequence run with zero pasts; works for both kv and plain
    decoder graphs. Returns logits [L, V] (batch dropped)."""
    outputs = model._decoder.run(
        None,
        {
            "input_ids": np.array([feed], dtype=np.int64),
            "encoder_hidden_states": hidden,
            **model._pasts,
        },
    )
    return outputs[0][0]


def plain_greedy(model: Model, hidden, max_len: int) -> list[int]:
    """Full-sequence greedy over the same plain path the verifier
    decisions use (zero pasts; works for kv and plain graphs). The
    exactness theorem compares against THIS, not the KV path — int8
    near-ties can flip between execution paths."""
    feed = [PAD_ID]
    out: list[int] = []
    for _ in range(max_len):
        logits = plain_logits(model, hidden, feed)
        token = int(np.argmax(logits[-1]))
        if token == EOS_ID:
            break
        out.append(token)
        feed.append(token)
    return out


def draft(model: Model, hidden, seq: list[int], k: int, max_len: int) -> list[int]:
    """Greedily draft k tokens conditioned on seq (the verifier-
    authoritative prefix)."""
    if len(seq) >= max_len:
        return []
    feed = [PAD_ID] + seq
    if model._kv_session:
        pasts = dict(model._pasts)
        draft_out: list[int] = []
        current = np.array([feed], dtype=np.int64)
        while len(draft_out) < k and len(seq) + len(draft_out) < max_len:
            outputs = model._decoder.run(
                None,
                {"input_ids": current, "encoder_hidden_states": hidden, **pasts},
            )
            results = dict(zip(model._output_names, outputs, strict=True))
            token = int(np.argmax(results["logits"][0, -1]))
            if token == EOS_ID:
                draft_out.append(token)
                break
            draft_out.append(token)
            pasts = {
                name: results[name.replace("past_", "present_", 1)] for name in pasts
            }
            current = np.array([[token]], dtype=np.int64)
        return draft_out
    # plain graph: stepwise full re-run
    draft_out = []
    while len(draft_out) < k and len(seq) + len(draft_out) < max_len:
        logits = plain_logits(model, hidden, feed + draft_out)
        token = int(np.argmax(logits[-1]))
        if token == EOS_ID:
            draft_out.append(token)
            break
        draft_out.append(token)
    return draft_out


def spec_decode(drafter: Model, verifier: Model, hidden_d, hidden_v, max_len: int):
    seq: list[int] = []
    drafted = 0
    accepted = 0  # draft tokens the verifier kept (bonus excluded)
    bonus = 0
    blocks = 0
    while len(seq) < max_len:
        block = draft(drafter, hidden_d, seq, K, max_len)
        if not block:
            break
        blocks += 1
        drafted += len(block)
        feed = [PAD_ID] + seq + block
        logits = plain_logits(verifier, hidden_v, feed)
        n_acc = 0
        correction = None
        for i, d in enumerate(block):
            v = int(np.argmax(logits[len(seq) + i]))
            if v == d:
                n_acc += 1
            else:
                correction = v
                break
        if correction is not None:
            seq.extend(block[:n_acc])
            accepted += n_acc
            if correction == EOS_ID:
                return seq, dict(
                    drafted=drafted, accepted=accepted, bonus=bonus, blocks=blocks
                )
            seq.append(correction)
            continue
        accepted += n_acc
        seq.extend(block[:-1] if block[-1] == EOS_ID else block)
        if block[-1] == EOS_ID:
            return seq, dict(
                drafted=drafted, accepted=accepted, bonus=bonus, blocks=blocks
            )
        # seq already contains the block; the last logits position has
        # seen all of it and predicts the bonus token
        next_tok = int(np.argmax(logits[len(seq)]))
        bonus += 1
        if next_tok == EOS_ID:
            return seq, dict(
                drafted=drafted, accepted=accepted, bonus=bonus, blocks=blocks
            )
        seq.append(next_tok)
    return seq, dict(drafted=drafted, accepted=accepted, bonus=bonus, blocks=blocks)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    drafter = Model.load(DRAFTER)
    verifier = Model.load(VERIFIER)
    rows = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for idx, row in enumerate(rows):
        text = row["input"]
        max_len = max(256, 4 * len(text.encode("utf-8")))
        ids_d = np.array([encode(text)], dtype=np.int64)
        ids_v = np.array([encode(text)], dtype=np.int64)
        hidden_d = drafter._encoder.run(None, {"input_ids": ids_d})[0]
        hidden_v = verifier._encoder.run(None, {"input_ids": ids_v})[0]
        t0 = time.time()
        seq, stats = spec_decode(drafter, verifier, hidden_d, hidden_v, max_len)
        dt = time.time() - t0
        # exactness vs the SAME execution path the verifier decisions
        # came from: plain full-sequence greedy. (KV-greedy can differ
        # from plain at int8 near-ties — that is the quantized-parity
        # phenomenon, not a probe bug; reported separately.)
        in_bytes = len(text.encode("utf-8"))
        if in_bytes <= EXACT_LIMIT_BYTES:
            ref_plain = plain_greedy(verifier, hidden_v, max_len)
            exact: bool | None = seq == ref_plain
            kv_match = verifier.generate(text, max_len=max_len) == ref_plain
        else:
            exact = None
            kv_match = None
        progress = len(seq)
        results.append(
            {
                "row": idx,
                "in_bytes": in_bytes,
                "out_tokens": progress,
                "exact": exact,
                "kv_match": kv_match,
                "acceptance": stats["accepted"] / max(stats["drafted"], 1),
                "tokens_per_verify": progress / max(stats["blocks"], 1),
                "blocks": stats["blocks"],
                "bonus": stats["bonus"],
                "sec": round(dt, 2),
            }
        )
        print(
            f"row {idx:2d} in={results[-1]['in_bytes']:5d}B out={progress:4d} "
            f"acc={results[-1]['acceptance']:.3f} tok/verify="
            f"{results[-1]['tokens_per_verify']:.2f} exact={exact} "
            f"kv_match={kv_match} {dt:.1f}s",
            flush=True,
        )
    tot_out = sum(r["out_tokens"] for r in results)
    tot_blocks = sum(r["blocks"] for r in results)
    exact_rows = [r for r in results if r["exact"] is not None]
    summary = {
        "drafter": DRAFTER,
        "verifier": VERIFIER,
        "k": K,
        "rows": len(results),
        "exact_checked_rows": len(exact_rows),
        "all_exact": all(r["exact"] for r in exact_rows),
        "plain_kv_identical": all(r["kv_match"] for r in exact_rows),
        "mean_acceptance": sum(r["acceptance"] for r in results) / len(results),
        "tokens_per_verify_overall": tot_out / max(tot_blocks, 1),
        "results": results,
    }
    (OUT / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
