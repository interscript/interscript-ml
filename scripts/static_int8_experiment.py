"""Static-int8 decoder experiment (TODO.impl/11, from the framing finding).

The shipped int8 graphs use quantize_dynamic: DynamicQuantizeLinear
computes ACTIVATION scales per fed tensor at runtime, so decode framing
(single-step vs batched feeds) changes the numerics materially
(RESULTS.md 2026-09-12). This experiment re-quantizes the decoder with
STATIC activation scales (quantize_static, calibrated on real decode
feeds in both framings) and measures:

  A. framing equality - the same greedy decode driven token-by-token
     vs driven in 8-token batched calls: identical trajectories?
  B. drift vs the fp32 decoder's greedy (quality proxy; full-set DER is
     the gate if framing clears)
  C. speed - KV-greedy tokens/sec vs the shipped dynamic int8

Usage: python3 scripts/static_int8_experiment.py <fp32.zip> <workdir>
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "src"))
from interscript_ml.tokens import EOS_ID, PAD_ID, encode  # noqa: E402

GOLDEN = Path.home() / "ml-logs" / "golden" / "ara-diac-small-2.1-int8.jsonl"
DYNAMIC_INT8_ZIP = (
    Path.home() / ".cache" / "secryst" / "models"
    / "ara-diac-small-2.1-int8" / "ara-diac-small-2.1-int8.zip"
)
STEPS = 96


def session(path: Path | str | bytes) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        path if isinstance(path, (bytes, bytearray)) else str(path), opts
    )


class Decoder:
    """The decoder graph driven through one door; `pasts` are carried
    externally so any framing can be expressed on top."""

    def __init__(self, sess: ort.InferenceSession):
        self.sess = sess
        self.past_names = [i.name for i in sess.get_inputs() if i.name.startswith("past_")]
        self.present_names = [o.name for o in sess.get_outputs() if o.name.startswith("present_")]

    def run(self, tokens: list[int], hidden, pasts: dict[str, np.ndarray] | None):
        meta = {i.name: i for i in self.sess.get_inputs()}
        feed = {
            "input_ids": np.array([tokens], dtype=np.int64),
            "encoder_hidden_states": hidden,
        }
        for name in self.past_names:
            feed[name] = (
                pasts[name]
                if pasts and name in pasts
                else np.zeros(
                    (1, meta[name].shape[1], 0, meta[name].shape[3]), dtype=np.float32
                )
            )
        out = self.sess.run(None, feed)
        names = [o.name for o in self.sess.get_outputs()]
        return dict(zip(names, out, strict=True))

    def split(self, out) -> tuple[int, dict[str, np.ndarray]]:
        argmax = int(np.argmax(out["logits"][0, -1]))
        pasts = {
            n.replace("present_", "past_"): out[n]
            for n in self.present_names
        }
        return argmax, pasts

    @staticmethod
    def trim(pasts: dict[str, np.ndarray], seq_len: int) -> dict[str, np.ndarray]:
        """Keep the first `seq_len` positions of each KV tensor."""
        return {k: v[:, :, :seq_len, :].copy() for k, v in pasts.items()}


def greedy(dec: Decoder, hidden, batch: int) -> list[int]:
    """Greedy decode driven with `batch`-token feeds: position i's
    prediction comes from a call fed the `batch` tokens ending at i.
    batch=1 is the classic incremental loop; batch=8 is the speculative
    verification framing."""
    argmax, pasts = dec.split(dec.run([PAD_ID], hidden, None))
    traj = [argmax]
    while len(traj) < STEPS and traj[-1] != EOS_ID:
        window = traj[-batch:]
        out = dec.run(window, hidden, Decoder.trim(pasts, len(pasts[next(iter(pasts))]) if pasts else 0))
        argmax, _ = dec.split(out)
        # rebuild pasts through honest incremental steps (the cache must
        # reflect every consumed token, mirroring a real runtime)
        for tok in window:
            _, pasts = dec.split(dec.run([tok], hidden, pasts))
        traj.append(argmax)
    return traj


def encode_hidden(enc: ort.InferenceSession, text: str):
    ids = encode(text)
    return enc.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]


def calibration_samples(fp32: Decoder, enc, rows) -> list[dict]:
    """Decoder feeds across framings: prefills, incremental steps, and
    batched windows (the shapes whose scales must hold)."""
    samples = []

    def record(tokens, hidden, pasts):
        feed = {"input_ids": np.array([tokens], dtype=np.int64),
                "encoder_hidden_states": hidden}
        for name in fp32.past_names:
            feed[name] = (
                pasts[name]
                if pasts and name in pasts
                else np.zeros((1, feed_shapes[name][0], 0, feed_shapes[name][1]), dtype=np.float32)
            )
        samples.append(feed)

    feed_shapes = {
        name: (i.shape[1], i.shape[3])
        for i in fp32.sess.get_inputs()
        for name in [i.name]
        if name.startswith("past_")
    }

    for text in rows:
        hidden = encode_hidden(enc, text)
        argmax, pasts = fp32.split(fp32.run([PAD_ID], hidden, None))
        record([PAD_ID], hidden, None)
        window = []
        for _ in range(64):
            window.append(argmax)
            record([argmax], hidden, pasts)
            argmax, pasts = fp32.split(fp32.run([argmax], hidden, pasts))
            if argmax == EOS_ID:
                break
            if len(window) == 8:
                record(window, hidden, Decoder.trim(pasts, max(pasts[next(iter(pasts))].shape[2] - len(window), 0)))
                window = []
    return samples


def main() -> None:
    fp32_zip, workdir = Path(sys.argv[1]), Path(sys.argv[2])
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fp32_zip) as zf:
        for name in zf.namelist():
            if name.endswith(".onnx"):
                (workdir / name).write_bytes(zf.read(name))
    enc = session(workdir / "encoder.onnx")
    fp32 = Decoder(session(workdir / "decoder-kv.onnx"))
    rows = [json.loads(l)["input"] for l in GOLDEN.read_text().splitlines() if l.strip()]

    print("calibration feeds...", flush=True)
    samples = calibration_samples(fp32, enc, rows[:5])
    print(f"  {len(samples)} samples", flush=True)

    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from imf.export import head_matmul_names

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data

        def get_next(self):
            return self.data.pop(0) if self.data else None

        def rewind(self):
            pass  # one pass; the sample set is the calibration corpus

    static_path = workdir / "decoder-kv-static.onnx"
    print("quantize_static...", flush=True)
    quantize_static(
        str(workdir / "decoder-kv.onnx"),
        str(static_path),
        calibration_data_reader=Reader(samples),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=head_matmul_names(workdir / "decoder-kv.onnx"),
    )
    static = Decoder(session(static_path))

    test_rows = rows[:5]
    for name, dec in (("fp32", fp32), ("static-int8", static)):
        same = total = 0
        drift = 0
        for text in test_rows:
            hidden = encode_hidden(enc, text)
            single = greedy(dec, hidden, 1)
            batched = greedy(dec, hidden, 8)
            for a, b in zip(single, batched):
                total += 1
                same += a == b
            ref = greedy(fp32, hidden, 1) if name != "fp32" else single
            drift += sum(1 for a, b in zip(single, ref) if a != b)
        print(f"{name}: framing single==batched {same}/{total}; drift-vs-fp32 {drift}/{total}")

    # speed vs the shipped dynamic int8
    with zipfile.ZipFile(DYNAMIC_INT8_ZIP) as zf:
        dyn_bytes = zf.read("decoder-kv.onnx")
    dynamic = Decoder(session(dyn_bytes))
    for name, dec in (("dynamic-int8", dynamic), ("static-int8", static)):
        t0 = time.time()
        toks = 0
        for text in test_rows:
            hidden = encode_hidden(enc, text)
            argmax, pasts = dec.split(dec.run([PAD_ID], hidden, None))
            while toks < 10_000 and argmax != EOS_ID:
                toks += 1
                argmax, pasts = dec.split(dec.run([argmax], hidden, pasts))
        dt = time.time() - t0
        print(f"{name}: {toks} tokens in {dt:.1f}s = {toks / dt:.0f} tok/s")


if __name__ == "__main__":
    main()
