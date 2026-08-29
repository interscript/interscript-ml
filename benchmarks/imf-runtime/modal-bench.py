"""E2: server-tier IMF benchmark on the production inference shape
(Modal, 4 vCPU / 8 GiB, models pre-staged on the secryst-models
volume). Measures cold load (zip + member verify + ORT init) and
decode latency by input length.

    modal run --detach benchmarks/imf-runtime/modal-bench.py --model-id tha-g2p-small-1.0
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("onnxruntime==1.23.2", "pyyaml>=6.0", "numpy>=1.26")
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent.parent),
        "/root/interscript-ml",
        copy=True,
    )
    .workdir("/root/interscript-ml")
)
VOL = modal.Volume.from_name("secryst-models")

SAMPLES = {
    "ara-diac-small-1.0-int8": [
        "كتاب",
        "السلام عليكم ورحمة الله وبركاته",
        "السلام عليكم ورحمة الله وبركاته" * 8,
    ],
    "default": ["สวัสดี", "สวัสดีครับกรุงเทพมหานคร", "สวัสดีครับกรุงเทพมหานคร" * 8],
}

app = modal.App("imf-runtime-bench", image=image)


@app.function(cpu=4, memory=8 * 1024, timeout=30 * 60, volumes={"/v": VOL})
def bench(model_id: str, filename: str) -> dict:
    import glob
    import sys

    sys.path.insert(0, "/root/interscript-ml/src")
    from imf.export import onnx_greedy_kv
    from imf.parity import _sessions_from_zip  # parity-verified loader

    matches = [z for z in glob.glob(f"/v/imf/*/{filename}") if z.endswith(".zip")]
    path = Path(matches[0])
    t0 = time.perf_counter()
    enc, kv = _sessions_from_zip(path)
    cold_s = time.perf_counter() - t0

    out = {"model": model_id, "zip_mb": round(path.stat().st_size / 1e6),
           "cold_load_s": round(cold_s, 2), "decodes_ms": []}
    for text in SAMPLES.get(model_id, SAMPLES["default"]):
        t0 = time.perf_counter()
        onnx_greedy_kv(enc, kv, text, max_len=3 * len(text.encode()) + 256)
        out["decodes_ms"].append(round((time.perf_counter() - t0) * 1000))
    return out


@app.local_entrypoint()
def main(
    model_id: str = "tha-g2p-small-1.0",
    filename: str = "",
) -> None:
    if not filename:
        import yaml

        index = yaml.safe_load(open("models.yaml", encoding="utf-8"))
        filename = index["models"][model_id]["filename"]
    print(bench.remote(model_id, filename))
