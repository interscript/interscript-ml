"""Modal app: score a SHIPPED IMF artifact end-to-end through the
Python runtime — the protocol's 1400-byte windowing, greedy ONNX
decode, haraqat projection, sadeedbench DER — so a published number
can be re-derived from the exact bytes users download.

    modal run --detach src/gpu/modal_score_artifact.py::score \\
        --zip imf/ara-diac-small-2/ara-diac-small-2.0-fp32.zip \\
        --expect-sha d9aa95d0... --out artifact-2.0-preds.jsonl
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime==1.23.2",
        "numpy>=1.26",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "pyarabic>=0.6",
        "prettytable>=3.9",
        "pyyaml>=6.0",
    )
    .add_local_dir(str(REPO_ROOT), "/root/interscript-ml", copy=True)
    .add_local_dir(str(REPO_ROOT.parent / "rababa"), "/opt/rababa", copy=True)
    .workdir("/root/interscript-ml")
)

MODELS_VOLUME = modal.Volume.from_name("secryst-models")

app = modal.App("interscript-ml-artifact-score", image=IMAGE)


@app.function(cpu=8, memory=32 * 1024, timeout=5 * 3600,
              volumes={"/outputs": MODELS_VOLUME})
def score(zip_path: str, expect_sha: str = "", out: str = "") -> dict:
    import hashlib
    import json
    import sys

    sys.path.insert(0, "/root/interscript-ml/runtime/src")
    sys.path.insert(0, "/root/interscript-ml/src")

    volume_zip = Path("/outputs") / zip_path
    digest = hashlib.sha256(volume_zip.read_bytes()).hexdigest()
    if expect_sha and digest != expect_sha:
        raise RuntimeError(f"volume zip sha {digest} != expected {expect_sha}")

    import pandas as pd

    from harness.sadeed import project_haraqat, split_windows, strip_diacritics
    from interscript_ml import Model
    from sadeedbench.scoring import score_predictions

    inputs = pd.read_parquet("/opt/rababa/data/sadeed-diac-25/train.parquet")[
        "input"
    ].tolist()
    model = Model.load(volume_zip)

    # resumable: worker evictions restart the input; completed rows are
    # durably on the volume and skipped on relaunch
    out_path = Path("/outputs") / out if out else None
    done: dict[int, str] = {}
    if out_path and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["idx"]] = row["student"]
        print(f"resume: {len(done)} rows already complete", flush=True)

    preds: list[str] = []
    for i, src in enumerate(inputs):
        if i in done:
            preds.append(done[i])
            continue
        stripped = strip_diacritics(src)
        pieces = []
        for w in split_windows(stripped, 1400):
            out_text = model.translate(w, max_len=max(256, 2 * len(w)))
            pieces.append(project_haraqat(out_text, w))
        preds.append("".join(pieces))
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(inputs)}", flush=True)
            if out_path:
                with out_path.open("w", encoding="utf-8") as fh:
                    for j, q in enumerate(preds):
                        if j < len(preds):
                            fh.write(json.dumps({"idx": j, "student": q}, ensure_ascii=False) + "\n")
                MODELS_VOLUME.commit()

    gts = pd.read_parquet("/opt/rababa/data/sadeed-diac-25/train.parquet")[
        "output"
    ].tolist()
    result = score_predictions(preds, gts)
    result["sha256"] = digest
    return result


@app.local_entrypoint()
def main(zip_path: str, expect_sha: str = "", out: str = "") -> None:
    print(score.remote(zip_path, expect_sha, out))
