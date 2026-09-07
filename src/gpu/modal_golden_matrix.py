"""Modal app: generate per-model golden sets from the RELEASED zips —
the cross-runtime byte-parity corpus. For every models.yaml entry,
decode N test inputs through the Python runtime using the exact
release artifact (sha-pinned to the index) and write
golden/<model-id>.jsonl rows {input, output}. Published as a release
asset so the TS and Ruby CI jobs verify against the same bytes.

    modal run --detach src/gpu/modal_golden_matrix.py --n 25
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
        "pyyaml>=6.0",
    )
    .add_local_dir(str(REPO_ROOT), "/root/interscript-ml", copy=True)
    .add_local_dir(str(REPO_ROOT.parent / "rababa"), "/opt/rababa", copy=True)
    .workdir("/root/interscript-ml")
)

MODELS_VOLUME = modal.Volume.from_name("secryst-models")
DATASETS = modal.Volume.from_name("rababa-datasets")
SECRYST_DATASETS = modal.Volume.from_name("secryst-datasets")
URDU_G2P = modal.Volume.from_name("urdu-g2p-datasets")
URDU_DIAC = modal.Volume.from_name("urdu-diacrit-datasets")
PERSIAN = modal.Volume.from_name("persian-g2p-datasets")

app = modal.App("interscript-ml-golden-matrix", image=IMAGE)

# index id -> volume zip path + test pairs (volume-relative)
SOURCES = {
    "khm-latn-1.0": ("imf/khm-latn/khm-latn-1.0-fp32.zip", "/secryst-datasets/khmer-translit/test.jsonl"),
    "urd-g2p-1.0": ("imf/urd-g2p/urd-g2p-1.0-fp32.zip", "/ud-g2p/urdu-g2p/test.jsonl"),
    "urd-diac-1.0": ("imf/urd-diac/urd-diac-1.0-fp32.zip", "/ud-diacrit/urdu-diacrit/test.jsonl"),
    "tha-g2p-base-1.0": ("imf/tha-g2p-base/tha-g2p-base-1.0-fp32.zip", "/secryst-datasets/thai-ipa/test.jsonl"),
    "tha-g2p-small-1.0": ("imf/tha-g2p-small/tha-g2p-small-1.0-int8.zip", "/secryst-datasets/thai-ipa/test.jsonl"),
    "fas-g2p-1.0": ("imf/fas-g2p/fas-g2p-1.0-fp32.zip", "/persian/persian-g2p/test.jsonl"),
    "heb-diac-1.0": ("imf/heb-diac/heb-diac-1.0-fp32.zip", "nakdimon/test-imf.jsonl"),
    "heb-diac-1.1": ("imf/heb-diac/heb.zip", "nakdimon/test-imf.jsonl"),
    "heb-diac-small-1.0": ("imf/heb-diac-small/heb-diac-small-1.0-fp32.zip", "nakdimon/test-imf.jsonl"),
    "ara-diac-1.0": ("imf/ara-diac/ara-diac-1.0-fp32.zip", "arabic-sadeed-imf/test.jsonl"),
    "ara-diac-2.0-int8": ("imf/ara-diac2/ara-diac-2.0-int8.zip", "arabic-sadeed-imf/test.jsonl"),
    "ara-diac-small-2.1-int8": ("imf/ara-diac-small-21/ara-diac-small-2.1-int8.zip", "arabic-sadeed-imf/test.jsonl"),
    "ara-diac-layerdrop-1.0-int4": ("imf/ara-diac-layerdrop/ara-diac-layerdrop-1.0-int4.zip", "arabic-sadeed-imf/test.jsonl"),
}


@app.function(cpu=8, memory=24 * 1024, timeout=4 * 3600,
              volumes={
        "/outputs": MODELS_VOLUME,
        "/datasets": DATASETS,
        "/secryst-datasets": SECRYST_DATASETS,
        "/ud-g2p": URDU_G2P,
        "/ud-diacrit": URDU_DIAC,
        "/persian": PERSIAN,
    })
def generate(n: int = 25) -> dict:
    import hashlib
    import json
    import sys

    sys.path.insert(0, "/root/interscript-ml/runtime/src")
    from interscript_ml import Model

    out_dir = Path("/outputs/golden")
    out_dir.mkdir(exist_ok=True)
    report = {}
    for model_id, (zip_path, test_path) in SOURCES.items():
        zip_full = Path("/outputs") / zip_path
        if not zip_full.exists():
            report[model_id] = "zip missing on volume"
            continue
        if test_path is None:
            report[model_id] = "no test pairs known — skipped"
            continue
        pairs = []
        tp = Path(test_path)
        if not tp.is_absolute():
            tp = Path("/datasets") / tp
        for line in tp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                pairs.append(row.get("src") or row.get("input"))
        pairs = [p for p in pairs if p][:n]
        model = Model.load(zip_full)
        rows = []
        for src in pairs:
            out = model.translate(src, max_len=max(256, 4 * len(src)))
            rows.append({"input": src, "output": out})
        digest = hashlib.sha256()
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        digest.update(body.encode("utf-8"))
        (out_dir / f"{model_id}.jsonl").write_text(body, encoding="utf-8")
        report[model_id] = {"n": len(rows), "sha256": digest.hexdigest()}
        print(f"{model_id}: {len(rows)} rows", flush=True)
    MODELS_VOLUME.commit()
    return report


@app.local_entrypoint()
def main(n: int = 25) -> None:
    print(generate.remote(n))
