"""Modal app: export IMF v1 zips from checkpoints on Modal volumes (WO02).

One CPU function per model — exports never compete with A100 training.

    modal run --detach src/gpu/modal_export.py --model khm-latn
    modal run --detach src/gpu/modal_export.py --model urd-g2p --precisions fp16,int8

Watchdog (server evictions happen; export is idempotent, so retries are
the resume mechanism — each model's zips are written atomically at the
end, and per-model work is independent):

    until modal run --detach src/gpu/modal_export.py --model khm-latn; do sleep 60; done

Outputs land on the secryst-models volume under /imf/<model>/.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=5.0",
        "onnx>=1.16",
        "onnxruntime>=1.17",
        "pyyaml>=6.0",
    )
    .copy_directory(str(REPO_ROOT), "/root/ml-models")
    .workdir("/root/ml-models")
)

CHECKPOINT_VOLUMES = {
    "/volumes/secryst-checkpoints": modal.Volume.from_name("secryst-checkpoints"),
    "/volumes/urdu-g2p-checkpoints": modal.Volume.from_name("urdu-g2p-checkpoints"),
    "/volumes/urdu-diacrit-checkpoints": modal.Volume.from_name(
        "urdu-diacrit-checkpoints"
    ),
    "/volumes/rababa-checkpoints": modal.Volume.from_name("rababa-checkpoints"),
}

MODELS_VOLUME = modal.Volume.from_name("secryst-models")

# model id -> (checkpoint volume mount, checkpoint path, metadata, readme)
MODELS: dict[str, dict[str, str]] = {
    "khm-latn": {
        "volume": "/volumes/secryst-checkpoints",
        "checkpoint": "/khmer_byt5/run-001/best",
        "metadata": "models/khm-latn/khm-latn-1.0.metadata.yaml",
        "readme": "models/khm-latn/khm-latn-1.0.README.md",
        "probe": "ភាសា",
    },
    "urd-g2p": {
        "volume": "/volumes/urdu-g2p-checkpoints",
        "checkpoint": "/urdu_g2p/run-001/best",
        "metadata": "models/urd-g2p/urd-g2p-1.0.metadata.yaml",
        "readme": "models/urd-g2p/urd-g2p-1.0.README.md",
        "probe": "اردو",
    },
    "urd-diac": {
        "volume": "/volumes/urdu-diacrit-checkpoints",
        "checkpoint": "/urdu_diacrit/run-001/best",
        "metadata": "models/urd-diac/urd-diac-1.0.metadata.yaml",
        "readme": "models/urd-diac/urd-diac-1.0.README.md",
        "probe": "اردو",
    },
}

app = modal.App("interscript-ml-export", image=IMAGE)


@app.function(
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={**CHECKPOINT_VOLUMES, "/outputs": MODELS_VOLUME},
)
def export_model(model_id: str, precisions: list[str]) -> dict[str, str]:
    import sys

    sys.path.insert(0, "/root/ml-models/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    metadata_path = Path("/root/ml-models") / spec["metadata"]
    readme_path = Path("/root/ml-models") / spec["readme"]

    from imf.export import export_zips, load_byte_seq2seq, onnx_greedy_kv
    from imf.validator import validate_zip

    model = load_byte_seq2seq(checkpoint)
    out_dir = Path("/outputs/imf") / model_id
    zips = export_zips(
        model,
        metadata_path,
        readme_path.read_text(encoding="utf-8"),
        out_dir,
        precisions=tuple(precisions),
    )
    MODELS_VOLUME.commit()

    report: dict[str, str] = {}
    import zipfile

    import onnxruntime as ort

    for z in zips:
        result = validate_zip(z)
        if not result.ok:
            raise RuntimeError(f"{z.name} failed validation: {result.errors}")
        with zipfile.ZipFile(z) as zf:
            zf.extract("encoder.onnx", "/tmp/check")
            zf.extract("decoder-kv.onnx", "/tmp/check")
        enc = ort.InferenceSession(
            "/tmp/check/encoder.onnx", providers=["CPUExecutionProvider"]
        )
        kv = ort.InferenceSession(
            "/tmp/check/decoder-kv.onnx", providers=["CPUExecutionProvider"]
        )
        tokens = onnx_greedy_kv(enc, kv, spec["probe"], max_len=32)
        report[z.name] = f"{z.stat().st_size} bytes, probe -> {len(tokens)} tokens"
    return report


@app.local_entrypoint()
def main(model: str, precisions: str = "fp32,fp16,int8") -> None:
    report = export_model.remote(model, precisions.split(","))
    for name, status in report.items():
        print(f"{name}: {status}")
