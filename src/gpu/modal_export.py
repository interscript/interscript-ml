"""Modal app: export IMF v1 zips from checkpoints on Modal volumes (WO02),
then gate them with the WO03 parity check — all CPU, never competing
with A100 training.

    modal run --detach src/gpu/modal_export.py --model khm-latn
    modal run --detach src/gpu/modal_export.py::parity --model khm-latn

Watchdog (server evictions happen; both steps are idempotent — retries
are the resume mechanism, and each model's zips are written atomically):

    until modal run --detach src/gpu/modal_export.py --model khm-latn; do sleep 60; done

Zips land on the secryst-models volume under /imf/<model>/; parity is
written into the zip in place (a zip is only shippable strict-validated).
Versions are pinned to the ones the export was verified against locally
(transformers 5.15 breaks T5 tracing with "multiple values for
use_cache").
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "transformers==5.14.1",
        "onnx==1.22.0",
        "onnxruntime==1.23.2",
        "pyyaml>=6.0",
    )
    .add_local_dir(str(REPO_ROOT), "/root/ml-models", copy=True)
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

DATASET_VOLUMES = {
    "/datasets/rababa": modal.Volume.from_name("rababa-datasets"),
    "/datasets/secryst": modal.Volume.from_name("secryst-datasets"),
    "/datasets/urdu-g2p": modal.Volume.from_name("urdu-g2p-datasets"),
    "/datasets/urdu-diacrit": modal.Volume.from_name("urdu-diacrit-datasets"),
}

MODELS_VOLUME = modal.Volume.from_name("secryst-models")

MODELS: dict[str, dict[str, str]] = {
    "khm-latn": {
        "volume": "/volumes/secryst-checkpoints",
        "checkpoint": "khmer_byt5/run-001/best",
        "metadata": "models/khm-latn/khm-latn-1.0.metadata.yaml",
        "readme": "models/khm-latn/khm-latn-1.0.README.md",
        "test_volume": "/datasets/secryst",
        "test_data": "khmer-translit/test.jsonl",
        "probe": "ភាសា",
    },
    "urd-g2p": {
        "volume": "/volumes/urdu-g2p-checkpoints",
        "checkpoint": "urdu_g2p/run-001/best",
        "metadata": "models/urd-g2p/urd-g2p-1.0.metadata.yaml",
        "readme": "models/urd-g2p/urd-g2p-1.0.README.md",
        "test_volume": "/datasets/urdu-g2p",
        "test_data": "urdu-g2p/test.jsonl",
        "probe": "اردو",
    },
    "heb-diac": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_hebrew_byt5_s43/run-001/best",
        "metadata": "models/heb-diac/heb-diac-1.0.metadata.yaml",
        "readme": "models/heb-diac/heb-diac-1.0.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "nakdimon/test-imf.jsonl",
        "probe": "שלום",
    },
    "urd-diac": {
        "volume": "/volumes/urdu-diacrit-checkpoints",
        "checkpoint": "urdu_diacrit/run-001/best",
        "metadata": "models/urd-diac/urd-diac-1.0.metadata.yaml",
        "readme": "models/urd-diac/urd-diac-1.0.README.md",
        "test_volume": "/datasets/urdu-diacrit",
        "test_data": "urdu-diacrit/test.jsonl",
        "probe": "اردو",
    },
    "tha-g2p-base": {
        "volume": "/volumes/secryst-checkpoints",
        "checkpoint": "secryst_thai_g2p_distill_small/run-004/best",
        "metadata": "models/tha-g2p-base/tha-g2p-base-1.0.metadata.yaml",
        "readme": "models/tha-g2p-base/tha-g2p-base-1.0.README.md",
        "test_volume": "/datasets/secryst",
        "test_data": "thai-ipa/test.jsonl",
        "probe": "สวัสดี",
    },
}

app = modal.App("interscript-ml-export", image=IMAGE)


def _load_pairs(path: Path) -> list[tuple[str, str]]:
    import json

    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            pairs.append(
                (
                    row.get("input", row.get("src", "")),
                    row.get("target", row.get("tgt", row.get("gold", ""))),
                )
            )
        else:
            pairs.append((row[0], row[1] if len(row) > 1 else ""))
    return pairs


@app.function(
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={**CHECKPOINT_VOLUMES, **DATASET_VOLUMES, "/outputs": MODELS_VOLUME},
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

    import zipfile

    import onnxruntime as ort

    report: dict[str, str] = {}
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


@app.function(
    cpu=8,
    memory=32 * 1024,
    # heb-diac (ByT5-base, 1,864 long sentences) needs >2h; the batched
    # reference cut the Urdu gates to ~1h but not this one.
    timeout=5 * 3600,
    volumes={**CHECKPOINT_VOLUMES, **DATASET_VOLUMES, "/outputs": MODELS_VOLUME},
)
def parity_model(model_id: str, precisions: list[str], limit: int = 0) -> dict[str, str]:
    """WO03 gate on Modal: torch reference vs ONNX decode over the test
    split; writes the parity block into each zip (strict gate enforced)."""
    import sys

    sys.path.insert(0, "/root/ml-models/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    test_path = Path(spec["test_volume"]) / spec["test_data"]

    from imf.export import load_byte_seq2seq
    from imf.parity import reference_decode, run_parity, write_parity

    model = load_byte_seq2seq(checkpoint)
    pairs = _load_pairs(test_path)
    if limit:
        pairs = pairs[:limit]

    reference = reference_decode(model, [src for src, _ in pairs], max_len=128)

    out_dir = Path("/outputs/imf") / model_id
    reports: dict[str, str] = {}
    for precision in precisions:
        zip_path = out_dir / f"{model_id}-1.0-{precision}.zip"
        report = run_parity(model, zip_path, pairs, max_len=128, reference=reference)
        reports[precision] = (
            f"samples={report.samples} cer_ref={report.cer_reference}pp "
            f"cer_onnx={report.cer_onnx}pp delta={report.cer_delta}pp "
            f"mismatches={report.token_mismatches}"
        )
        if not report.passed:
            raise RuntimeError(f"parity gate FAILED for {zip_path.name}")
        write_parity(zip_path, report)
    MODELS_VOLUME.commit()
    return reports


@app.local_entrypoint()
def main(model: str, precisions: str = "fp32,fp16,int8") -> None:
    report = export_model.remote(model, precisions.split(","))
    for name, status in report.items():
        print(f"{name}: {status}")


@app.local_entrypoint()
def parity(model: str, precisions: str = "fp32,fp16,int8", limit: int = 0) -> None:
    reports = parity_model.remote(model, precisions.split(","), limit)
    for precision, status in reports.items():
        print(f"{model} [{precision}] {status}")
