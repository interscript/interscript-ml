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

import re
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
    .add_local_dir(str(REPO_ROOT), "/root/interscript-ml", copy=True)
    .workdir("/root/interscript-ml")
)

CHECKPOINT_VOLUMES = {
    "/volumes/secryst-checkpoints": modal.Volume.from_name("secryst-checkpoints"),
    "/volumes/urdu-g2p-checkpoints": modal.Volume.from_name("urdu-g2p-checkpoints"),
    "/volumes/urdu-diacrit-checkpoints": modal.Volume.from_name(
        "urdu-diacrit-checkpoints"
    ),
    "/volumes/rababa-checkpoints": modal.Volume.from_name("rababa-checkpoints"),
    "/volumes/persian-checkpoints": modal.Volume.from_name("persian-g2p-checkpoints"),
}

DATASET_VOLUMES = {
    "/datasets/rababa": modal.Volume.from_name("rababa-datasets"),
    "/datasets/secryst": modal.Volume.from_name("secryst-datasets"),
    "/datasets/urdu-g2p": modal.Volume.from_name("urdu-g2p-datasets"),
    "/datasets/urdu-diacrit": modal.Volume.from_name("urdu-diacrit-datasets"),
    "/datasets/persian": modal.Volume.from_name("persian-g2p-datasets"),
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
        "checkpoint": "rababa_hebrew/run-s46-phonikud-plus/run-002-gold-ft/best",
        "metadata": "models/heb-diac/heb-diac-1.1.metadata.yaml",
        "readme": "models/heb-diac/heb-diac-1.1.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "nakdimon/test-imf.jsonl",
        "probe": "שלום",
    },
    "ara-diac": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_arabic_byt5/run-006-morph/best",
        "metadata": "models/ara-diac/ara-diac-1.0.metadata.yaml",
        "readme": "models/ara-diac/ara-diac-1.0.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "arabic-sadeed-imf/test.jsonl",
        "probe": "مكتبة",
    },
    "ara-diac-small": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_arabic_distill_small/run-002/best",
        "metadata": "models/ara-diac-small/ara-diac-small-1.0.metadata.yaml",
        "readme": "models/ara-diac-small/ara-diac-small-1.0.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "arabic-sadeed-imf/test.jsonl",
        "probe": "قوله",
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
    "heb-diac-small": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_hebrew_distill_small/run-001/best",
        "metadata": "models/heb-diac-small/heb-diac-small-1.0.metadata.yaml",
        "readme": "models/heb-diac-small/heb-diac-small-1.0.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "nakdimon/test-imf.jsonl",
        "probe": "שלום",
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
    "tha-g2p-small": {
        "volume": "/volumes/secryst-checkpoints",
        "checkpoint": "secryst_thai_g2p_distill_small/run-003/best",
        "metadata": "models/tha-g2p-small/tha-g2p-small-1.0.metadata.yaml",
        "readme": "models/tha-g2p-small/tha-g2p-small-1.0.README.md",
        "test_volume": "/datasets/secryst",
        "test_data": "thai-ipa/test.jsonl",
        "probe": "สวัสดี",
    },
    "fas-g2p": {
        "volume": "/volumes/persian-checkpoints",
        "checkpoint": "persian_g2p/run-001/best",
        "metadata": "models/fas-g2p/fas-g2p-1.0.metadata.yaml",
        "readme": "models/fas-g2p/fas-g2p-1.0.README.md",
        "test_volume": "/datasets/persian",
        "test_data": "persian-g2p/test.jsonl",
        "probe": "سلام",
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

    sys.path.insert(0, "/root/interscript-ml/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    metadata_path = Path("/root/interscript-ml") / spec["metadata"]
    readme_path = Path("/root/interscript-ml") / spec["readme"]

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

    sys.path.insert(0, "/root/interscript-ml/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    test_path = Path(spec["test_volume"]) / spec["test_data"]

    from imf.export import load_byte_seq2seq
    from imf.parity import (
        reference_decode,
        run_margin_analysis,
        run_parity,
        write_margin_report,
        write_parity,
    )

    model = load_byte_seq2seq(checkpoint)
    pairs = _load_pairs(test_path)
    if limit:
        pairs = pairs[:limit]

    reference = reference_decode(model, [src for src, _ in pairs], max_len=128)

    out_dir = Path("/outputs/imf") / model_id
    meta_path = Path("/root/interscript-ml", spec["metadata"])
    mid = re.search(r"^id:\s*(\S+)", meta_path.read_text(encoding="utf-8"), re.M).group(1)
    reports: dict[str, str] = {}
    for precision in precisions:
        zip_path = out_dir / f"{mid}-{precision}.zip"
        report = run_parity(model, zip_path, pairs, max_len=128, reference=reference)
        reports[precision] = (
            f"samples={report.samples} cer_ref={report.cer_reference}pp "
            f"cer_onnx={report.cer_onnx}pp delta={report.cer_delta}pp "
            f"mismatches={report.token_mismatches}"
        )
        if not report.passed:
            raise RuntimeError(f"parity gate FAILED for {zip_path.name}")
        write_parity(zip_path, report)
        margins = run_margin_analysis(model, zip_path, pairs, max_len=128)
        write_margin_report(margins, out_dir / f"{mid}-margins-{precision}.json")
        reports[precision] += (
            f" | margin flips={margins.flip_rate:.4%} kld={margins.kld_mean:.2e} "
            f"p10={margins.margin_p10} low-share={margins.flip_low_margin_share}"
        )
        # margin release policy (E1): near-tie flips are inherent to flat
        # byte models, but confident-position flips mean the artifact's
        # decision surface moved. Pre-fix heb-diac int8 measured 7.5%
        # confident flips; every head-fp32 artifact measures < 0.7%.
        confident_flip_rate = margins.flip_rate * (1 - margins.flip_low_margin_share)
        if confident_flip_rate > 0.01:
            raise RuntimeError(
                f"margin gate FAILED for {zip_path.name}: "
                f"{confident_flip_rate:.2%} of positions flip argmax at "
                f"confident margins (budget: 1%)"
            )
    MODELS_VOLUME.commit()
    return reports


@app.function(
    cpu=8,
    memory=32 * 1024,
    timeout=5 * 3600,
    volumes={**CHECKPOINT_VOLUMES, **DATASET_VOLUMES, "/outputs": MODELS_VOLUME},
)
def margin_model(model_id: str, precisions: list[str], limit: int = 0) -> dict[str, str]:
    """Margin analysis alone over already-exported zips — read-only for the
    zips (diagnostic JSON only); validates published artifacts without
    touching their metadata."""
    import sys

    sys.path.insert(0, "/root/interscript-ml/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    test_path = Path(spec["test_volume"]) / spec["test_data"]

    from imf.export import load_byte_seq2seq
    from imf.parity import run_margin_analysis, write_margin_report

    model = load_byte_seq2seq(checkpoint)
    pairs = _load_pairs(test_path)
    if limit:
        pairs = pairs[:limit]

    out_dir = Path("/outputs/imf") / model_id
    meta_path = Path("/root/interscript-ml", spec["metadata"])
    mid = re.search(r"^id:\s*(\S+)", meta_path.read_text(encoding="utf-8"), re.M).group(1)
    reports: dict[str, str] = {}
    for precision in precisions:
        zip_path = out_dir / f"{mid}-{precision}.zip"
        if not zip_path.exists():
            reports[precision] = "zip not exported (skipped)"
            continue
        report = run_margin_analysis(model, zip_path, pairs, max_len=128)
        write_margin_report(report, out_dir / f"{mid}-margins-{precision}.json")
        reports[precision] = (
            f"samples={report.samples} tokens={report.tokens} "
            f"flips={report.flipped_tokens} ({report.flip_rate:.4%}) "
            f"kld={report.kld_mean:.2e} margins p1/p10/p50="
            f"{report.margin_p1}/{report.margin_p10}/{report.margin_p50} "
            f"low-margin-flip-share={report.flip_low_margin_share}"
        )
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


@app.local_entrypoint()
def margins(model: str, precisions: str = "fp32,fp16,int8", limit: int = 0) -> None:
    reports = margin_model.remote(model, precisions.split(","), limit)
    for precision, status in reports.items():
        print(f"{model} [{precision}] {status}")


@app.function(
    cpu=8,
    memory=32 * 1024,
    timeout=5 * 3600,
    volumes={**CHECKPOINT_VOLUMES, **DATASET_VOLUMES, "/outputs": MODELS_VOLUME},
)
def int8_pc_probe(model_id: str = "heb-diac", limit: int = 300) -> dict:
    """E1 follow-up: does per-channel int8 remove the confident-position
    argmax flips? Rebuilds the int8 graphs from the fp32 zip with
    per_channel=True, packages them as a probe zip (copy of the shipped
    int8 zip with graphs swapped — NOT a release artifact), and compares
    margin reports on the same pairs."""
    import sys
    import tempfile
    import zipfile

    sys.path.insert(0, "/root/interscript-ml/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    test_path = Path(spec["test_volume"]) / spec["test_data"]

    from imf.export import load_byte_seq2seq, quantize_int8
    from imf.parity import run_margin_analysis

    model = load_byte_seq2seq(checkpoint)
    pairs = _load_pairs(test_path)[:limit]

    out_dir = Path("/outputs/imf") / model_id
    meta_path = Path("/root/interscript-ml", spec["metadata"])
    mid = re.search(r"^id:\s*(\S+)", meta_path.read_text(encoding="utf-8"), re.M).group(1)
    fp32_zip = out_dir / f"{mid}-fp32.zip"
    int8_zip = out_dir / f"{mid}-int8.zip"
    if not fp32_zip.exists() or not int8_zip.exists():
        raise RuntimeError(f"need both {fp32_zip.name} and {int8_zip.name} on the volume")

    shipped = run_margin_analysis(model, int8_zip, pairs, max_len=128)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(fp32_zip) as zf:
            zf.extract("encoder.onnx", tmp)
            dec = "decoder-kv.onnx" if "decoder-kv.onnx" in zf.namelist() else "decoder.onnx"
            zf.extract(dec, tmp)
        enc_pc = tmp / "encoder-pc.onnx"
        dec_pc = tmp / dec.replace(".onnx", "-pc.onnx")
        quantize_int8(tmp / "encoder.onnx", enc_pc, per_channel=True)
        quantize_int8(tmp / dec, dec_pc, per_channel=True)

        probe_zip = tmp / f"{mid}-int8-pc-probe.zip"
        with zipfile.ZipFile(int8_zip) as src, zipfile.ZipFile(
            probe_zip, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for name in src.namelist():
                if name == "encoder.onnx":
                    dst.writestr(name, enc_pc.read_bytes())
                elif name == dec:
                    dst.writestr(name, dec_pc.read_bytes())
                else:
                    dst.writestr(name, src.read(name))

        per_channel_report = run_margin_analysis(model, probe_zip, pairs, max_len=128)
        size_shipped = int8_zip.stat().st_size
        size_probe = probe_zip.stat().st_size

    def row(r):
        return {
            "flips": r.flipped_tokens, "tokens": r.tokens,
            "flip_rate": r.flip_rate, "kld_mean": r.kld_mean,
            "flip_low_margin_share": r.flip_low_margin_share,
        }

    return {
        "model": model_id, "pairs": len(pairs),
        "shipped_int8": row(shipped), "per_channel_int8": row(per_channel_report),
        "size_bytes": {"shipped": size_shipped, "per_channel": size_probe},
    }


@app.local_entrypoint()
def int8_pc(model: str = "heb-diac", limit: int = 300) -> None:
    print(int8_pc_probe.remote(model, limit))


@app.function(
    cpu=8,
    memory=32 * 1024,
    timeout=5 * 3600,
    volumes={**CHECKPOINT_VOLUMES, **DATASET_VOLUMES, "/outputs": MODELS_VOLUME},
)
def int8_head_probe(model_id: str = "heb-diac", limit: int = 300) -> dict:
    """E1 follow-up 2: per-channel alone did NOT fix heb-diac's 9.3%
    flip rate (8.5% remaining, 78% still at confident positions). This
    probe keeps the logits-producing MatMul (the tied head) in fp32 and
    quantizes only the body — per-tensor and per-channel variants."""
    import sys
    import tempfile
    import zipfile

    sys.path.insert(0, "/root/interscript-ml/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    test_path = Path(spec["test_volume"]) / spec["test_data"]

    from imf.export import head_matmul_names, load_byte_seq2seq
    from imf.parity import run_margin_analysis

    model = load_byte_seq2seq(checkpoint)
    pairs = _load_pairs(test_path)[:limit]

    out_dir = Path("/outputs/imf") / model_id
    meta_path = Path("/root/interscript-ml", spec["metadata"])
    mid = re.search(r"^id:\s*(\S+)", meta_path.read_text(encoding="utf-8"), re.M).group(1)
    fp32_zip = out_dir / f"{mid}-fp32.zip"
    int8_zip = out_dir / f"{mid}-int8.zip"
    if not fp32_zip.exists() or not int8_zip.exists():
        raise RuntimeError(f"need both {fp32_zip.name} and {int8_zip.name} on the volume")

    shipped = run_margin_analysis(model, int8_zip, pairs, max_len=128)

    results: dict = {
        "model": model_id, "pairs": len(pairs),
        "shipped_int8": {
            "flips": shipped.flipped_tokens, "tokens": shipped.tokens,
            "flip_rate": shipped.flip_rate, "kld_mean": shipped.kld_mean,
            "flip_low_margin_share": shipped.flip_low_margin_share,
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(fp32_zip) as zf:
            zf.extract("encoder.onnx", tmp)
            dec = "decoder-kv.onnx" if "decoder-kv.onnx" in zf.namelist() else "decoder.onnx"
            zf.extract(dec, tmp)

        head_nodes = head_matmul_names(tmp / dec)
        results["head_nodes_excluded"] = head_nodes

        from onnxruntime.quantization import QuantType, quantize_dynamic

        for variant, per_channel in (("head32", False), ("head32_pc", True)):
            enc_q = tmp / f"encoder-{variant}.onnx"
            dec_q = tmp / dec.replace(".onnx", f"-{variant}.onnx")
            quantize_dynamic(
                str(tmp / "encoder.onnx"), str(enc_q),
                weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                per_channel=per_channel,
            )
            quantize_dynamic(
                str(tmp / dec), str(dec_q),
                weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                per_channel=per_channel, nodes_to_exclude=head_nodes,
            )
            probe_zip = tmp / f"{mid}-int8-{variant}-probe.zip"
            with zipfile.ZipFile(int8_zip) as src, zipfile.ZipFile(
                probe_zip, "w", zipfile.ZIP_DEFLATED
            ) as dst:
                for name in src.namelist():
                    if name == "encoder.onnx":
                        dst.writestr(name, enc_q.read_bytes())
                    elif name == dec:
                        dst.writestr(name, dec_q.read_bytes())
                    else:
                        dst.writestr(name, src.read(name))
            report = run_margin_analysis(model, probe_zip, pairs, max_len=128)
            results[f"int8_{variant}"] = {
                "flips": report.flipped_tokens, "tokens": report.tokens,
                "flip_rate": report.flip_rate, "kld_mean": report.kld_mean,
                "flip_low_margin_share": report.flip_low_margin_share,
                "size_bytes": probe_zip.stat().st_size,
            }
    return results


@app.local_entrypoint()
def int8_head(model: str = "heb-diac", limit: int = 300) -> None:
    print(int8_head_probe.remote(model, limit))
