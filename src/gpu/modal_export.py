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
    "ara-diac2": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_arabic_byt5/run-007-news/best",
        "metadata": "models/ara-diac/ara-diac-2.0.metadata.yaml",
        "readme": "models/ara-diac/ara-diac-2.0.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "arabic-sadeed-imf/test.jsonl",
        "probe": "مكتبة",
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
    "ara-diac-small-2": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_arabic_distill_small/run-006-r7-muon/best",
        "metadata": "models/ara-diac-small/ara-diac-small-2.0.metadata.yaml",
        "readme": "models/ara-diac-small/ara-diac-small-2.0.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "arabic-sadeed-imf/test.jsonl",
        "probe": "قوله",
    },
    "ara-diac-small-21": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_arabic_distill_small/run-007-r7-muon-6ep/best",
        "metadata": "models/ara-diac-small/ara-diac-small-2.1.metadata.yaml",
        "readme": "models/ara-diac-small/ara-diac-small-2.1.README.md",
        "test_volume": "/datasets/rababa",
        "test_data": "arabic-sadeed-imf/test.jsonl",
        "probe": "قوله",
    },
    "ara-diac-layerdrop": {
        "volume": "/volumes/rababa-checkpoints",
        "checkpoint": "rababa_arabic_distill_small/run-009-layerdrop-6ep/best",
        "metadata": "models/ara-diac-layerdrop/ara-diac-layerdrop-1.0.metadata.yaml",
        "readme": "models/ara-diac-layerdrop/ara-diac-layerdrop-1.0.README.md",
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


def normalize_precisions(precisions: "str | list[str]") -> list[str]:
    """Accept both invocation forms: the parity/margins entrypoints pass
    a pre-split list, direct ::parity_model-style CLI calls pass a
    comma string."""
    if isinstance(precisions, str):
        precisions = precisions.split(",")
    return [q.strip() for q in precisions if q.strip()]


def pending_precisions(
    out_dir: Path, mid: str, precisions: "str | list[str]"
) -> list[str]:
    """Precision stages still to run: the margin report is the last
    artifact a stage writes, so its presence means the stage (parity
    block in the zip included) completed durably. Preemption restarts
    resume at the next stage instead of redoing hours of decode."""
    return [
        p
        for p in normalize_precisions(precisions)
        if not (out_dir / f"{mid}-margins-{p}.json").exists()
    ]


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
def export_model(model_id: str, precisions: str = "fp32,fp16,int8") -> dict[str, str]:
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
        precisions=tuple(p.strip() for p in precisions.split(",") if p.strip()),
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
def parity_model(
    model_id: str, precisions: str = "fp32,fp16,int8", limit: int = 0
) -> dict[str, str]:
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

    def stage(event: str) -> None:
        # the gate's observable interface: durable stage log on the
        # volume (silent container kills are otherwise undiagnosable —
        # seven consecutive ara-diac2 attempts died without a traceback)
        import time as _time

        out_dir0 = Path("/outputs/imf") / model_id
        out_dir0.mkdir(parents=True, exist_ok=True)
        with (out_dir0 / "parity_stages.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(f'{{"t": {round(_time.time())}, "event": "{event}"}}\n')
        MODELS_VOLUME.commit()

    stage(f"start model={model_id} pairs={len(pairs)}")
    reference = reference_decode(
        model,
        [src for src, _ in pairs],
        max_len=128,
        resume_path=Path("/outputs/imf") / model_id / "reference_progress.jsonl",
    )
    stage("reference-decode done")

    out_dir = Path("/outputs/imf") / model_id
    meta_path = Path("/root/interscript-ml", spec["metadata"])
    mid = re.search(r"^id:\s*(\S+)", meta_path.read_text(encoding="utf-8"), re.M).group(1)
    reports: dict[str, str] = {}
    todo = pending_precisions(out_dir, mid, precisions)
    for precision in todo:
        zip_path = out_dir / f"{mid}-{precision}.zip"
        stage(f"onnx decode {precision}")
        report = run_parity(model, zip_path, pairs, max_len=128, reference=reference)
        stage(f"parity report {precision} delta={report.cer_delta}")
        reports[precision] = (
            f"samples={report.samples} cer_ref={report.cer_reference}pp "
            f"cer_onnx={report.cer_onnx}pp delta={report.cer_delta}pp "
            f"mismatches={report.token_mismatches}"
        )
        if not report.passed:
            raise RuntimeError(f"parity gate FAILED for {zip_path.name}")
        write_parity(zip_path, report)
        margins = run_margin_analysis(model, zip_path, pairs, max_len=128)
        stage(f"margin report {precision} flips={margins.flip_rate:.4%}")
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
    timeout=12 * 3600,
    volumes={**CHECKPOINT_VOLUMES, **DATASET_VOLUMES, "/outputs": MODELS_VOLUME},
)
def margin_model(
    model_id: str, precisions: list[str], limit: int = 0, dump_positions: bool = False
) -> dict[str, str]:
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
    for precision in normalize_precisions(precisions):
        zip_path = out_dir / f"{mid}-{precision}.zip"
        if not zip_path.exists():
            reports[precision] = "zip not exported (skipped)"
            continue
        dump = out_dir / f"{mid}-positions-{precision}.jsonl" if dump_positions else None
        report = run_margin_analysis(model, zip_path, pairs, max_len=128, dump_positions=dump)
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
    report = export_model.remote(model, precisions)
    for name, status in report.items():
        print(f"{name}: {status}")


@app.local_entrypoint()
def parity(model: str, precisions: str = "fp32,fp16,int8", limit: int = 0) -> None:
    reports = parity_model.remote(model, precisions.split(","), limit)
    for precision, status in reports.items():
        print(f"{model} [{precision}] {status}")


@app.local_entrypoint()
def margins(
    model: str, precisions: str = "fp32,fp16,int8", limit: int = 0, dump_positions: bool = False
) -> None:
    reports = margin_model.remote(model, precisions.split(","), limit, dump_positions)
    for precision, status in reports.items():
        print(f"{model} [{precision}] {status}")

@app.function(
    cpu=8,
    memory=32 * 1024,
    timeout=5 * 3600,
    volumes={**CHECKPOINT_VOLUMES, **DATASET_VOLUMES, "/outputs": MODELS_VOLUME},
)
def rebuild_int8_head32(model_id: str, limit: int = 0) -> dict:
    """Rebuild a shipped int8 zip with the head kept in fp32 (the E1
    fix) and run the full release gate stack on it: CER parity (written
    into the zip) + margin analysis + the confident-flip budget.

    The corrected artifact lands as {mid}-int8-head32.zip next to the
    shipped one; swapping it into the release name is a version-number
    decision, made separately."""
    import sys
    import tempfile
    import zipfile

    sys.path.insert(0, "/root/interscript-ml/src")

    spec = MODELS[model_id]
    checkpoint = Path(spec["volume"]) / spec["checkpoint"]
    test_path = Path(spec["test_volume"]) / spec["test_data"]

    from imf.export import (
        head_matmul_names,
        load_byte_seq2seq,
        quantize_int8,
        refresh_member_shas,
    )
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

    out_dir = Path("/outputs/imf") / model_id
    meta_path = Path("/root/interscript-ml", spec["metadata"])
    mid = re.search(r"^id:\s*(\S+)", meta_path.read_text(encoding="utf-8"), re.M).group(1)
    fp32_zip = out_dir / f"{mid}-fp32.zip"
    int8_zip = out_dir / f"{mid}-int8.zip"
    for path in (fp32_zip, int8_zip):
        if not path.exists():
            raise RuntimeError(f"{path.name} missing on the volume")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(fp32_zip) as zf:
            zf.extract("encoder.onnx", tmp)
            dec = "decoder-kv.onnx" if "decoder-kv.onnx" in zf.namelist() else "decoder.onnx"
            zf.extract(dec, tmp)
        enc_q, dec_q = tmp / "encoder-h32.onnx", tmp / dec.replace(".onnx", "-h32.onnx")
        quantize_int8(tmp / "encoder.onnx", enc_q)
        quantize_int8(tmp / dec, dec_q, nodes_to_exclude=head_matmul_names(tmp / dec))

        new_zip = out_dir / f"{mid}-int8-head32.zip"
        with zipfile.ZipFile(int8_zip) as src, zipfile.ZipFile(
            new_zip, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for name in src.namelist():
                if name == "encoder.onnx":
                    dst.writestr(name, enc_q.read_bytes())
                elif name == dec:
                    dst.writestr(name, dec_q.read_bytes())
                else:
                    dst.writestr(name, src.read(name))

    # re-quantized graphs replaced the members; the internal sha table
    # must be refreshed or strict validation (write_parity) rejects the zip
    refresh_member_shas(new_zip)

    reference = reference_decode(
        model,
        [s for s, _ in pairs],
        max_len=128,
        resume_path=Path("/outputs/imf") / model_id / "reference_progress.jsonl",
    )
    report = run_parity(model, new_zip, pairs, max_len=128, reference=reference)
    if not report.passed:
        raise RuntimeError(f"parity gate FAILED for {new_zip.name}: {report}")
    write_parity(new_zip, report)
    margins = run_margin_analysis(model, new_zip, pairs, max_len=128)
    write_margin_report(margins, out_dir / f"{mid}-int8-head32-margins.json")
    confident = margins.flip_rate * (1 - margins.flip_low_margin_share)
    if confident > 0.01:
        raise RuntimeError(
            f"margin gate FAILED for {new_zip.name}: {confident:.2%} confident flips"
        )
    MODELS_VOLUME.commit()
    return {
        "model": model_id, "zip": new_zip.name,
        "parity": {"samples": report.samples, "cer_delta": report.cer_delta},
        "margins": {"flip_rate": margins.flip_rate, "kld": margins.kld_mean,
                    "low_share": margins.flip_low_margin_share,
                    "confident_flip_rate": round(confident, 6)},
        "size_bytes": new_zip.stat().st_size,
    }


@app.local_entrypoint()
def rebuild_int8(model: str, limit: int = 0) -> None:
    print(rebuild_int8_head32.remote(model, limit))



@app.function(
    cpu=1,
    memory=1024,
    timeout=600,
    volumes={"/outputs": MODELS_VOLUME},
)
def zip_meta(model_id: str, precision: str) -> dict:
    """Read metadata.yaml out of a zip on the volume — one-shot provenance
    check (parity block, precision) without multi-GB downloads."""
    import sys
    import zipfile

    import yaml

    sys.path.insert(0, "/root/interscript-ml/src")
    out_dir = Path("/outputs/imf") / model_id
    meta_path = Path("/root/interscript-ml", MODELS[model_id]["metadata"])
    mid = re.search(r"^id:\s*(\S+)", meta_path.read_text(encoding="utf-8"), re.M).group(1)
    with zipfile.ZipFile(out_dir / f"{mid}-{precision}.zip") as zf:
        m = yaml.safe_load(zf.read("metadata.yaml"))
        return {"id": m["id"], "precision": m.get("precision"),
                "parity": m.get("parity"), "sha_members": len(m.get("sha256", {}))}


@app.local_entrypoint()
def zmeta(model: str, precisions: str = "fp32,fp16,int8") -> None:
    for precision in precisions.split(","):
        print(precision, zip_meta.remote(model, precision))
