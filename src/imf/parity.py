"""WO03 parity gate: ONNX greedy vs the torch reference, precision-aware
CER-delta limits (0.2pp fp32, 1.0pp fp16, 2.0pp int8, 3.0pp int4).

The reference is the transformers decoder loop itself (the exact math the
export wraps) rather than ``model.generate`` — generate's behavior is
config-dependent (eos/start-token defaults) and none of it is implemented
by the runtimes. Comparing against the module-level math catches exactly
what export bugs can break.

``write_parity`` rewrites the parity block inside an existing zip
(metadata.yaml is never sha256-covered, graphs are untouched) and then
requires the zip to pass strict validation — the release gate.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from framework.evaluator import char_error_rate
from imf.export import (
    BYTE_OFFSET,
    EOS_ID,
    PAD_ID,
    _zero_pasts,
    encode_bytes,
    onnx_greedy_kv,
)
from imf.schema import ModelMetadata, Parity


@dataclass(frozen=True)
class ParityReport:
    samples: int
    cer_reference: float  # percentage points
    cer_onnx: float
    cer_delta: float
    token_mismatches: int
    precision: str = "fp32"

    @property
    def passed(self) -> bool:
        return (
            self.cer_delta <= Parity.max_cer_delta(self.precision)
            and self.samples >= Parity.MIN_SAMPLES
        )


@dataclass(frozen=True)
class MarginReport:
    """Teacher-forced fragility analysis: what the CER gate cannot see.

    Byte students have flat top-1 margins, so quantization noise can flip
    near-tie argmaxes without moving CER on a golden set. This report
    measures that directly: per-position top1−top2 margins of the torch
    reference, argmax flip rate against the zip, KL divergence, and the
    share of flips that land on near-tie positions (benign) versus
    confident ones (dangerous)."""

    samples: int
    tokens: int
    flipped_tokens: int
    flip_rate: float  # fraction of teacher-forced positions with argmax disagreement
    kld_mean: float  # mean KL(reference || zip) over positions
    margin_p1: float  # reference top1−top2 margin quantiles, in logits
    margin_p10: float
    margin_p50: float
    flip_low_margin_share: float  # flips at margin < p10 / all flips (1.0 = benign)
    precision: str = "fp32"


def _torch_greedy_tokens(model, text: str, max_len: int) -> list[int]:
    import torch

    ids = torch.tensor([encode_bytes(text)], dtype=torch.long)
    if ids.shape[1] == 1:
        return []
    enc = model.get_encoder()(input_ids=ids)[0]
    dec_ids = torch.tensor([[PAD_ID]], dtype=torch.long)
    outs: list[int] = []
    for _ in range(max_len):
        hidden = model.get_decoder()(
            input_ids=dec_ids, encoder_hidden_states=enc
        )[0]
        logits = model.lm_head(hidden * (model.config.d_model ** -0.5))
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS_ID:
            break
        outs.append(nxt)
        dec_ids = torch.cat([dec_ids, torch.tensor([[nxt]], dtype=torch.long)], 1)
    return outs


def _decode_tokens(tokens: list[int]) -> str:
    return bytes((t - BYTE_OFFSET) % 256 for t in tokens).decode(
        "utf-8", errors="replace"
    )


def _sessions_from_zip(zip_path: Path):
    import tempfile

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("encoder.onnx", tmp)
            decoder = "decoder-kv.onnx" if "decoder-kv.onnx" in zf.namelist() else "decoder.onnx"
            zf.extract(decoder, tmp)
        enc = ort.InferenceSession(
            str(Path(tmp) / "encoder.onnx"), providers=["CPUExecutionProvider"]
        )
        dec = ort.InferenceSession(
            str(Path(tmp) / decoder), providers=["CPUExecutionProvider"]
        )
        return enc, dec


def reference_decode(model, sources, max_len: int = 256) -> list[list[int]]:
    """Torch-reference greedy decode of many inputs, computed once and
    shared across precision variants by run_parity."""
    return [_torch_greedy_tokens(model, source, max_len) for source in sources]


def run_parity(
    model, zip_path: Path | str, pairs, max_len: int = 256, reference=None
) -> ParityReport:
    """pairs: iterable of (source_text, gold_target). Measures both sides
    against gold; the gate is the CER distance between the two.

    ``reference`` (from ``reference_decode``) skips the torch side — the
    reference is precision-independent, so multi-zip gates decode it once.
    """
    zip_path = Path(zip_path)
    enc, kv = _sessions_from_zip(zip_path)

    import yaml

    with zipfile.ZipFile(zip_path) as zf:
        precision = yaml.safe_load(zf.read("metadata.yaml"))["precision"]

    n = 0
    mismatches = 0
    cer_ref_sum = 0.0
    cer_onnx_sum = 0.0
    for i, (source, gold) in enumerate(pairs):
        n += 1
        ref = reference[i] if reference is not None else _torch_greedy_tokens(
            model, source, max_len
        )
        got = onnx_greedy_kv(enc, kv, source, max_len)
        if ref != got:
            mismatches += 1
        cer_ref_sum += char_error_rate(_decode_tokens(ref), gold)
        cer_onnx_sum += char_error_rate(_decode_tokens(got), gold)

    cer_ref = 100.0 * cer_ref_sum / max(n, 1)
    cer_onnx = 100.0 * cer_onnx_sum / max(n, 1)
    return ParityReport(
        samples=n,
        cer_reference=round(cer_ref, 4),
        cer_onnx=round(cer_onnx, 4),
        cer_delta=round(abs(cer_onnx - cer_ref), 4),
        token_mismatches=mismatches,
        precision=precision,
    )


def _margin_stats(ref, got):
    """Per-position stats for one sequence. ``ref``/``got`` are (T, V)
    float64 teacher-forced logits from the torch reference and the zip."""
    import numpy as np

    top2 = np.partition(ref, -2, axis=-1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]
    flips = ref.argmax(axis=-1) != got.argmax(axis=-1)

    ref_s = np.exp(ref - ref.max(axis=-1, keepdims=True))
    ref_s = ref_s / ref_s.sum(axis=-1, keepdims=True)
    got_s = np.exp(got - got.max(axis=-1, keepdims=True))
    got_s = got_s / got_s.sum(axis=-1, keepdims=True)
    kld = (ref_s * (np.log(ref_s + 1e-12) - np.log(got_s + 1e-12))).sum(axis=-1)
    return margins, flips, kld


def _torch_forced_logits(model, source: str, target_ids: list[int]):
    """Teacher-forced decoder logits — the exact math ``_torch_greedy_tokens``
    runs one step of, computed for every position at once."""
    import torch

    src = torch.tensor([encode_bytes(source)], dtype=torch.long)
    if src.shape[1] == 1:
        raise ValueError("source must be at least one byte")
    dec_ids = torch.tensor([[PAD_ID] + target_ids[:-1]], dtype=torch.long)
    enc = model.get_encoder()(input_ids=src)[0]
    hidden = model.get_decoder()(input_ids=dec_ids, encoder_hidden_states=enc)[0]
    return model.lm_head(hidden * (model.config.d_model**-0.5))[0]


def _onnx_forced_logits(enc_sess, dec_sess, source: str, target_ids: list[int]):
    """Teacher-forced logits from the zip's decoder graph. Works for both
    the plain decoder (one full-sequence call) and the KV decoder (zero
    pasts + full input_ids is the same computation)."""
    import numpy as np

    ids = np.array([encode_bytes(source)], dtype=np.int64)
    hidden = enc_sess.run(None, {"input_ids": ids})[0]
    dec_ids = np.array([[PAD_ID] + target_ids[:-1]], dtype=np.int64)
    inputs = {i.name for i in dec_sess.get_inputs()}
    if "encoder_hidden_states" in inputs and not any(
        name.startswith("past_") for name in inputs
    ):
        return dec_sess.run(
            None, {"input_ids": dec_ids, "encoder_hidden_states": hidden}
        )[0][0]
    out_names = [o.name for o in dec_sess.get_outputs()]
    out = dec_sess.run(
        None,
        {"input_ids": dec_ids, "encoder_hidden_states": hidden, **_zero_pasts(dec_sess)},
    )
    return dict(zip(out_names, out, strict=True))["logits"][0]


def run_margin_analysis(model, zip_path, pairs, max_len: int = 256) -> MarginReport:
    """pairs: iterable of (source_text, gold_target) — the same probe set
    the CER parity gate uses. Teacher-forces both sides and measures the
    argmax flip rate, reference top1−top2 margin quantiles, and KL
    divergence. Complements ``run_parity``: CER measures what already
    broke, margins measure how close the rest is to breaking."""
    import numpy as np
    import yaml

    zip_path = Path(zip_path)
    enc, dec = _sessions_from_zip(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        precision = yaml.safe_load(zf.read("metadata.yaml"))["precision"]

    samples = 0
    kld_sum = 0.0
    margin_chunks: list = []
    flip_chunks: list = []
    for source, target in pairs:
        target_ids = encode_bytes(target)[:max_len]  # trailing EOS included
        if len(target_ids) < 2:
            continue
        samples += 1
        ref = _torch_forced_logits(model, source, target_ids)
        ref = ref.detach().numpy().astype("float64")
        got = np.asarray(_onnx_forced_logits(enc, dec, source, target_ids), dtype="float64")
        margins, flips, kld = _margin_stats(ref, got)
        margin_chunks.append(margins)
        flip_chunks.append(flips)
        kld_sum += float(kld.sum())

    margins = np.concatenate(margin_chunks)
    flips = np.concatenate(flip_chunks)
    p1, p10, p50 = (float(np.quantile(margins, q)) for q in (0.01, 0.10, 0.50))
    n_flips = int(flips.sum())
    return MarginReport(
        samples=samples,
        tokens=int(margins.size),
        flipped_tokens=n_flips,
        flip_rate=round(n_flips / max(margins.size, 1), 6),
        kld_mean=round(kld_sum / max(margins.size, 1), 8),
        margin_p1=round(p1, 4),
        margin_p10=round(p10, 4),
        margin_p50=round(p50, 4),
        flip_low_margin_share=round(
            float((margins[flips] < p10).mean()) if n_flips else 0.0, 4
        ),
        precision=precision,
    )


def write_margin_report(report: MarginReport, out_path: Path | str) -> Path:
    """Emit the margin analysis as JSON next to a release zip (diagnostic
    artifact; the release gate remains the CER parity block)."""
    import json
    from dataclasses import asdict

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )
    return out_path


def write_parity(zip_path: Path | str, report: ParityReport) -> Path:
    """Write the parity block into the zip's metadata and enforce strict
    validation. Raises if the gate does not pass."""
    from imf.validator import validate_zip

    zip_path = Path(zip_path)
    result = validate_zip(zip_path)
    if not result.ok or result.metadata is None:
        raise RuntimeError(f"cannot write parity into invalid zip: {result.errors}")
    metadata = result.metadata
    if not report.passed:
        limit = Parity.max_cer_delta(metadata.precision)
        raise RuntimeError(
            f"parity gate FAILED: cer_delta {report.cer_delta}pp over "
            f"{report.samples} samples (limits: <= {limit}pp for "
            f"{metadata.precision}, >= {Parity.MIN_SAMPLES} samples)"
        )

    import tempfile

    updated = ModelMetadata(
        format=metadata.format,
        id=metadata.id,
        task=metadata.task,
        source_script=metadata.source_script,
        target=metadata.target,
        tokenizer=metadata.tokenizer,
        opset=metadata.opset,
        decoder=metadata.decoder,
        precision=metadata.precision,
        license=metadata.license,
        trained_from=metadata.trained_from,
        metrics=metadata.metrics,
        parity=Parity(samples=report.samples, cer_delta=report.cer_delta),
        sha256=metadata.sha256,
    )

    import yaml

    from imf.pack import _to_dict

    # Same filesystem as the target: os.replace is atomic within one
    # filesystem and fails with EXDEV across a volume mount.
    with tempfile.TemporaryDirectory(dir=zip_path.parent) as tmp:
        rewritten = Path(tmp) / "rewritten.zip"
        with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(
            rewritten, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for name in src.namelist():
                if name == "metadata.yaml":
                    dst.writestr(
                        name,
                        yaml.safe_dump(_to_dict(updated), sort_keys=False, allow_unicode=True),
                    )
                else:
                    dst.writestr(name, src.read(name))
        rewritten.replace(zip_path)

    strict = validate_zip(zip_path, strict=True)
    if not strict.ok:
        raise RuntimeError(f"strict gate failed after parity write: {strict.errors}")
    return zip_path


def write_golden(zip_path: Path | str, inputs, out_path: Path | str, max_len: int = 256) -> Path:
    """Emit the cross-runtime golden set: fixed inputs + reference outputs
    from the ONNX graphs (Python is the reference implementation)."""


    zip_path = Path(zip_path)
    out_path = Path(out_path)
    enc, kv = _sessions_from_zip(zip_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for source in inputs:
            tokens = onnx_greedy_kv(enc, kv, source, max_len)
            fh.write(
                json.dumps(
                    {"input": source, "tokens": tokens, "output": _decode_tokens(tokens)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path
