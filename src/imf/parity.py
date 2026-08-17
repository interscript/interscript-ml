"""WO03 parity gate: ONNX greedy vs the torch reference, CER delta <= 0.2pp.

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
from imf.export import BYTE_OFFSET, EOS_ID, PAD_ID, encode_bytes, onnx_greedy_kv
from imf.schema import ModelMetadata, Parity


@dataclass(frozen=True)
class ParityReport:
    samples: int
    cer_reference: float  # percentage points
    cer_onnx: float
    cer_delta: float
    token_mismatches: int

    @property
    def passed(self) -> bool:
        return (
            self.cer_delta <= Parity.MAX_CER_DELTA
            and self.samples >= Parity.MIN_SAMPLES
        )


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


def reference_decode(
    model, sources, max_len: int = 256, batch_size: int = 32
) -> list[list[int]]:
    """Torch-reference greedy decode of many inputs, computed once and
    shared across precision variants by run_parity.

    Batched: single-sequence semantics are preserved exactly — rows are
    padded on the encoder side with attention masks (T5 position bias is
    relative, so padded batch == unpadded singles), every row is fed one
    token per decoder step (no decoder padding), and each row is sliced
    at its first EOS. Verified token-identical against the sequential
    path on real checkpoints.
    """
    import torch

    results: list[list[int]] = []
    for start in range(0, len(sources), batch_size):
        batch = [encode_bytes(text) for text in sources[start : start + batch_size]]
        width = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
        attention = torch.zeros(len(batch), width, dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, : len(ids)] = 1
        with torch.no_grad():
            enc = model.get_encoder()(
                input_ids=input_ids, attention_mask=attention
            )[0]
            dec_ids = torch.full((len(batch), 1), PAD_ID, dtype=torch.long)
            step_tokens: list[list[int]] = [[] for _ in batch]
            finished_at: list[int | None] = [None] * len(batch)
            for step in range(max_len):
                hidden = model.get_decoder()(
                    input_ids=dec_ids,
                    encoder_hidden_states=enc,
                    encoder_attention_mask=attention,
                )[0]
                logits = model.lm_head(hidden * (model.config.d_model ** -0.5))
                tokens = logits[:, -1, :].argmax(dim=-1)
                all_done = True
                for row in range(len(batch)):
                    token = int(tokens[row])
                    if finished_at[row] is None:
                        if token == EOS_ID:
                            finished_at[row] = step
                        else:
                            step_tokens[row].append(token)
                        if finished_at[row] is None:
                            all_done = False
                    if finished_at[row] is None:
                        all_done = False
                if all_done:
                    break
                dec_ids = torch.cat([dec_ids, tokens.unsqueeze(1)], dim=1)
        results.extend(step_tokens)
    return results


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
    )


def write_parity(zip_path: Path | str, report: ParityReport) -> Path:
    """Write the parity block into the zip's metadata and enforce strict
    validation. Raises if the gate does not pass."""
    from imf.validator import validate_zip

    zip_path = Path(zip_path)
    result = validate_zip(zip_path)
    if not result.ok or result.metadata is None:
        raise RuntimeError(f"cannot write parity into invalid zip: {result.errors}")
    if not report.passed:
        raise RuntimeError(
            f"parity gate FAILED: cer_delta {report.cer_delta}pp over "
            f"{report.samples} samples (limits: <= {Parity.MAX_CER_DELTA}pp, "
            f">= {Parity.MIN_SAMPLES} samples)"
        )

    import tempfile

    metadata = result.metadata
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
