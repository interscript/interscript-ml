"""ONNX export of byte-level (ByT5/T5) checkpoints into IMF v1 zips.

Generalizes secryst PR #44's scripts/export_onnx_byt5.py:

- encoder.onnx        input_ids -> last_hidden_state
- decoder.onnx        input_ids, encoder_hidden_states -> logits (fallback)
- decoder-kv.onnx     + past_* inputs / present_* outputs (default artifact)
- opset pinned to 14 (the Ruby onnxruntime gem's bundled ORT is old)
- fp16 (keep IO fp32) and int8 (dynamic quantization) variants
- fixture mode: a tiny random T5 for CI, via ``make_fixture_checkpoint``

The KV graph caches self-attention only (``past_key_i`` / ``past_value_i``
inputs, ``present_key_i`` / ``present_value_i`` outputs). Cross-attention
K/V are a deterministic projection of ``encoder_hidden_states``, so they
are recomputed each step instead of being cached — this keeps the
attention-mask length consistent for any past length and spares runtimes
all cross-cache bookkeeping. Step 0 feeds zero-length self pasts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from imf.pack import pack_zip
from imf.schema import ModelMetadata

OPSET = 14
GRAPH_NAMES = ("encoder.onnx", "decoder.onnx", "decoder-kv.onnx")

# Canonical ByT5 byte table (google/byt5): pad=0, eos=1, unk=2, and
# every UTF-8 byte b is token id b + 3, up to 258. The vocab is 384-wide;
# ids > 258 are unused in practice. "tokenizer: bytes" in IMF metadata
# means THIS fixed table — no vocab files, but ids are NOT raw byte values
# (feeding text.bytes directly silently produces garbage).
BYTE_OFFSET = 3
PAD_ID = 0
EOS_ID = 1


def encode_bytes(text: str) -> list[int]:
    """Canonical byte-level tokenization: byte ids + trailing EOS."""
    return [b + BYTE_OFFSET for b in text.encode("utf-8")] + [EOS_ID]


def load_byte_seq2seq(checkpoint_dir: Path | str):
    """Load a T5-family checkpoint in eager attention (export-safe)."""
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_dir, attn_implementation="eager"
    ).eval()
    return model


def make_fixture_checkpoint(out_dir: Path | str, seed: int = 42) -> Path:
    """Tiny random T5 with a byte-sized vocab, for CI and --fixture runs."""
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = T5Config(
        vocab_size=384,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        decoder_start_token_id=0,
        feed_forward_proj="relu",
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = T5ForConditionalGeneration(config).eval()
    model.save_pretrained(out_dir)
    return out_dir


def _decoder_plain(model):
    import torch.nn as nn

    class DecoderPlain(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.get_decoder()
            self.lm_head = model.lm_head
            self.scale = model.config.d_model ** -0.5

        def forward(self, input_ids, encoder_hidden_states):
            hidden = self.decoder(
                input_ids=input_ids, encoder_hidden_states=encoder_hidden_states
            )[0]
            return self.lm_head(hidden * self.scale)

    return DecoderPlain(model)


def _decoder_kv(model):
    import torch.nn as nn
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    num_layers = model.config.num_decoder_layers or model.config.num_layers

    class DecoderKV(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.get_decoder()
            self.lm_head = model.lm_head
            self.scale = model.config.d_model ** -0.5

        def forward(self, input_ids, encoder_hidden_states, *pasts):
            # is_updated must stay unset: EncoderDecoderCache.__init__ bakes
            # it as a Python bool from cross-cache lengths, and a True would
            # trace the "reuse cross K/V" branch, which mismatches the
            # attention mask for any fed cross-past length. Cross K/V are a
            # deterministic projection of encoder_hidden_states anyway, so
            # the graph recomputes them each step and never caches them.
            cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
            for i in range(num_layers):
                cache.self_attention_cache.update(pasts[2 * i], pasts[2 * i + 1], i)
            hidden = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=cache,
                use_cache=True,
            )[0]
            logits = self.lm_head(hidden * self.scale)
            outputs = [logits]
            for i in range(num_layers):
                outputs.append(cache.self_attention_cache.layers[i].keys)
                outputs.append(cache.self_attention_cache.layers[i].values)
            return tuple(outputs)

    return DecoderKV(model)


def _kv_io_names(num_layers: int) -> tuple[list[str], list[str]]:
    inputs, outputs = [], ["logits"]
    for i in range(num_layers):
        inputs += [f"past_key_{i}", f"past_value_{i}"]
        outputs += [f"present_key_{i}", f"present_value_{i}"]
    return inputs, outputs


def _sample_pasts(model, encoder_hidden_states):
    """Harvest one decoder step's self-attention cache for shape examples."""
    import torch

    out = model.get_decoder()(
        input_ids=torch.tensor([[0]]),
        encoder_hidden_states=encoder_hidden_states,
        use_cache=True,
    )
    pasts = []
    for layer in out[1].self_attention_cache.layers:
        pasts += [layer.keys.clone(), layer.values.clone()]
    return pasts


def export_graphs(model, out_dir: Path | str) -> dict[str, Path]:
    """Export encoder + plain decoder + KV decoder as fp32 ONNX, opset 14.

    Traces a deepcopy: TorchScript tracing leaves the traced module's
    runtime behavior subtly altered (observed empirically), and callers
    (e.g. the WO03 parity harness) need the reference model pristine.
    """
    import copy

    import torch

    model = copy.deepcopy(model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = torch.tensor([[104, 101]])  # "he"
    with torch.no_grad():
        hidden = model.get_encoder()(input_ids=ids)[0]

    paths = {}

    torch.onnx.export(
        model.get_encoder(),
        (ids,),
        out_dir / "encoder.onnx",
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        opset_version=OPSET,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
    )
    paths["encoder.onnx"] = out_dir / "encoder.onnx"

    torch.onnx.export(
        _decoder_plain(model),
        (torch.tensor([[0]]), hidden),
        out_dir / "decoder.onnx",
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "encoder_hidden_states": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
    )
    paths["decoder.onnx"] = out_dir / "decoder.onnx"

    num_layers = model.config.num_decoder_layers or model.config.num_layers
    pasts = _sample_pasts(model, hidden)
    kv_inputs = ["input_ids", "encoder_hidden_states"] + _kv_io_names(num_layers)[0]
    kv_outputs = _kv_io_names(num_layers)[1]
    dynamic = {
        "input_ids": {0: "batch", 1: "cur_seq"},
        "encoder_hidden_states": {0: "batch", 1: "enc_seq"},
        "logits": {0: "batch", 1: "cur_seq"},
    }
    for name in kv_inputs[2:]:
        dynamic[name] = {0: "batch", 2: "past_seq"}
    for name in kv_outputs[1:]:
        dynamic[name] = {0: "batch", 2: "present_seq"}

    with torch.no_grad():
        torch.onnx.export(
            _decoder_kv(model),
            (torch.tensor([[0]]), hidden, *pasts),
            out_dir / "decoder-kv.onnx",
            input_names=kv_inputs,
            output_names=kv_outputs,
            opset_version=OPSET,
            dynamo=False,
            dynamic_axes=dynamic,
        )
    paths["decoder-kv.onnx"] = out_dir / "decoder-kv.onnx"
    return paths


def convert_fp16(model):
    """A fp16 copy of the model for torch-native half export.

    The onnxruntime float16 CONVERTER is not usable here: on real ByT5
    checkpoints it produces all-zero encoder hiddens (found 2026-08-16,
    khm-latn — 1939pp CER); exporting the torch model under .half() is
    exact on gold pairs. Graph IO becomes float16 (input_ids stay int64);
    runtimes read dtypes from the session, and _zero_pasts follows them.
    """
    import copy

    return copy.deepcopy(model).half()


def quantize_int8(src: Path | str, dst: Path | str) -> Path:
    """fp32 -> dynamically quantized int8 (MatMul weights QInt8).

    MatMul-only: quantizing other ops inserts precision casts that break
    ORT's session-time SimplifiedLayerNormFusion, and preprocessing the
    graph (quant_pre_process) pins concrete example shapes into
    DynamicQuantizeLinear buffers.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    return Path(dst)


def quantize_int4(src: Path | str, dst: Path | str, block_size: int = 64) -> Path:
    """fp32 -> 4-bit blockwise MatMul (MatMulNBits, com.microsoft domain).

    Halves int8 again at some quality cost; meant for the browser/edge
    client tier. NOTE: old runtimes (the Ruby gem's bundled ORT) cannot
    execute MatMulNBits — int4 zips are client-crystal territory and
    their loaders should fail loudly rather than silently fall back.
    """
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
    )

    model = onnx.load(str(src))
    quant = MatMulNBitsQuantizer(
        model=model, block_size=block_size, is_symmetric=True
    )
    quant.process()
    quant.model.save_model_to_file(str(dst))
    return Path(dst)


def onnx_greedy_plain(encoder_sess, decoder_sess, text: str, max_len: int = 256) -> list[int]:
    """Greedy decode over ONNX sessions (plain decoder). Self-check helper.

    Returns generated token ids (the vocab is 384-wide; only a trained
    byte-level model reliably stays < 256)."""
    import numpy as np

    ids = np.array([encode_bytes(text)], dtype=np.int64)
    if ids.shape[1] == 1:
        return []
    hidden = encoder_sess.run(None, {"input_ids": ids})[0]
    dec_ids = np.array([[PAD_ID]], dtype=np.int64)
    generated: list[int] = []
    for _ in range(max_len):
        logits = decoder_sess.run(
            None, {"input_ids": dec_ids, "encoder_hidden_states": hidden}
        )[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == EOS_ID:
            break
        generated.append(nxt)
        dec_ids = np.concatenate([dec_ids, np.array([[nxt]], dtype=np.int64)], axis=1)
    return generated


def _zero_pasts(kv_sess) -> dict[str, object]:
    """Zero-length past inputs for step 0, shapes from session metadata."""
    import numpy as np

    pasts = {}
    for meta in kv_sess.get_inputs():
        if not meta.name.startswith("past_"):
            continue
        shape = meta.shape  # [batch, heads, past_seq, d_kv] with str dynamic dims
        heads = shape[1] if isinstance(shape[1], int) else 4
        d_kv = shape[3] if isinstance(shape[3], int) else 8
        dtype = np.float16 if meta.type == "tensor(float16)" else np.float32
        pasts[meta.name] = np.zeros((1, heads, 0, d_kv), dtype=dtype)
    return pasts


def onnx_greedy_kv(encoder_sess, kv_sess, text: str, max_len: int = 256) -> list[int]:
    """Greedy decode over ONNX sessions (KV decoder). Self-check helper."""
    import numpy as np

    ids = np.array([encode_bytes(text)], dtype=np.int64)
    if ids.shape[1] == 1:
        return []
    hidden = encoder_sess.run(None, {"input_ids": ids})[0]
    out_names = [o.name for o in kv_sess.get_outputs()]
    pasts = _zero_pasts(kv_sess)
    cur = np.array([[0]], dtype=np.int64)
    generated: list[int] = []
    for _ in range(max_len):
        out = kv_sess.run(None, {"input_ids": cur, "encoder_hidden_states": hidden, **pasts})
        results = dict(zip(out_names, out, strict=True))
        nxt = int(np.argmax(results["logits"][0, -1]))
        if nxt == EOS_ID:
            break
        generated.append(nxt)
        pasts = {
            name: results[name.replace("past_", "present_", 1)]
            for name in pasts
        }
        cur = np.array([[nxt]], dtype=np.int64)
    return generated


def export_zips(
    model,
    metadata_path: Path | str,
    readme: str,
    out_dir: Path | str,
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8"),
) -> list[Path]:
    """Full pipeline: loaded model -> fp32 graphs -> precision variants -> IMF zips."""
    import tempfile

    metadata = ModelMetadata.from_yaml(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata.decoder != "kv":
        raise ValueError("WO02 exports declare decoder: kv (plain is the fallback)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        graphs = export_graphs(model, tmp / "graphs")
        graphs_16 = (
            export_graphs(convert_fp16(model), tmp / "graphs-fp16")
            if "fp16" in precisions
            else {}
        )

        for precision in precisions:
            variant_dir = tmp / precision
            variant_dir.mkdir()
            sources = graphs if precision != "fp16" else graphs_16
            for name, src in sources.items():
                dst = variant_dir / name
                if precision == "fp32" or precision == "fp16":
                    dst.write_bytes(src.read_bytes())
                elif precision == "int8":
                    quantize_int8(graphs[name], dst)
                elif precision == "int4":
                    quantize_int4(graphs[name], dst)
                else:
                    raise ValueError(f"unknown precision {precision!r}")
            meta = replace(metadata, precision=precision)
            zips.append(
                pack_zip(
                    variant_dir,
                    meta,
                    readme,
                    out_dir / f"{metadata.id}-{precision}.zip",
                )
            )
    return zips

