"""student_t5_config: one translation of spec vocabulary -> T5Config,
shared by the sequence and logit-KD paths (the logit path's raw
T5Config(**spec) silently took T5 defaults for depths and produced a
6/6 student from a 6/4 spec — the layer-copy then indexed out of
range)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

modal = pytest.importorskip("modal")


def test_layerdrop_spec_translates_depths() -> None:
    from gpu.modal_distill import student_t5_config

    cfg = student_t5_config(
        {
            "d_model": 1472,
            "d_kv": 64,
            "d_ff": 3584,
            "num_heads": 6,
            "enc_layers": 6,
            "dec_layers": 4,
            "feed_forward_proj": "gated-gelu",
        }
    )
    assert cfg.num_layers == 6
    assert cfg.num_decoder_layers == 4
    assert cfg.vocab_size == 259
    assert cfg.decoder_start_token_id == 0


def test_byte_model_defaults() -> None:
    from gpu.modal_distill import student_t5_config

    cfg = student_t5_config({})
    assert cfg.num_layers == 8 and cfg.num_decoder_layers == 8
    assert cfg.d_model == 384


def test_vocab_size_override_matches_teacher() -> None:
    # logit-KD computes teacher-vs-student KL: the student vocab must
    # match the teacher's (ByT5 = 384), while the sequence path's
    # byte-table default (259) stays intact
    from gpu.modal_distill import student_t5_config

    cfg = student_t5_config({"vocab_size": 384, "enc_layers": 6, "dec_layers": 4})
    assert cfg.vocab_size == 384
    assert student_t5_config({"enc_layers": 6, "dec_layers": 4}).vocab_size == 259
