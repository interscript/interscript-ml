"""Index-driven model-id → volume-filename resolution (the harness
contract: model ids resolve through models.yaml, never hardcoded
naming conventions)."""

from pathlib import Path

import yaml

from src.api.model_resolution import resolve_zip_filename

INDEX = yaml.safe_load(Path("models.yaml").read_text())["models"]


def test_exact_index_filename_wins():
    volume = ["ara-diac-small-1.0-fp32.zip", "ara-diac-small-1.0-int8.zip"]
    assert resolve_zip_filename("ara-diac-small-1.0-int8", INDEX, volume) == (
        "ara-diac-small-1.0-int8.zip"
    )


def test_precision_fallback_when_volume_name_differs_from_index():
    # heb-diac-1.1 ships as heb.zip on GH Releases but the volume copy
    # landed as heb-diac-1.1-fp32.zip
    volume = ["heb-diac-1.1-fp32.zip", "heb-diac-1.1-fp16.zip"]
    assert resolve_zip_filename("heb-diac-1.1", INDEX, volume) == "heb-diac-1.1-fp32.zip"


def test_int4_variant_resolves():
    volume = ["tha-g2p-small-1.0-int8.zip", "tha-g2p-small-1.0-int4.zip"]
    assert resolve_zip_filename("tha-g2p-small-1.0-int4", INDEX, volume) == (
        "tha-g2p-small-1.0-int4.zip"
    )


def test_fp32_convention_fallback():
    volume = ["ara-diac-1.0-fp32.zip"]
    assert resolve_zip_filename("ara-diac-1.0", INDEX, volume) == "ara-diac-1.0-fp32.zip"


def test_unknown_id_raises_keyerror():
    import pytest

    with pytest.raises(KeyError):
        resolve_zip_filename("nope-1.0", INDEX, ["a-fp32.zip"])
