"""Tests for ``framework.registry`` + ``framework.evaluator``."""

from __future__ import annotations

import pytest

from framework.evaluator import (
    BaseEvaluator,
    accuracy,
    char_error_rate,
    edit_distance,
    most_common_error_pairs,
    token_error_rate,
)
from framework.registry import (
    register_data_module,
    register_evaluator,
    resolve_data_module,
    resolve_evaluator,
    resolve_model_module,
)


def test_edit_distance_basic() -> None:
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("kitten", "sitting") == 3


def test_char_error_rate_zero_when_identical() -> None:
    assert char_error_rate("hello", "hello") == 0.0


def test_char_error_rate_full_when_gold_empty() -> None:
    assert char_error_rate("abc", "") == 1.0


def test_token_error_rate_phonemes() -> None:
    pred = ["a", "b", "c"]
    gold = ["a", "b", "c"]
    assert token_error_rate(pred, gold) == 0.0


def test_accuracy_perfect() -> None:
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0


def test_accuracy_partial() -> None:
    assert accuracy(["a", "x"], ["a", "b"]) == 0.5


def test_most_common_error_pairs() -> None:
    pred = ["ab", "ac"]
    gold = ["ab", "ab"]
    pairs = most_common_error_pairs(pred, gold)
    assert "b->c" in pairs


def test_register_and_resolve_data(tmp_path) -> None:
    @register_data_module("__test_data__")
    class _TestData:
        pass

    assert resolve_data_module("__test_data__") is _TestData

    # Cleanup by re-importing is awkward; just confirm duplicate raises
    with pytest.raises(ValueError):

        @register_data_module("__test_data__")
        class _Other:
            pass


def test_register_and_resolve_evaluator() -> None:
    @register_evaluator("__test_eval__")
    class _TestEval(BaseEvaluator):
        name = "__test_eval__"

        def compute_metric(self, predictions, gold) -> float:
            return 0.5

    resolved = resolve_evaluator("__test_eval__")
    assert resolved is _TestEval
    instance = resolved()
    metric = instance.evaluate(["x"], ["y"])
    assert metric.value == 0.5
    assert metric.metric == "__test_eval__"


def test_resolve_unknown_raises() -> None:
    with pytest.raises(KeyError):
        resolve_data_module("__definitely_not_registered__")
    with pytest.raises(KeyError):
        resolve_model_module("__definitely_not_registered__")
    with pytest.raises(KeyError):
        resolve_evaluator("__definitely_not_registered__")


def test_evaluator_length_mismatch_raises() -> None:
    class _E(BaseEvaluator):
        name = "x"

        def compute_metric(self, predictions, gold) -> float:
            return 0.0

    with pytest.raises(ValueError):
        _E().evaluate(["a"], ["a", "b"])
