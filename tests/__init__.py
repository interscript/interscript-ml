"""Tests for the framework layer.

These run without torch or onnxruntime — they exercise the pure-Python
abstractions (config, registry, data, evaluator, pipeline). Heavy
training tests live in ``test_train.py`` under ``@pytest.mark.gpu``.
"""
