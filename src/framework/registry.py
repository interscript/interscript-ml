"""Typed registry for pluggable framework components.

Each task ``config.yaml`` names its data/model/evaluator modules by
string. The registry resolves those strings to classes. This is OCP:
registering a new data module does not require editing framework code.
"""

from __future__ import annotations

from typing import TypeVar

_TData = TypeVar("_TData")
_TModel = TypeVar("_TModel")
_TEval = TypeVar("_TEval")

_data_modules: dict[str, type] = {}
_model_modules: dict[str, type] = {}
_evaluators: dict[str, type] = {}


def register_data_module(name: str):
    """Decorator: register a ``DataModule`` subclass under ``name``."""

    def decorator(cls: type[_TData]) -> type[_TData]:
        if name in _data_modules:
            raise ValueError(f"Data module already registered: {name}")
        _data_modules[name] = cls
        return cls

    return decorator


def register_model_module(name: str):
    """Decorator: register a ``ModelModule`` subclass under ``name``."""

    def decorator(cls: type[_TModel]) -> type[_TModel]:
        if name in _model_modules:
            raise ValueError(f"Model module already registered: {name}")
        _model_modules[name] = cls
        return cls

    return decorator


def register_evaluator(name: str):
    """Decorator: register a ``BaseEvaluator`` subclass under ``name``."""

    def decorator(cls: type[_TEval]) -> type[_TEval]:
        if name in _evaluators:
            raise ValueError(f"Evaluator already registered: {name}")
        _evaluators[name] = cls
        return cls

    return decorator


def resolve_data_module(name: str) -> type:
    if name not in _data_modules:
        raise KeyError(
            f"Unknown data module '{name}'. Did you import the task package?"
        )
    return _data_modules[name]


def resolve_model_module(name: str) -> type:
    if name not in _model_modules:
        raise KeyError(
            f"Unknown model module '{name}'. Did you import the task package?"
        )
    return _model_modules[name]


def resolve_evaluator(name: str) -> type:
    if name not in _evaluators:
        raise KeyError(
            f"Unknown evaluator '{name}'. Did you import the task package?"
        )
    return _evaluators[name]


def _reset_registry() -> None:
    """Test-only: clear all registrations."""
    _data_modules.clear()
    _model_modules.clear()
    _evaluators.clear()
