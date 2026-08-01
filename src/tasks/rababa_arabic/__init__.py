"""Rababa Arabic task package.

Importing this package registers the data module, model module, and
evaluator with the framework registry. The pipeline resolves them by
name from ``config.yaml``.
"""

from tasks.rababa_arabic.data import RababaArabicData
from tasks.rababa_arabic.metrics import DEREvaluator
from tasks.rababa_arabic.student import RababaStudent

__all__ = ["RababaArabicData", "DEREvaluator", "RababaStudent"]
