"""Rababa Hebrew task package.

Same architecture as rababa_arabic but a different alphabet + nikud
point set. Registers its own data module, model module, and reuses
the shared DER evaluator (since DER is script-agnostic).
"""

from tasks.rababa_hebrew.data import RababaHebrewData
from tasks.rababa_hebrew.metrics import DEREvaluatorHebrew
from tasks.rababa_hebrew.student import RababaHebrewStudent

__all__ = ["RababaHebrewData", "DEREvaluatorHebrew", "RababaHebrewStudent"]
