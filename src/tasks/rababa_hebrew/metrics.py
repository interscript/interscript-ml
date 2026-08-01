"""DER evaluator for Hebrew (nikud error rate).

Same metric as Arabic DER, just re-registered under a Hebrew-specific
name so it appears in the registry alongside the data module.
"""

from framework.registry import register_evaluator
from tasks.rababa_arabic.metrics import DEREvaluator


@register_evaluator("der_hebrew")
class DEREvaluatorHebrew(DEREvaluator):
    """Same DER math, different name for clarity in task listings."""

    name = "der_hebrew"
