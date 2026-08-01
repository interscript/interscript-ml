"""``tasks`` package marker.

Each subpackage (rababa_arabic, secryst_thai_ipa, etc.) registers its
classes via ``@register_data_module`` / ``@register_model_module`` /
``@register_evaluator`` decorators on import. The pipeline triggers
those imports dynamically via ``importlib.import_module``.
"""
