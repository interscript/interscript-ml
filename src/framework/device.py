"""Device detection. Single source of truth for hardware choice.

Keeps the rest of the framework device-agnostic: callers ask
``resolve_device(config)`` and get back a torch device string. No
scattered ``torch.device("cuda" if ...)`` calls elsewhere.
"""

from __future__ import annotations

import os


def detect_best_device() -> str:
    """Pick the fastest available device.

    Order: CUDA > MPS (Apple Silicon) > CPU.
    Detection is environment-aware; safe to call when torch isn't
    installed (returns "cpu").
    """
    forced = os.environ.get("INTERSCRIPT_DEVICE")
    if forced in {"cpu", "cuda", "mps"}:
        return forced
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str) -> str:
    """Resolve a config ``device`` field to a concrete device string.

    - ``"auto"`` → :func:`detect_best_device`
    - ``"cpu"`` / ``"cuda"`` / ``"mps"`` → as-is (after availability check)
    """
    if requested == "auto":
        return detect_best_device()
    if requested == "cuda":
        try:
            import torch  # type: ignore
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        try:
            import torch  # type: ignore
        except ImportError:
            return "cpu"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return "cpu"


def device_label(device: str) -> str:
    """Human-friendly label for logging."""
    labels = {"cpu": "CPU", "cuda": "CUDA GPU", "mps": "Apple MPS"}
    return labels.get(device, device)
