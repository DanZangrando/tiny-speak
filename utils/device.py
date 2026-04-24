"""
utils/device.py — Detección del dispositivo de cómputo disponible.
"""

from __future__ import annotations

import torch


def encontrar_device() -> torch.device:
    """Retorna el mejor dispositivo disponible: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
