"""
utils/serialization.py — Carga y guardado seguro de configuraciones JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Convierte tipos numpy a tipos nativos de Python para serialización JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_dataset_config(config_path: str | Path) -> dict | None:
    """Carga un archivo JSON de configuración de manera segura.

    Retorna None si el archivo no existe o no se puede parsear.
    """
    try:
        p = Path(config_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_dataset_config(config: dict, config_path: str | Path) -> bool:
    """Guarda un dict como JSON convirtiendo tipos numpy.

    Retorna True si tuvo éxito, False en caso contrario.
    """
    try:
        clean = convert_numpy_types(config)
        with Path(config_path).open("w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"Error guardando {config_path}: {exc}")
        return False
