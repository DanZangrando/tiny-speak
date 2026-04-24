"""
utils/checkpoints.py — Gestión de checkpoints de modelos entrenados.

La estructura de directorios esperada es:

    data/checkpoints/
        tiny_ears_phonemes/   ← PhonologicalPathway entrenado sobre fonemas
        tiny_ears_words/      ← PhonologicalPathway entrenado sobre palabras
        tiny_eyes/            ← VisualPathway
        tiny_speller/         ← TinySpeller    (Stage 1)
        tiny_reader/          ← TinyReaderP2W  (Stage 2 / end-to-end)

Para compatibilidad con la estructura anterior, también se busca en:
    models/listener/
    models/listener_checkpoints/
    models/recognizer/
    models/recognizer_checkpoints/
    models/reader/
    models/reader_checkpoints/
    experiments/models/
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Estructura canónica de directorios
# ---------------------------------------------------------------------------

#: Mapeo de tipo de modelo → carpetas donde buscar (en orden de preferencia)
_SEARCH_DIRS: dict[str, list[str]] = {
    "tiny_ears_phonemes": [
        "data/checkpoints/tiny_ears_phonemes",
        "models/listener",
        "models/listener_checkpoints",
    ],
    "tiny_ears_words": [
        "data/checkpoints/tiny_ears_words",
        "models/listener",
        "models/listener_checkpoints",
    ],
    "tiny_eyes": [
        "data/checkpoints/tiny_eyes",
        "models/recognizer",
        "models/recognizer_checkpoints",
    ],
    "tiny_speller": [
        "data/checkpoints/tiny_speller",
        "models/reader",
        "models/reader_checkpoints",
    ],
    "tiny_reader": [
        "data/checkpoints/tiny_reader",
        "models/reader",
        "models/reader_checkpoints",
    ],
    # Nombres legacy (usados por páginas antiguas)
    "listener": [
        "data/checkpoints/tiny_ears_phonemes",
        "data/checkpoints/tiny_ears_words",
        "models/listener",
        "models/listener_checkpoints",
    ],
    "recognizer": [
        "data/checkpoints/tiny_eyes",
        "models/recognizer",
        "models/recognizer_checkpoints",
    ],
    "reader": [
        "data/checkpoints/tiny_speller",
        "data/checkpoints/tiny_reader",
        "models/reader",
        "models/reader_checkpoints",
    ],
}


def _repo_root() -> Path:
    """Retorna la raíz del repositorio."""
    return Path(__file__).resolve().parent.parent


def list_checkpoints(model_type: str) -> list[dict[str, Any]]:
    """Lista los checkpoints disponibles para un tipo de modelo.

    Args:
        model_type: uno de los tipos definidos en ``_SEARCH_DIRS``.

    Returns:
        Lista de dicts con keys: path, filename, timestamp, meta.
        Ordenada de más reciente a más antiguo.
    """
    repo = _repo_root()
    dirs_to_search = [repo / d for d in _SEARCH_DIRS.get(model_type, [])]

    checkpoints: list[dict[str, Any]] = []
    seen: set[str] = set()

    for directory in dirs_to_search:
        if not directory.exists():
            continue
        for ckpt_path in directory.rglob("*.ckpt"):
            key = str(ckpt_path.resolve())
            if key in seen:
                continue
            seen.add(key)

            meta: dict[str, Any] = {}
            meta_path = ckpt_path.with_suffix(".ckpt.meta.json")
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            timestamp = meta.get("timestamp", ckpt_path.stat().st_mtime)
            checkpoints.append(
                {
                    "path": str(ckpt_path),
                    "filename": ckpt_path.name,
                    "timestamp": timestamp,
                    "meta": meta,
                }
            )

    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    return checkpoints


def save_model_metadata(ckpt_path: str | Path, config: dict, metrics: dict, history: list = None) -> None:
    """Guarda un archivo .ckpt.meta.json junto al checkpoint."""
    meta_path = Path(ckpt_path).with_suffix(".ckpt.meta.json")
    data = {
        "timestamp": time.time(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": metrics,
    }
    if history is not None:
        data["history"] = history
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def delete_run_artifacts(model_type: str, metrics_filename: str) -> bool:
    """Elimina el archivo de métricas y el checkpoint asociado si existe."""
    repo = _repo_root()
    dirs = [repo / d for d in _SEARCH_DIRS.get(model_type, [])]

    deleted = False
    for directory in dirs:
        json_path = directory / metrics_filename
        if json_path.exists():
            json_path.unlink()
            deleted = True

            ckpt_name = metrics_filename.replace(".json", ".ckpt")
            ckpt_path = directory / ckpt_name
            if ckpt_path.exists():
                ckpt_path.unlink()
                meta = ckpt_path.with_suffix(".ckpt.meta.json")
                if meta.exists():
                    meta.unlink()
            break

    return deleted


def save_training_metrics(model_type: str, name: str, data: dict) -> str:
    """Guarda métricas de entrenamiento en JSON dentro de metrics/<model_type>/."""
    repo = _repo_root()
    metrics_dir = repo / "metrics" / model_type
    metrics_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c for c in name if c.isalnum() or c in " -_").rstrip()
    file_path = metrics_dir / f"{safe_name}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(file_path)


def load_training_metrics(model_type: str, filename: str) -> dict:
    """Carga métricas de entrenamiento desde metrics/<model_type>/<filename>."""
    repo = _repo_root()
    file_path = repo / "metrics" / model_type / filename
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_metrics_files(model_type: str) -> list[dict[str, Any]]:
    """Lista archivos de métricas disponibles para un tipo de modelo."""
    repo = _repo_root()
    metrics_dir = repo / "metrics" / model_type
    if not metrics_dir.exists():
        return []

    files: list[dict[str, Any]] = []
    for json_path in metrics_dir.glob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            files.append(
                {
                    "filename": json_path.name,
                    "path": str(json_path),
                    "timestamp": data.get("timestamp", json_path.stat().st_mtime),
                    "config": data.get("config", {}),
                }
            )
        except Exception:
            continue

    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return files
