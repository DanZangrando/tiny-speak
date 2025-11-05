"""Config helpers for TinySpeak training pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def get_repo_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parent.parent


def load_master_dataset_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load the shared master dataset configuration.

    Parameters
    ----------
    config_path:
        Optional override path for the configuration file. Defaults to
        ``<repo_root>/master_dataset_config.json``.
    """
    if config_path is None:
        config_path = get_repo_root() / "master_dataset_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Master dataset configuration not found at {config_path}. "
            "Generate datasets from the Streamlit dataset managers first."
        )

    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
