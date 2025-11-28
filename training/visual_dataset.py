"""Dataset helpers for the TinySpeak visual pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import get_repo_root, load_master_dataset_config
from utils import get_language_letters
import numpy as np


Split = Literal["train", "val", "test"]


@dataclass
class VisualSample:
    """Container describing a single visual dataset entry."""

    path: Path
    label: int
    letter: str
    metadata: Dict


DEFAULT_SPLIT_RATIOS: Dict[Split, float] = {"train": 0.7, "val": 0.15, "test": 0.15}


def _stratified_split(
    samples: Dict[str, List[Dict]],
    *,
    split_ratios: Dict[Split, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
) -> Dict[Split, List[Tuple[str, Dict]]]:
    """Create stratified train/val/test splits for each letter."""

    rng = random.Random(seed)
    splits: Dict[Split, List[Tuple[str, Dict]]] = {"train": [], "val": [], "test": []}

    for letter, entries in samples.items():
        valid_entries = [entry for entry in entries if entry.get("file_path")]
        if not valid_entries:
            continue

        rng.shuffle(valid_entries)
        n_total = len(valid_entries)

        # Compute split sizes while ensuring the sum equals ``n_total``.
        train_size = int(round(n_total * split_ratios["train"]))
        val_size = int(round(n_total * split_ratios["val"]))
        # Ensure at least one item goes to the train split when possible.
        if train_size == 0 and n_total > 0:
            train_size = 1

        # Adjust sizes to avoid exceeding the total count.
        if train_size + val_size > n_total:
            val_size = max(0, n_total - train_size)

        test_size = max(0, n_total - train_size - val_size)

        # Guard against empty splits when total samples are scarce.
        if train_size == 0 and n_total > 0:
            train_size = 1
        if train_size + val_size == n_total and test_size == 0 and n_total >= 2:
            val_size = max(1, val_size)
            train_size = max(1, n_total - val_size)
        test_size = max(0, n_total - train_size - val_size)

        train_items = valid_entries[:train_size]
        val_items = valid_entries[train_size : train_size + val_size]
        test_items = valid_entries[train_size + val_size :]

        splits["train"].extend((letter, item) for item in train_items)
        splits["val"].extend((letter, item) for item in val_items)
        splits["test"].extend((letter, item) for item in test_items)

    return splits


class VisualLetterDataset(Dataset[Dict[str, torch.Tensor]]):
    """Torch dataset backed by the generated visual dataset configuration."""

    def __init__(
        self,
        *,
        split: Split = "train",
        transform: transforms.Compose | None = None,
        augment: bool = False,
        split_ratios: Dict[Split, float] | None = None,
        seed: int = 42,
        whitelist_chars: List[str] | None = None,
        target_language: str | None = None,
    ) -> None:
        config = load_master_dataset_config()
        visual_cfg = config.get("visual_dataset", {})
        generated = visual_cfg.get("generated_images", {})
        
        if whitelist_chars:
            generated = {k: v for k, v in generated.items() if k in whitelist_chars}
            
        # Filtrar por idioma si se especifica
        if target_language:
            # 1. Obtener letras válidas para el idioma
            allowed_letters = set(get_language_letters(target_language))
            
            filtered_generated = {}
            for letter, entries in generated.items():
                # Si la letra no es del idioma, saltar
                if letter.lower() not in allowed_letters:
                    continue
                
                # 2. Filtrar imágenes: Aceptar idioma específico, 'dataset' (genérico) o None (legacy)
                valid_entries = [
                    e for e in entries 
                    if e.get('language') == target_language or 
                       e.get('language') == 'dataset' or
                       e.get('language') is None or
                       (e.get('metadata', {}).get('language') == target_language)
                ]
                if valid_entries:
                    filtered_generated[letter] = valid_entries
            generated = filtered_generated
            
        if not generated:
            msg = "El dataset visual no contiene imágenes generadas"
            if target_language:
                msg += f" para el idioma '{target_language}'"
            raise ValueError(msg + ".")

        self.repo_root = get_repo_root()
        self.letters: List[str] = sorted(generated.keys())
        self.letter_to_index = {letter: idx for idx, letter in enumerate(self.letters)}

        ratios = split_ratios or DEFAULT_SPLIT_RATIOS
        split_map = _stratified_split(generated, split_ratios=ratios, seed=seed)

        if split not in split_map:
            raise ValueError(f"Split desconocido '{split}'. Usa train, val o test.")

        self.samples: List[VisualSample] = []
        for letter, metadata in split_map[split]:
            rel_path = metadata.get("file_path")
            if not rel_path:
                continue
            abs_path = (self.repo_root / rel_path).resolve()
            if not abs_path.exists():
                # Intentar también path relativo a dataset base para robustez.
                fallback = self.repo_root / "data" / "visual" / rel_path
                abs_path = fallback.resolve()
            if not abs_path.exists():
                continue
            self.samples.append(
                VisualSample(
                    path=abs_path,
                    letter=letter,
                    label=self.letter_to_index[letter],
                    metadata=metadata,
                )
            )

        if not self.samples:
            raise ValueError(
                f"No hay muestras válidas para el split '{split}'. "
                "Genera más imágenes con Visual Dataset Manager."
            )

        base_transforms: List[Callable] = [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
        if augment and split == "train":
            base_transforms.insert(0, transforms.RandomApply([
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ], p=0.5))

        if transform is None:
            self.transform = transforms.Compose(base_transforms)
        else:
            self.transform = transform

    def __len__(self) -> int:  # noqa: D401 - standard Dataset API
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return len(self.letters)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:  # noqa: D401
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        
        # Convert metadata floats to float32 for MPS compatibility
        metadata = _convert_floats_to_float32(sample.metadata)
        
        return {
            "image": tensor,
            "label": sample.label,
            "letter": sample.letter,
            "metadata": metadata,
        }


def _convert_floats_to_float32(obj):
    """Recursively convert Python floats to float32."""
    if isinstance(obj, float):
        # We need to return numpy float32 to force float32 tensor.
        return np.float32(obj)
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_float32(v) for v in obj]
    return obj


def build_visual_dataloaders(
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    split_ratios: Dict[Split, float] | None = None,
    whitelist_chars: List[str] | None = None,
    target_language: str | None = None,
) -> Tuple[VisualLetterDataset, VisualLetterDataset, VisualLetterDataset, Dict[Split, DataLoader]]:
    """Create stratified dataloaders for train/val/test splits."""

    train_ds = VisualLetterDataset(split="train", augment=True, seed=seed, split_ratios=split_ratios, whitelist_chars=whitelist_chars, target_language=target_language)
    val_ds = VisualLetterDataset(split="val", augment=False, seed=seed, split_ratios=split_ratios, whitelist_chars=whitelist_chars, target_language=target_language)
    test_ds = VisualLetterDataset(split="test", augment=False, seed=seed, split_ratios=split_ratios, whitelist_chars=whitelist_chars, target_language=target_language)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return train_ds, val_ds, test_ds, loaders
