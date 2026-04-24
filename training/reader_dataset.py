"""
training/reader_dataset.py — Dataset que asocia audio con secuencias de imágenes de grafemas.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from training.audio_dataset import build_audio_datasets
from training.visual_dataset import build_visual_dataloaders, VisualLetterDataset
from training.config import load_master_dataset_config
from utils.graphemes import tokenize_graphemes, get_language_letters, get_phoneme_inventory


class ReaderDataset(Dataset):
    """
    Dataset que asocia una muestra de audio (palabra) con su deletreo visual (secuencia de imágenes).
    """

    def __init__(
        self,
        audio_dataset: Dataset,
        visual_config: Dict[str, Any],
        target_language: str | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.audio_dataset = audio_dataset
        self.visual_config = visual_config
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # Filtrar grafemas disponibles según el idioma
        all_graphemes = list(visual_config.keys())
        if target_language:
            allowed = set(get_language_letters(target_language))
            self.available_graphemes = [g for g in all_graphemes if g.lower() in allowed]
        else:
            self.available_graphemes = all_graphemes
            
        self.repo_root = Path(__file__).parent.parent
        self.class_names = getattr(audio_dataset, "class_names", [])

    def __len__(self) -> int:
        return len(self.audio_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Obtener audio y palabra del audio_dataset
        audio_item = self.audio_dataset[index]
        word = audio_item["word"]
        
        # Tokenizar palabra en grafemas
        tokens = tokenize_graphemes(word, self.available_graphemes)
        
        # Obtener secuencia de imágenes
        images = []
        for token in tokens:
            entries = self.visual_config.get(token, [])
            if not entries:
                entries = self.visual_config.get(token.lower(), [])
            
            if entries:
                entry = random.choice(entries)
                rel_path = entry.get("file_path")
                full_path = self.repo_root / rel_path
                try:
                    img = Image.open(full_path).convert("RGB")
                    img_tensor = self.transform(img)
                except Exception:
                    img_tensor = torch.zeros(3, 64, 64)
            else:
                img_tensor = torch.zeros(3, 64, 64)
            images.append(img_tensor)
            
        if not images:
            images = [torch.zeros(3, 64, 64)]
            
        return {
            "waveform": audio_item["waveform"],
            "image": torch.stack(images), # (L, C, H, W)
            "label": audio_item["label"],
            "label_str": word,
        }


class AtomicSpellerDataset(Dataset):
    """
    Dataset atómico para TinySpeller (G2P). 
    En lugar de palabras, asocia un único grafema (imagen) con su fonema objetivo.
    Garantiza balanceo entre clases mediante sobremuestreo de caracteres raros.
    """

    def __init__(
        self,
        visual_dataset: VisualLetterDataset,
        phoneme_to_idx: Dict[str, int],
        samples_per_class: int = 100
    ) -> None:
        self.visual_dataset = visual_dataset
        self.phoneme_to_idx = phoneme_to_idx
        
        # Organizar muestras originales por letra
        from collections import defaultdict
        letter_samples = defaultdict(list)
        for i in range(len(visual_dataset)):
            # Usamos el acceso directo para velocidad en la inicialización si es posible
            sample = visual_dataset.samples[i]
            # Solo incluir si el grafema existe en nuestro inventario de fonemas
            if sample.letter in phoneme_to_idx:
                letter_samples[sample.letter].append(i)
        
        # Balancear: Crear lista de índices sobremuestreando si es necesario
        self.balanced_indices = []
        for letter, indices in letter_samples.items():
            if not indices: continue
            
            # Repetir o truncar para alcanzar samples_per_class
            repeated = (indices * (samples_per_class // len(indices) + 1))[:samples_per_class]
            self.balanced_indices.extend(repeated)
            
        random.shuffle(self.balanced_indices)
        self.class_names = visual_dataset.letters

    def __len__(self) -> int:
        return len(self.balanced_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        orig_idx = self.balanced_indices[index]
        visual_item = self.visual_dataset[orig_idx]
        
        letter = visual_item["letter"]
        # Mapear el índice del dataset visual al índice del fonema del Reader
        label = self.phoneme_to_idx.get(letter, 0)
        
        return {
            "image": visual_item["image"], # (C, H, W) -> será (B, 1, C, H, W) en el collate
            "label": label,
            "label_str": letter,
            "is_atomic": True
        }


def reader_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function personalizada para manejar secuencias de imágenes y audio de longitud variable.
    """
    from torch.nn.utils.rnn import pad_sequence
    
    waveforms = [item["waveform"] for item in batch]
    # waveforms_padded = pad_sequence(waveforms, batch_first=True)
    
    # Padear secuencias de imágenes (B, max_L, C, H, W)
    images = [item["image"] for item in batch]
    images_padded = pad_sequence(images, batch_first=True)
    
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    label_strs = [item["label_str"] for item in batch]
    
    return {
        "waveforms": waveforms, # Lista para el módulo
        "image": images_padded,  # Tensor para evaluate_and_save (compatibilidad)
        "label": labels,
        "label_str": label_strs,
        "waveform": pad_sequence(waveforms, batch_first=True) # Para evaluate_and_save (compatibilidad)
    }


def build_reader_dataloaders(
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    split_ratios: Dict[str, float] | None = None,
    target_language: str | None = None,
    training_phase: str = "end_to_end",
    phoneme_to_idx: Dict[str, int] | None = None,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, DataLoader]]:
    """
    Construye dataloaders para TinyReader / TinySpeller.
    """
    # FORZAR MODO ATÓMICO EN G2P
    if training_phase == "g2p":
        # --- MODO ATÓMICO: Speller Balanceado ---
        from training.visual_dataset import build_visual_dataloaders
        
        inventory = get_phoneme_inventory(target_language) if target_language else []
        
        # Sincronización Estricta con Fallback Determinista
        if not phoneme_to_idx:
            # Si no hay mapa, usamos el inventario del idioma ORDENADO para consistencia
            phoneme_to_idx = {p: i for i, p in enumerate(sorted(inventory))}
            print(f"⚠️ phoneme_to_idx no proporcionado. Usando fallback ordenado ({len(phoneme_to_idx)} fonemas).")
        
        # Filtrar inventario por lo que el modelo realmente conoce
        valid_chars = [c for c in inventory if c in phoneme_to_idx]
        if not valid_chars:
            # Fallback final a todo el inventario si el filtrado falló
            valid_chars = inventory
        
        _, _, _, v_loaders = build_visual_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            split_ratios=split_ratios,
            target_language=target_language,
            whitelist_chars=valid_chars
        )
        
        datasets = {}
        for split in ["train", "val", "test"]:
            v_ds = v_loaders[split].dataset
            datasets[split] = AtomicSpellerDataset(v_ds, phoneme_to_idx, samples_per_class=100)
            
        loaders = {}
        for split in ["train", "val", "test"]:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=speller_atomic_collate_fn,
                pin_memory=torch.cuda.is_available()
            )
        return datasets["train"], datasets["val"], datasets["test"], loaders

    # --- MODO SECUENCIAL: Reader Clásico ---
    # 1. Cargar datasets de audio (base para el reader)
    audio_datasets = build_audio_datasets(
        seed=seed,
        split_ratios=split_ratios,
        target_language=target_language,
        use_phonemes=False # Entrenamos para palabras completas
    )
    
    # 2. Cargar configuración visual para las imágenes de grafemas
    config = load_master_dataset_config()
    visual_config = config.get("visual_dataset", {}).get("generated_images", {})
    
    # 3. Envolver en ReaderDataset
    datasets: Dict[str, ReaderDataset] = {}
    for split, ds in audio_datasets.items():
        datasets[split] = ReaderDataset(ds, visual_config, target_language=target_language)
    
    # 4. Crear Loaders
    loaders: Dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=reader_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
    return datasets["train"], datasets["val"], datasets["test"], loaders


def speller_atomic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate para modo atómico. Convierte imágenes 3D en secuencias de longitud 1 
    para mantener compatibilidad con el proyector del Reader.
    """
    images = torch.stack([item["image"] for item in batch]) # (B, C, H, W)
    # El Reader espera (B, L, C, H, W). Expandimos L=1.
    images = images.unsqueeze(1)
    
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    label_strs = [item["label_str"] for item in batch]
    
    return {
        "image": images,
        "label": labels,
        "label_str": label_strs,
        "is_atomic": True
    }
