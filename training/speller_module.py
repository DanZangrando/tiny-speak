"""
Módulo de entrenamiento para TinySpeller mejorado.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import TinySpellerImproved
from training.audio_dataset import AudioWordDataset
from training.config import load_master_dataset_config
from training.visual_dataset import VisualLetterDataset


@dataclass
class MultimodalSample:
    """Container for a multimodal training sample."""
    word: str
    word_idx: int
    letter_paths: List[Path]
    audio_waveform: torch.Tensor
    seq_length: int
    metadata: Dict


class MultimodalWordDataset(Dataset):
    """
    Dataset que combina audio y secuencias visuales para entrenamiento de TinySpeller.
    
    Para cada palabra:
    - Audio: waveform del AudioWordDataset
    - Visual: secuencia de imágenes de letras del VisualLetterDataset
    """
    
    def __init__(
        self,
        words: List[str],
        visual_dataset: VisualLetterDataset,
        audio_dataset: AudioWordDataset,
        max_word_length: int = 10,
        augment: bool = False,
        transform: Optional[transforms.Compose] = None
    ):
        self.words = words
        self.word_to_idx = {word: idx for idx, word in enumerate(words)}
        self.max_word_length = max_word_length
        self.augment = augment
        
        # Transform para imágenes
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Augmentations para entrenamiento
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.aug_transform = self.transform
        
        # Crear mapeo letra -> imágenes disponibles
        self.letter_to_images: Dict[str, List[Path]] = {}
        for sample in visual_dataset.samples:
            self.letter_to_images.setdefault(sample.letter, []).append(sample.path)
        
        # Filtrar samples de audio que tienen palabras en nuestro vocabulario
        self.samples: List[MultimodalSample] = []
        for audio_sample in audio_dataset.samples:
            word = audio_sample.word.lower().strip()
            if word in self.word_to_idx and len(word) <= max_word_length:
                # Verificar que todas las letras están disponibles
                if all(letter in self.letter_to_images for letter in word):
                    # Obtener paths de imágenes para cada letra
                    letter_paths = []
                    for letter in word:
                        available_paths = self.letter_to_images[letter]
                        letter_paths.append(random.choice(available_paths))
                    
                    sample = MultimodalSample(
                        word=word,
                        word_idx=self.word_to_idx[word],
                        letter_paths=letter_paths,
                        audio_waveform=audio_sample.waveform,
                        seq_length=len(word),
                        metadata=audio_sample.metadata
                    )
                    self.samples.append(sample)
        
        if not self.samples:
            raise ValueError("No se encontraron samples válidos para el dataset multimodal")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Cargar y procesar imágenes de letras
        letter_images = []
        for letter_path in sample.letter_paths:
            image = Image.open(letter_path).convert('RGB')
            
            # Aplicar transform (con o sin augmentation)
            if self.augment:
                image_tensor = self.aug_transform(image)
            else:
                image_tensor = self.transform(image)
                
            letter_images.append(image_tensor)
        
        # Pad sequence a max_word_length
        while len(letter_images) < self.max_word_length:
            # Pad con imagen negra
            letter_images.append(torch.zeros_like(letter_images[0]))
        
        # Truncate si es necesario
        letter_images = letter_images[:self.max_word_length]
        
        return {
            'images': torch.stack(letter_images),       # [max_word_length, 3, 64, 64]
            'audio': sample.audio_waveform,             # [audio_length]
            'word_label': sample.word_idx,              # int
            'word': sample.word,                        # str
            'seq_length': sample.seq_length,            # int
            'mask': torch.cat([                         # [max_word_length] - máscara para padding
                torch.ones(sample.seq_length),
                torch.zeros(self.max_word_length - sample.seq_length)
            ])
        }


class TinySpellerLightning(pl.LightningModule):
    """PyTorch Lightning module para entrenar TinySpeller mejorado."""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,  # número de letras únicas
        embed_dim: int = 768,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_recognizer: bool = False,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Modelo mejorado
        self.model = TinySpellerImproved(
            num_classes=num_classes,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            freeze_recognizer=freeze_recognizer
        )
        
        # Loss con label smoothing para mejor generalización
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        
        # Para tracking de predicciones durante validación
        self.validation_predictions: List[Dict] = []
        self.test_predictions: List[Dict] = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits
    
    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        images = batch['images']         # [batch, seq_len, 3, 64, 64]
        labels = batch['word_label']     # [batch]
        
        # Forward pass
        logits = self.forward(images)    # [batch, vocab_size]
        loss = self.criterion(logits, labels)
        
        # Métricas
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Top-k accuracy
        top3_acc = self._compute_topk_accuracy(logits, labels, k=3)
        top5_acc = self._compute_topk_accuracy(logits, labels, k=5)
        
        # Log métricas
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_top3', top3_acc, sync_dist=True)
        self.log(f'{stage}_top5', top5_acc, sync_dist=True)
        
        # Guardar predicciones para análisis
        if stage in ['val', 'test']:
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=1)
            
            predictions = {
                'words': batch['word'],
                'true_labels': labels.cpu().tolist(),
                'predicted_labels': preds.cpu().tolist(),
                'probabilities': probs.cpu().tolist(),
                'top_indices': top_indices.cpu().tolist(),
                'top_probs': top_probs.cpu().tolist(),
            }
            
            if stage == 'val':
                self.validation_predictions.append(predictions)
            else:
                self.test_predictions.append(predictions)
        
        return loss
    
    def _compute_topk_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """Compute top-k accuracy"""
        if k > logits.size(1):
            k = logits.size(1)
        
        _, pred_indices = logits.topk(k, dim=1)
        correct = pred_indices.eq(labels.view(-1, 1).expand_as(pred_indices))
        correct_k = correct.sum().float()
        
        return correct_k / labels.size(0) * 100.0
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')
    
    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'test')
    
    def on_validation_epoch_start(self) -> None:
        self.validation_predictions.clear()
    
    def on_test_epoch_start(self) -> None:
        self.test_predictions.clear()
    
    def configure_optimizers(self):
        """Configurar optimizadores con learning rate scheduling"""
        
        # Usar learning rates diferenciados si el recognizer no está congelado
        if not self.hparams.freeze_recognizer:
            # Learning rate más bajo para el backbone pre-entrenado
            optimizer = torch.optim.AdamW([
                {
                    'params': self.model.visual_encoder.parameters(), 
                    'lr': self.hparams.learning_rate * 0.1  # 10x más lento
                },
                {
                    'params': list(self.model.sequence_encoder.parameters()) + 
                             list(self.model.classifier.parameters()),
                    'lr': self.hparams.learning_rate
                }
            ], weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


def multimodal_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Función de collate personalizada para manejar audio de tamaños diferentes.
    """
    # Obtener tensores de audio y encontrar la longitud máxima
    audio_tensors = [item['audio'] for item in batch]
    max_audio_length = max(audio.shape[0] for audio in audio_tensors)
    
    # Pad todos los audios a la longitud máxima
    padded_audios = []
    audio_masks = []
    
    for audio in audio_tensors:
        current_length = audio.shape[0]
        if current_length < max_audio_length:
            # Pad con zeros
            padding = torch.zeros(max_audio_length - current_length)
            padded_audio = torch.cat([audio, padding])
        else:
            padded_audio = audio
            
        padded_audios.append(padded_audio)
        
        # Crear máscara (1 para audio real, 0 para padding)
        mask = torch.cat([
            torch.ones(current_length),
            torch.zeros(max_audio_length - current_length)
        ])
        audio_masks.append(mask)
    
    # Stack otros tensores normalmente
    return {
        'images': torch.stack([item['images'] for item in batch]),
        'audio': torch.stack(padded_audios),
        'audio_mask': torch.stack(audio_masks),
        'word_label': torch.tensor([item['word_label'] for item in batch]),
        'word': [item['word'] for item in batch],  # Lista de strings
        'seq_length': torch.tensor([item['seq_length'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch])
    }


def build_multimodal_dataloaders(
    words: List[str],
    batch_size: int = 16,
    num_workers: int = 0,
    max_word_length: int = 10,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construir dataloaders multimodales para train/val/test.
    """
    
    # Cargar datasets base
    from training.visual_dataset import build_visual_dataloaders
    from training.audio_dataset import build_audio_datasets, DEFAULT_AUDIO_SPLIT_RATIOS
    
    # Visual datasets
    train_visual, val_visual, test_visual, _ = build_visual_dataloaders(
        batch_size=batch_size, num_workers=0, seed=seed
    )
    
    # Audio datasets  
    audio_datasets = build_audio_datasets(seed=seed, split_ratios=DEFAULT_AUDIO_SPLIT_RATIOS)
    
    # Crear datasets multimodales
    train_multimodal = MultimodalWordDataset(
        words=words,
        visual_dataset=train_visual,
        audio_dataset=audio_datasets['train'],
        max_word_length=max_word_length,
        augment=True
    )
    
    val_multimodal = MultimodalWordDataset(
        words=words,
        visual_dataset=val_visual,
        audio_dataset=audio_datasets['val'],
        max_word_length=max_word_length,
        augment=False
    )
    
    test_multimodal = MultimodalWordDataset(
        words=words,
        visual_dataset=test_visual,
        audio_dataset=audio_datasets['test'],
        max_word_length=max_word_length,
        augment=False
    )
    
    # Crear dataloaders con collate_fn personalizada
    train_loader = DataLoader(
        train_multimodal, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn
    )
    
    val_loader = DataLoader(
        val_multimodal,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn
    )
    
    test_loader = DataLoader(
        test_multimodal,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn
    )
    
    return train_loader, val_loader, test_loader