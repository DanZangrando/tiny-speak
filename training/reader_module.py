"""PyTorch Lightning module para entrenar TinyReader (Generación Top-Down)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import random
from pathlib import Path
from PIL import Image

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision import transforms

from models import TinyReader, PhonologicalPathway, VisualPathway
from utils import WAV2VEC_DIM, load_wav2vec_model
from training.config import load_master_dataset_config

class TinyReaderLightning(pl.LightningModule):
    """
    LightningModule para TinyReader.
    Aprende a generar embeddings de Wav2Vec2 a partir de secuencias de logits de letras (Spelling).
    Usa VisualPathway para "leer" las letras y PhonologicalPathway como "Oído Interno".
    """

    def __init__(
        self,
        class_names: Sequence[str],
        listener_checkpoint_path: str,
        recognizer_checkpoint_path: str, # Nuevo: Path al reconocedor
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        w_mse: float = 1.0,
        w_cos: float = 1.0,
        w_perceptual: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.class_names = list(class_names)
        # self.num_classes ya no se usa para el encoder lineal, sino para mapear labels a texto
        
        # Cargar configuración para buscar imágenes
        self.dataset_config = load_master_dataset_config()
        self.visual_config = self.dataset_config.get("visual_dataset", {}).get("generated_images", {})
        
        # Transformaciones para imágenes (deben coincidir con las del Recognizer)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # 1. Modelo Perceptivo (Listener - Oído Interno)
        self.listener = self._load_listener(listener_checkpoint_path)
        self.listener.eval()
        for p in self.listener.parameters():
            p.requires_grad = False
        # self.listener.tiny_speak.train() # YA NO EXISTE TINY SPEAK
        
        # 2. Modelo Visual (Recognizer - Ojo)
        self.recognizer = self._load_recognizer(recognizer_checkpoint_path)
        self.recognizer.eval()
        for p in self.recognizer.parameters():
            p.requires_grad = False
            
        # Obtener dimensión de salida del recognizer (num_letters)
        # Asumimos que el clasificador es lineal
        if self.recognizer.classifier:
            input_dim = self.recognizer.classifier.out_features
        else:
            # Fallback si es directo CORnet (aunque VisualPathway siempre tiene classifier)
            input_dim = 26 # Default

        # 3. Modelo Generativo (Reader)
        self.reader = TinyReader(
            input_dim=input_dim, # Dimensión de logits de letras
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            wav2vec_dim=WAV2VEC_DIM
        )
            
        # Pérdidas
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.perceptual_loss = nn.CrossEntropyLoss()

    def _load_recognizer(self, checkpoint_path: str) -> VisualPathway:
        """Carga el VisualPathway desde checkpoint para usarlo como ojo."""
        import importlib
        RecognizerPL = importlib.import_module("training.visual_module").VisualPathwayLightning
        
        print(f"Cargando VisualPathway (Ojo) desde {checkpoint_path}...")
        # Cargar el módulo completo
        # Nota: No capturamos excepciones aquí para que falle visiblemente si el checkpoint está mal.
        pl_module = RecognizerPL.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device if hasattr(self, "device") else "cpu"
        )
        return pl_module.model

    def _get_word_images(self, word: str) -> torch.Tensor:
        """
        Obtiene una secuencia de imágenes para deletrear la palabra.
        Retorna: (L, C, H, W)
        """
        images = []
        repo_root = Path(self.dataset_config.get("repo_root", "."))
        
        for char in word:
            # Buscar imágenes para el caracter
            # Normalizar caracter si es necesario (ej. lower)
            char_key = char # Asumimos que el config tiene las keys correctas
            
            entries = self.visual_config.get(char_key, [])
            if not entries:
                # Fallback: intentar lower/upper o generar ruido/negro
                entries = self.visual_config.get(char_key.lower(), [])
            
            if entries:
                # Elegir una al azar (Data Augmentation implícito)
                entry = random.choice(entries)
                rel_path = entry.get("file_path")
                full_path = repo_root / rel_path
                
                try:
                    img = Image.open(full_path).convert("RGB")
                    img_tensor = self.transform(img)
                except Exception:
                    # Si falla carga, imagen negra
                    img_tensor = torch.zeros(3, 64, 64)
            else:
                # Si no hay imágenes para la letra, usar negro
                img_tensor = torch.zeros(3, 64, 64)
                
            images.append(img_tensor)
            
        if not images:
            return torch.zeros(1, 3, 64, 64)
            
        return torch.stack(images)

    def _load_listener(self, checkpoint_path: str) -> PhonologicalPathway:
        """Carga el PhonologicalPathway desde checkpoint para usarlo como juez."""
        # Importar aquí para evitar ciclos si fuera necesario, aunque ya está arriba
        from training.audio_module import PhonologicalPathwayLightning as ListenerPL
        
        print(f"Cargando PhonologicalPathway (Oído Interno) desde {checkpoint_path}...")
        # Cargar el módulo completo
        pl_module = ListenerPL.load_from_checkpoint(
            checkpoint_path,
            class_names=self.class_names,
            map_location=self.device if hasattr(self, "device") else "cpu"
        )
        return pl_module.model

    def forward(self, x_seq, target_length=None):
        return self.reader(x_seq, target_length)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        # Datos reales
        waveforms = [w.to(self.device) for w in batch["waveforms"]]
        labels = batch["label"].to(self.device)
        batch_size = len(waveforms)
        
        # 1. Obtener Ground Truth Embeddings (Bottom-Up)
        with torch.no_grad():
            # Pad waveforms to create a batch tensor (B, T)
            waveforms_padded = pad_sequence(waveforms, batch_first=True)
            
            # Usamos la parte de extracción de TinyListener
            real_embeddings = self.listener.extract_hidden_activations(waveforms_padded)
            real_embeddings, lengths = self.listener.mask_hidden_activations(real_embeddings)
            real_embeddings, lengths = self.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
            real_embeddings = real_embeddings.squeeze(0)
            
        # 2. Generar Imaginación (Top-Down)
        # A. Obtener secuencia de logits de letras (Spelling)
        # Convertir labels (indices) a palabras
        words = [self.class_names[i] for i in labels]
        
        # Obtener imágenes y pasar por Recognizer
        logits_sequences = []
        for word in words:
            # (L_word, C, H, W)
            images = self._get_word_images(word).to(self.device)
            
            with torch.no_grad():
                # (L_word, NumLetters)
                # Recognizer retorna (logits, embeddings) o (logits, logits)
                # Asumimos que forward retorna tupla y queremos el primero
                res = self.recognizer(images)
                if isinstance(res, tuple):
                    word_logits = res[0]
                else:
                    word_logits = res
            
            logits_sequences.append(word_logits)
            
        # Pad sequences para batch processing en Reader
        # (B, L_max, NumLetters)
        padded_logits = pad_sequence(logits_sequences, batch_first=True, padding_value=0.0)
        
        # B. Generar audio embeddings
        # (B, T, D)
        max_len = real_embeddings.size(1)
        generated_embeddings = self.reader(padded_logits, target_length=max_len)
        
        # 3. Calcular Pérdidas
        
        # A. Reconstruction Loss (MSE)
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        mse = F.mse_loss(generated_embeddings[mask], real_embeddings[mask])
        
        # B. Cosine Similarity Loss
        gen_flat = generated_embeddings[mask]
        real_flat = real_embeddings[mask]
        target_ones = torch.ones(gen_flat.size(0), device=self.device)
        cosine = self.cosine_loss(gen_flat, real_flat, target_ones)
        
        # C. Perceptual Loss
        # El nuevo TinyListener usa Mean Pooling + Linear Classifier sobre los embeddings
        # generated_embeddings: (B, T, D)
        # Masking para mean pooling correcto
        mask_float = mask.float().unsqueeze(-1) # (B, T, 1)
        pooled_gen = (generated_embeddings * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        
        listener_logits = self.listener.classifier(pooled_gen)
        perceptual = F.cross_entropy(listener_logits, labels)
        
        # Total Loss
        total_loss = (
            self.hparams.w_mse * mse + 
            self.hparams.w_cos * cosine + 
            self.hparams.w_perceptual * perceptual
        )
        
        # Logging
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_mse", mse)
        self.log(f"{stage}_cos", cosine)
        self.log(f"{stage}_perceptual", perceptual)
        
        return total_loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def get_predictions(self, batch: Dict) -> Tuple[List[str], torch.Tensor]:
        """
        Retorna las etiquetas reales y los logits predichos para un batch.
        Usado para visualización durante el entrenamiento.
        """
        # Solo cambiamos el modo del reader para evitar efectos secundarios globales
        was_training = self.reader.training
        self.reader.eval()
        try:
            with torch.no_grad():
                # Datos
                waveforms = [w.to(self.device) for w in batch["waveforms"]]
                labels = batch["label"].to(self.device)
                
                # Ground Truth (para longitud)
                waveforms_padded = pad_sequence(waveforms, batch_first=True)
                real_embeddings = self.listener.extract_hidden_activations(waveforms_padded)
                real_embeddings, lengths = self.listener.mask_hidden_activations(real_embeddings)
                real_embeddings, lengths = self.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
                max_len = real_embeddings.size(1)
                
                # Generación Top-Down
                words = [self.class_names[i] for i in labels]
                logits_sequences = []
                for word in words:
                    images = self._get_word_images(word).to(self.device)
                    res = self.recognizer(images)
                    word_logits = res[0] if isinstance(res, tuple) else res
                    logits_sequences.append(word_logits)
                
                padded_logits = pad_sequence(logits_sequences, batch_first=True, padding_value=0.0)
                
                # Generar embeddings
                generated_embeddings = self.reader(padded_logits, target_length=max_len)
                
                # Clasificar embeddings generados con el Listener
                # Masking
                mask = torch.arange(max_len, device=self.device).expand(len(waveforms), max_len) < lengths.unsqueeze(1)
                mask_float = mask.float().unsqueeze(-1)
                pooled_gen = (generated_embeddings * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
                
                listener_logits = self.listener.classifier(pooled_gen)
                
                return words, listener_logits
        finally:
            self.reader.train(was_training)
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.reader.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }
