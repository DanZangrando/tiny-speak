"""PyTorch Lightning module para entrenar TinyReader (Generación Top-Down)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from models import TinyReader, TinyListener, TinySpeak
from utils import WAV2VEC_DIM, load_wav2vec_model

class TinyReaderLightning(pl.LightningModule):
    """
    LightningModule para TinyReader.
    Aprende a generar embeddings de Wav2Vec2 a partir de logits de palabras.
    Usa TinyListener como "Oído Interno" para la pérdida perceptiva.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        listener_checkpoint_path: str,
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
        self.num_classes = len(self.class_names)
        
        # Modelo Generativo (Reader)
        self.reader = TinyReader(
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            wav2vec_dim=WAV2VEC_DIM
        )
        
        # Modelo Perceptivo (Listener - Oído Interno)
        # Se carga desde un checkpoint y se congela
        self.listener = self._load_listener(listener_checkpoint_path)
        self.listener.eval()
        for p in self.listener.parameters():
            p.requires_grad = False
            
        # FIX: cudnn RNN backward requires training mode
        # Forzamos que la parte recurrente (TinySpeak) esté en train()
        # Como no hay dropout por defecto en TinySpeak, esto es seguro (determinista)
        self.listener.tiny_speak.train()
            
        # Pérdidas
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.perceptual_loss = nn.CrossEntropyLoss()

    def _load_listener(self, checkpoint_path: str) -> TinyListener:
        """Carga el TinyListener desde checkpoint para usarlo como juez."""
        # Importar aquí para evitar ciclos si fuera necesario, aunque ya está arriba
        from training.audio_module import TinyListenerLightning as ListenerPL
        
        print(f"Cargando TinyListener (Oído Interno) desde {checkpoint_path}...")
        try:
            # Cargar el módulo completo
            pl_module = ListenerPL.load_from_checkpoint(
                checkpoint_path,
                class_names=self.class_names,
                map_location=self.device if self.device.type != 'cpu' else 'cpu'
            )
            return pl_module.listener
        except Exception as e:
            print(f"Error cargando listener: {e}")
            # Fallback dummy (no debería pasar en producción)
            wav2vec = load_wav2vec_model(device='cpu')
            speak = TinySpeak(words=self.class_names)
            return TinyListener(speak, wav2vec)

    def forward(self, logits, target_length=None):
        return self.reader(logits, target_length)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        # Datos reales
        # waveforms = batch["waveforms"] # No usamos waveforms crudos aquí, sino embeddings pre-extraídos si fuera eficiente, 
        # pero TinyListener extrae embeddings on-the-fly.
        # Para entrenar TinyReader necesitamos los "Target Embeddings" (Wav2Vec2 features reales).
        # Como el dataset devuelve waveforms, necesitamos pasarlos por el Wav2Vec2 del Listener (congelado)
        # para obtener el "Ground Truth" de los embeddings.
        
        waveforms = [w.to(self.device) for w in batch["waveforms"]]
        labels = batch["label"].to(self.device)
        batch_size = len(waveforms)
        
        # 1. Obtener Ground Truth Embeddings (Bottom-Up)
        with torch.no_grad():
            # Usamos la parte de extracción de TinyListener
            # extract_hidden_activations devuelve (B, L, D)
            real_embeddings = self.listener.extract_hidden_activations(waveforms)
            
            # Aplicar máscara y selección de capa (igual que en TinyListener.forward)
            real_embeddings, lengths = self.listener.mask_hidden_activations(real_embeddings)
            
            # Downsample (igual que en TinyListener.forward)
            real_embeddings, lengths = self.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
            
            # Squeeze para obtener (B, T, D)
            real_embeddings = real_embeddings.squeeze(0)
            
        # 2. Generar Imaginación (Top-Down)
        # Input: One-hot de la palabra correcta (Simulando concepto perfecto)
        # Opcional: Podríamos usar logits suavizados o ruidosos para robustez
        concept_logits = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Generamos secuencia del mismo largo que la real para comparar
        # (B, T, D)
        # Nota: real_embeddings tiene padding. Reader genera para el max length del batch.
        max_len = real_embeddings.size(1)
        generated_embeddings = self.reader(concept_logits, target_length=max_len)
        
        # 3. Calcular Pérdidas
        
        # A. Reconstruction Loss (MSE) - "Sensory Prediction Error"
        # Aplicar máscara para no contar padding en el loss
        # Crear máscara basada en lengths
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        mse = F.mse_loss(generated_embeddings[mask], real_embeddings[mask])
        
        # B. Cosine Similarity Loss - "Structural Error"
        # Flatten para comparar vectores
        gen_flat = generated_embeddings[mask]
        real_flat = real_embeddings[mask]
        target_ones = torch.ones(gen_flat.size(0), device=self.device)
        cosine = self.cosine_loss(gen_flat, real_flat, target_ones)
        
        # C. Perceptual Loss - "Inner Ear Check"
        # Pasamos lo generado por el TinySpeak (clasificador del Listener)
        # TinySpeak espera PackedSequence
        packed_gen = pack_padded_sequence(
            generated_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # Listener forward espera waveform, pero aquí inyectamos embeddings directo al TinySpeak
        # Accedemos directo al tiny_speak del listener
        listener_logits, _ = self.listener.tiny_speak(packed_gen)
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
