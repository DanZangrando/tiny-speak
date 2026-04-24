"""
TinyEars - Phonological Pathway (Vía Auditiva).

Simula la corteza auditiva: Waveform → MelSpectrogram → CNN → Transformer → Embeddings.
Diseñada desde cero para ser "Tiny" y biológicamente plausible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ===========================================================================
# POSITIONAL ENCODING
# ===========================================================================

class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal estándar (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [Batch, SeqLen, Dim]  (batch_first=True)
        """
        # pe shape: (max_len, 1, d_model) → transpose for batch_first
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


# ===========================================================================
# PHONOLOGICAL PATHWAY  (TinyEars)
# ===========================================================================

class TinyEars(nn.Module):
    """
    Vía Fonológica personalizada (TinyEars).

    Pipeline:
        Waveform (T,)
        → MelSpectrogram (n_mels, T_spec)
        → Log-Mel
        → CNN Feature Extractor  (hidden_dim, T')
        → Positional Encoding
        → Transformer Encoder    (hidden_dim, T')
        → Mean Pooling + Classifier
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_conv_layers: int = 3,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        sample_rate: int = 16_000,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- 0. Audio Transform (Waveform → MelSpectrogram) ---
        try:
            import torchaudio

            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=400,
                hop_length=160,
            )
        except ImportError:
            raise ImportError(
                "torchaudio es necesario para TinyEars. "
                "Instálalo con: pip install torchaudio"
            )

        # --- 1. CNN Feature Extractor ---
        layers: list[nn.Module] = []
        in_channels = n_mels
        for i in range(num_conv_layers):
            out_channels = hidden_dim if i == num_conv_layers - 1 else 64 * (2**i)
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                    nn.GroupNorm(out_channels // 8 if out_channels > 8 else 1, out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.post_extract_proj = nn.Linear(in_channels, hidden_dim)

        # --- 2. Positional Encoding ---
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)

        # --- 3. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- 4. Classifier ---
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Para compatibilidad con Reader (target layer)
        self.target_layer = num_transformer_layers - 1

    def update_num_classes(self, num_classes: int) -> None:
        """Reinstancia el clasificador con un nuevo número de clases."""
        device = next(self.parameters()).device
        self.classifier = nn.Linear(self.hidden_dim, num_classes).to(device)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Waveforms (B, T) → CNN features (B, T', C)."""
        x = self.mel_spectrogram(waveforms)       # (B, n_mels, T_spec)
        x = torch.log(x + 1e-9)                   # Log-Mel
        features = self.feature_extractor(x)       # (B, C, T')
        return features.transpose(1, 2)            # (B, T', C)

    def extract_hidden_activations(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Interfaz requerida por TinyReader. Retorna hidden states del Transformer.

        Returns:
            Tensor (1, B, T', hidden_dim)  — fake-stack para compatibilidad con Reader.
        """
        features = self.extract_features(waveforms)
        features = self.post_extract_proj(features)
        features = self.pos_encoder(features)
        encoded = self.transformer(features)
        return encoded.unsqueeze(0)  # (1, B, T', D)

    def mask_hidden_activations(
        self, hidden_activations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden_activations (1, B, T, D) → (1, B, T, D), lengths (B,)."""
        hidden = hidden_activations[0]  # (B, T, D)
        B, T, _ = hidden.shape
        lengths = torch.full((B,), T, dtype=torch.long, device=hidden.device)
        return hidden.unsqueeze(0), lengths

    def downsample_hidden_activations(
        self,
        hidden_activations: torch.Tensor,
        lengths: torch.Tensor,
        factor: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """No-op downsampler (CNN ya reduce temporalmente)."""
        return hidden_activations, lengths

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveforms: (B, T)

        Returns:
            logits:  (B, num_classes)
            encoded: (B, T', hidden_dim)
        """
        features = self.extract_features(waveforms)   # (B, T', C)
        features = self.post_extract_proj(features)    # (B, T', D)
        features = self.pos_encoder(features)
        encoded = self.transformer(features)           # (B, T', D)
        pooled = encoded.mean(dim=1)                   # (B, D)
        logits = self.classifier(pooled)
        return logits, encoded
