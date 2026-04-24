"""
TinyReader - Modelos generativos (Top-Down).

Arquitecturas de mapeo neural:
1. TinySpeller (Stage 1):  Grapheme-to-Phoneme (Pointwise Projector).
2. TinyReaderP2W (Stage 2): Phoneme-to-Word (Sequential Assembly).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyReader(nn.Module):
    """
    Modelo Generativo base (Sequential Encoder).
    
    Toma una secuencia y la colapsa en un vector de contexto.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Encoder: lee la secuencia de entrada (fonemas)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Proyección al espacio de la imagen neural (Word Embedding)
        # 2 porque es bidireccional
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        target_length: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: (B, L, input_dim) - Secuencia de embeddings de fonemas
            target_length: Ignorado ( Stage 2 produce un único vector de palabra)
            
        Returns:
            word_embedding: (B, output_dim)
        """
        # Encode
        _, (h_n, c_n) = self.encoder(x_seq)
        
        # Concatenar los estados finales de ambas direcciones de la última capa
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        
        # Proyectar a la imagen neural de la palabra
        return self.output_projection(last_hidden)


class TinyReaderP2W(TinyReader):
    """
    Stage 2: Phoneme-to-Word (Sequential Assembly).

    Toma una secuencia de embeddings de fonemas (generados por TinySpeller)
    y los 'ensambla' en una única imagen neural de palabra compatible
    con el espacio latente de TinyEars (Words).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, num_layers)
