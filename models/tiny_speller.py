"""
TinySpeller - Grapheme-to-Phoneme Model (Stage 1).

Esta arquitectura se encarga del ensamblaje fonológico:
convierte grafemas visuales (desde TinyEyes) en representaciones auditivas 
de fonemas (capaces de ser clasificadas por TinyEars Phonemes).

A diferencia de un modelo Seq2Seq, este es un mapeador independiente (Pointwise),
lo que asegura que cada grafema se traduzca a sonido sin memoria temporal del resto de la palabra,
emulando el proceso de conciencia fonológica básica.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class TinySpeller(nn.Module):
    """
    TinySpeller (Stage 1): Grapheme-to-Phoneme Projector.

    Proyecta activaciones visuales latentes (IT cortex de TinyEyes, 512-dim) 
    al espacio latente fonético de TinyEars (Phonemes) de forma independiente por cada letra.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        
        # Red densa frame-a-frame
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ])
            curr_dim = hidden_dim
            
        # Salida al espacio latente del Phoneme Listener
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projector = nn.Sequential(*layers)

    def forward(
        self,
        x_seq: torch.Tensor,
        target_length: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: (B, L, input_dim) - Secuencia de grafemas
            target_length: Ignorado (mantenido por compatibilidad de firma)
            
        Returns:
            phoneme_embeddings: (B, L, output_dim)
        """
        return self.projector(x_seq)

# Alias para compatibilidad con código que aún use el nombre antiguo
TinyReaderG2P = TinySpeller
