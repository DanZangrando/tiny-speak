"""
TinyEyes - Visual Pathway (Vía Visual).

Simula el flujo ventral de la corteza visual (V1 → V2 → V4 → IT).
Arquitectura CNN diseñada desde cero para el reconocimiento de grafemas.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyEyes(nn.Module):
    """
    Vía Visual personalizada (TinyEyes).

    Pipeline:
        Imagen (B, 3, 64, 64)
        → Conv Block ×4  (simulando V1 → V2 → V4 → IT)
        → AdaptiveAvgPool
        → Classifier

    Retorna (logits, embedding) para compatibilidad con el sistema.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 64),          # 64×64 → 32×32  (V1)
            conv_block(64, 128),        # 32×32 → 16×16  (V2)
            conv_block(128, 256),       # 16×16 → 8×8    (V4)
            conv_block(256, hidden_dim), # 8×8  → 4×4    (IT)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def update_num_classes(self, num_classes: int) -> None:
        """Reinstancia el clasificador con un nuevo número de clases."""
        device = next(self.parameters()).device
        self.classifier = nn.Linear(self.hidden_dim, num_classes).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 64, 64)

        Returns:
            logits:    (B, num_classes)
            embedding: (B, hidden_dim)
        """
        x = self.features(x)
        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)  # (B, hidden_dim)
        logits = self.classifier(embedding)
        return logits, embedding
