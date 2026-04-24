"""
utils/plotting.py — Visualizaciones y pérdida Soft-DTW.

Contiene:
    plot_waveform_native    Gráfico Plotly de una forma de onda.
    plot_logits_native      Gráfico Plotly de logits/probabilidades.
    SoftDTW                 Pérdida de alineamiento temporal diferenciable.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def plot_waveform_native(
    waveform,
    title: str = "Waveform",
    sample_rate: int = 16_000,
) -> go.Figure:
    """Crea un gráfico Plotly de una forma de onda."""
    if hasattr(waveform, "cpu"):
        waveform_np = waveform.cpu().numpy()
    else:
        waveform_np = np.array(waveform)

    if waveform_np.ndim > 1:
        waveform_np = waveform_np.squeeze()

    time_axis = np.linspace(0, len(waveform_np) / sample_rate, len(waveform_np))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time_axis, y=waveform_np, mode="lines",
                   name="Waveform", line=dict(color="#FF6B6B", width=1))
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud",
        height=300,
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def plot_logits_native(
    logits,
    words: list[str],
    title: str = "Predicciones del Modelo",
) -> go.Figure:
    """Crea un gráfico Plotly de barras horizontales (Top-10 predicciones)."""
    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
    else:
        probs = np.array(logits)

    top_indices = np.argsort(probs)[-10:][::-1]
    top_words = [words[i] for i in top_indices]
    top_probs = probs[top_indices]

    fig = go.Figure(
        data=[
            go.Bar(
                y=top_words,
                x=top_probs,
                orientation="h",
                marker=dict(color=top_probs, colorscale="Viridis",
                            colorbar=dict(title="Probabilidad")),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Probabilidad",
        yaxis_title="Palabras",
        height=400,
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Soft-DTW
# ---------------------------------------------------------------------------

class SoftDTW(nn.Module):
    """
    Implementación pura de Soft-DTW (Cuturi & Blondel, 2017) en PyTorch.

    Calcula la pérdida de alineamiento temporal diferenciable entre secuencias.

    Args:
        gamma:     Parámetro de suavizado. Valores pequeños ≈ DTW estándar.
        normalize: Si True, retorna dtw(x, y) − 0.5 * [dtw(x,x) + dtw(y,y)].
    """

    def __init__(self, gamma: float = 1.0, normalize: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            y: (B, M, D)

        Returns:
            Scalar loss.
        """
        dist = torch.sum((x.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=3)
        loss = self._soft_dtw(dist)

        if self.normalize:
            loss_x = self._soft_dtw(torch.sum((x.unsqueeze(2) - x.unsqueeze(1)) ** 2, dim=3))
            loss_y = self._soft_dtw(torch.sum((y.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=3))
            return (loss - 0.5 * (loss_x + loss_y)).mean()

        return loss.mean()

    def _soft_dtw(self, D: torch.Tensor) -> torch.Tensor:
        B, N, M = D.size()
        device = D.device
        gamma = self.gamma

        R = torch.ones((B, N + 2, M + 2), device=device) * 1e10
        R[:, 0, 0] = 0.0

        for j in range(1, M + 1):
            for i in range(1, N + 1):
                r0 = -R[:, i - 1, j - 1] / gamma
                r1 = -R[:, i - 1, j] / gamma
                r2 = -R[:, i, j - 1] / gamma

                rmax = torch.max(torch.max(r0, r1), r2)
                softmin = -gamma * (
                    rmax + torch.log(
                        torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
                    )
                )
                R[:, i, j] = D[:, i - 1, j - 1] + softmin

        return R[:, N, M]
