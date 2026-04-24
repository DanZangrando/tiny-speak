"""
training/callbacks.py — Callbacks de PyTorch Lightning reutilizables.

Consolida todas las definiciones de callbacks que estaban dispersas
en utils.py y en las páginas individuales de Streamlit.

Clases:
    TrainingHistoryCallback     Acumula métricas por época.
    RealTimePlotCallback        Actualiza gráficos de Streamlit en tiempo real.
    ReaderPredictionCallback    Visualiza predicciones del Reader por época.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd
import pytorch_lightning as pl
import torch

if TYPE_CHECKING:
    pass  # evitar imports circulares en type hints


# ---------------------------------------------------------------------------
# History callback
# ---------------------------------------------------------------------------

class TrainingHistoryCallback(pl.Callback):
    """Acumula el historial de métricas por época para análisis posterior."""

    def __init__(self) -> None:
        self.history: List[dict] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
        }
        metrics["epoch"] = trainer.current_epoch
        self.history.append(metrics)


# ---------------------------------------------------------------------------
# Real-time Streamlit plotting callback
# ---------------------------------------------------------------------------

class RealTimePlotCallback(pl.Callback):
    """Actualiza gráficas de Streamlit en tiempo real usando Plotly para fijar límites.

    Args:
        placeholder_loss: ``st.empty()`` para la gráfica de pérdida.
        placeholder_acc:  ``st.empty()`` para la gráfica de accuracy.
        max_epochs:       int, total de épocas para escalar el eje X.
    """

    def __init__(self, placeholder_loss, placeholder_acc, max_epochs: int) -> None:
        self.placeholder_loss = placeholder_loss
        self.placeholder_acc = placeholder_acc
        self.max_epochs = max_epochs
        self.history: List[dict] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import streamlit as st
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
        }
        epoch = trainer.current_epoch
        metrics["epoch"] = epoch
        self.history.append(metrics)

        # Create DataFrame with full epoch index to fix X axis
        current_df = pd.DataFrame(self.history).set_index("epoch")
        full_index = pd.Index(range(self.max_epochs), name="epoch")
        
        # Plot Loss
        loss_cols = [c for c in current_df.columns if "loss" in c]
        if loss_cols:
            df_loss = pd.DataFrame(index=full_index).join(current_df[loss_cols], how="left")
            self.placeholder_loss.line_chart(df_loss, width='stretch')

        # Plot Accuracy / Metrics
        acc_cols = [c for c in current_df.columns if any(x in c for x in ["acc", "top1", "dtw", "perceptual"])]
        if acc_cols:
            df_acc = pd.DataFrame(index=full_index).join(current_df[acc_cols], how="left")
            
            # For accuracy columns that are 0-1, rescale to 0-100 for consistency if others are 0-100
            for col in acc_cols:
                if "acc" in col or "top1" in col:
                    if df_acc[col].max() <= 1.1:
                         df_acc[col] = df_acc[col] * 100
            
            self.placeholder_acc.line_chart(df_acc, width='stretch')


# ---------------------------------------------------------------------------
# Reader prediction callback
# ---------------------------------------------------------------------------

class ReaderPredictionCallback(pl.Callback):
    """Visualiza predicciones del TinyReader al final de cada época de validación.

    Args:
        val_loader:  DataLoader de validación.
        placeholder: ``st.empty()`` donde se renderizará la tabla de predicciones.
    """

    def __init__(self, val_loader, placeholder) -> None:
        self.val_loader = val_loader
        self.placeholder = placeholder
        # Pre-cargar un batch fijo para visualización consistente
        self.batch = next(iter(val_loader))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import streamlit as st

        try:
            res = pl_module.get_predictions(self.batch)
            is_g2p = getattr(pl_module, "training_phase", "end_to_end") == "g2p"
            
            if is_g2p and len(res) == 4:
                words, predictions, confidences, targets = res
            else:
                words, predictions, confidences = res[:3]
                targets = None

            results = []
            for i in range(min(len(words), 10)):
                real_word = words[i]
                pred = predictions[i]
                conf = confidences[i]
                
                if is_g2p and targets:
                    target_phonemes = targets[i]
                    # El estado se basa en si la predicción coincide con el target de fonemas
                    icon = "✅" if pred == target_phonemes else "❌"
                    results.append({
                        "Palabra": real_word,
                        "Target (Fonemas)": target_phonemes,
                        "Predicción": pred,
                        "Confianza": f"{conf:.2%}",
                        "Estado": icon,
                    })
                else:
                    icon = "✅" if real_word == pred else "❌"
                    results.append({
                        "Input (Letras)": real_word,
                        "Predicción": pred,
                        "Confianza": f"{conf:.2%}",
                        "Estado": icon,
                    })

            df = pd.DataFrame(results)

            with self.placeholder.container():
                st.markdown(f"### 🔮 Predicciones (Época {trainer.current_epoch})")
                st.dataframe(df, hide_index=True, width='stretch')

        except Exception as exc:
            import traceback
            with self.placeholder.container():
                st.warning(f"⚠️ Error en predicciones: {exc}")
                st.code(traceback.format_exc(), language="python")
