"""
🖼️ TinyRecognizer - Entrenamiento y analítica sobre el dataset visual actual.
"""

from __future__ import annotations

import copy
import os
import time
from datetime import datetime
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from PIL import Image
from matplotlib import pyplot as plt

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_recognizer_diagram
from components.code_viewer import get_function_source
from models import TinyRecognizer
from training.visual_dataset import VisualLetterDataset, build_visual_dataloaders, DEFAULT_SPLIT_RATIOS
from training.visual_module import TinyRecognizerLightning
from training.config import load_master_dataset_config
from utils import (
    WAV2VEC_DIM,
    encontrar_device,
    get_default_words,
    load_waveform,
    list_checkpoints
)

# Configurar página
st.set_page_config(
    page_title="TinyRecognizer - Visión",
    page_icon="👁️",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """

def save_model_metadata(ckpt_path, config, metrics):
    """Guarda metadatos del modelo para su gestión."""
    meta_path = Path(ckpt_path).with_suffix(".ckpt.meta.json")
    data = {
        "timestamp": time.time(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": metrics
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

class RecognizerHistoryCallback(pl.Callback):
    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in trainer.callback_metrics.items()}
        metrics['epoch'] = trainer.current_epoch
        self.history.append(metrics)

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("tiny_recognizer")
    
    st.markdown('<h1 class="main-header">👁️ TinyRecognizer: Visión Artificial</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("### 🧠 CORnet-Z: La Corteza Visual Artificial")
        
        st.markdown("""
        <div class="card">
            <b>Tarea:</b> Clasificación de imágenes de letras (64x64 RGB).<br>
            <b>Input:</b> Tensor (Batch, 3, 64, 64).<br>
            <b>Output:</b> Logits (Batch, NumClasses).<br>
            <b>Justificación:</b> Usamos <b>CORnet-Z</b>, una arquitectura diseñada para imitar la estructura de la corteza visual de los primates (V1, V2, V4, IT). 
            Esto permite que el modelo aprenda representaciones visuales biológicamente plausibles, alineándose con el objetivo cognitivo del proyecto.
        </div>
        """, unsafe_allow_html=True)
        
        st.graphviz_chart(get_recognizer_diagram())
        
        with st.expander("💻 Ver Código del Modelo (models.py)"):
            st.code(get_function_source(TinyRecognizer), language="python")

    # ==========================================
    # TAB 2: ENTRENAMIENTO
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Experimento")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Hiperparámetros")
            epochs = st.number_input("Épocas", 1, 100, 10)
            lr = st.number_input("Learning Rate", value=1e-3, format="%.1e")
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            
        with col2:
            st.markdown("#### Callbacks & Optimizador")
            st.info("⚡ Usamos PyTorch Lightning para gestionar el bucle de entrenamiento, checkpoints y logging automáticamente.")
            st.markdown("- **ModelCheckpoint**: Guarda el mejor modelo basado en `val_loss`.")
            st.markdown("- **ReduceLROnPlateau**: Reduce el LR si la loss se estanca.")
            
        if st.button("🚀 Iniciar Entrenamiento", type="primary"):
            run_training(epochs, lr, batch_size)
            
        with st.expander("💻 Ver Código de Entrenamiento (LightningModule)"):
            st.code(get_function_source(TinyRecognizerLightning), language="python")

    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📚 Gestión de Modelos")
        checkpoints = list_checkpoints("recognizer")
        
        if not checkpoints:
            st.info("No hay modelos entrenados.")
        else:
            for ckpt in checkpoints:
                with st.expander(f"🖼️ {ckpt['filename']} - {datetime.fromtimestamp(ckpt['timestamp']).strftime('%Y-%m-%d %H:%M')}"):
                    col_info, col_actions = st.columns([3, 1])
                    with col_info:
                        meta = ckpt.get('meta', {})
                        st.json(meta.get('config', {}), expanded=False)
                        metrics = meta.get('metrics', {})
                        if metrics:
                            st.metric("Val Accuracy", f"{metrics.get('val_top1', 0):.2%}")
                    with col_actions:
                        if st.button("🗑️ Eliminar", key=f"del_{ckpt['filename']}"):
                            Path(ckpt['path']).unlink(missing_ok=True)
                            Path(ckpt['path']).with_suffix(".ckpt.meta.json").unlink(missing_ok=True)
                            st.rerun()

    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    with tabs[3]:
        st.markdown("### 🧪 Prueba Interactiva")
        run_laboratory()

def run_training(epochs, lr, batch_size):
    # Cargar datos
    config = load_master_dataset_config()
    selected_dict = config.get("diccionario_seleccionado", {})
    words = selected_dict.get("palabras", [])
    
    if not words:
        st.error("No hay palabras en el diccionario.")
        return

    with st.spinner("Cargando datos..."):
        train_loader, val_loader, test_loader = build_visual_dataloaders(
            batch_size=batch_size,
            num_workers=0
        )
        
    # Modelo
    model = TinyRecognizerLightning(
        num_classes=len(words),
        learning_rate=lr
    )
    
    # Trainer
    history_cb = RecognizerHistoryCallback()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[history_cb],
        enable_progress_bar=True,
        default_root_dir="lightning_logs/tiny_recognizer"
    )
    
    progress_bar = st.progress(0)
    with st.spinner(f"Entrenando..."):
        trainer.fit(model, train_loader, val_loader)
        
    st.success("Entrenamiento completado!")
    
    # Guardar
    save_dir = Path("models/recognizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"recognizer_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    # Metadata
    meta_config = {"epochs": epochs, "lr": lr, "batch_size": batch_size}
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, meta_config, final_metrics)
    
    st.info(f"Modelo guardado en {final_path}")
    
    if history_cb.history:
        df = pd.DataFrame(history_cb.history)
        st.line_chart(df[['train_loss', 'val_loss']])
        st.line_chart(df[['train_top1', 'val_top1']])

def run_laboratory():
    checkpoints = list_checkpoints("recognizer")
    if not checkpoints:
        st.warning("Entrena un modelo primero.")
        return
        
    ckpt_opts = {c['filename']: c['path'] for c in checkpoints}
    sel_ckpt = st.selectbox("Seleccionar Modelo", list(ckpt_opts.keys()))
    
    # Cargar modelo
    config = load_master_dataset_config()
    words = config.get("diccionario_seleccionado", {}).get("palabras", [])
    
    if st.button("Cargar Modelo"):
        st.session_state['recognizer_model'] = TinyRecognizerLightning.load_from_checkpoint(
            ckpt_opts[sel_ckpt], num_classes=len(words)
        )
        st.session_state['recognizer_model'].eval()
        st.success("Modelo cargado!")
        
    if 'recognizer_model' in st.session_state:
        # Seleccionar imagen del dataset visual
        visual_dir = Path("data/visual")
        if not visual_dir.exists():
            st.error("No hay dataset visual.")
            return
            
        # Listar algunas imágenes
        images = list(visual_dir.glob("**/*.png"))[:20] # Solo muestra algunas para probar
        if not images:
            st.error("No se encontraron imágenes.")
            return
            
        sel_img_path = st.selectbox("Probar con imagen:", [str(p) for p in images])
        
        col_img, col_pred = st.columns(2)
        with col_img:
            image = Image.open(sel_img_path).convert("RGB")
            st.image(image, width=128)
            
        with col_pred:
            # Preprocesar
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                logits = st.session_state['recognizer_model'](img_tensor)
                probs = torch.softmax(logits, dim=1)
                
            # Top 3
            top_probs, top_idxs = torch.topk(probs[0], 3)
            for p, idx in zip(top_probs, top_idxs):
                st.write(f"**{words[idx]}**: {p:.2%}")
                st.progress(float(p))

if __name__ == "__main__":
    main()
