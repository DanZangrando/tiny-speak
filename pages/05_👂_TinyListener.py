"""
🎵 TinyListener - Entrenamiento y analítica sobre el dataset de audio.
"""

import streamlit as st
import pytorch_lightning as pl
from pathlib import Path
import torch
import pandas as pd
import time
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_listener_diagram
from components.code_viewer import get_function_source
from models import TinyListener, TinySpeak
from training.audio_dataset import build_audio_dataloaders, DEFAULT_AUDIO_SPLIT_RATIOS
from training.audio_module import TinyListenerLightning
from training.config import load_master_dataset_config
from utils import (
    WAV2VEC_DIM,
    WAV2VEC_SR,
    encontrar_device,
    get_default_words,
    load_waveform,
    list_checkpoints
)

# Configurar página
st.set_page_config(
    page_title="TinyListener - Audición",
    page_icon="👂",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #11998e, #38ef7d);
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
        border-left: 5px solid #11998e;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """

def save_model_metadata(ckpt_path, config, metrics):
    meta_path = Path(ckpt_path).with_suffix(".ckpt.meta.json")
    data = {
        "timestamp": time.time(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": metrics
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

class ListenerHistoryCallback(pl.Callback):
    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in trainer.callback_metrics.items()}
        metrics['epoch'] = trainer.current_epoch
        self.history.append(metrics)

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("tiny_listener")
    
    st.markdown('<h1 class="main-header">👂 TinyListener: Reconocimiento de Voz</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("### 👂 Wav2Vec 2.0 + TinySpeak")
        
        st.markdown("""
        <div class="card">
            <b>Tarea:</b> Reconocimiento de palabras habladas (ASR).<br>
            <b>Input:</b> Forma de onda de audio (Batch, Samples).<br>
            <b>Output:</b> Logits (Batch, NumClasses).<br>
            <b>Justificación:</b> Utilizamos <b>Wav2Vec 2.0</b> (pre-entrenado y congelado) como extractor de características auditivas robustas, simulando un sistema auditivo maduro. 
            Sobre él, entrenamos <b>TinySpeak</b> (una LSTM ligera) que aprende a asociar esas características con conceptos (palabras), imitando el aprendizaje del lenguaje.
        </div>
        """, unsafe_allow_html=True)
        
        st.graphviz_chart(get_listener_diagram())
        
        with st.expander("💻 Ver Código del Modelo (models.py)"):
            st.code(get_function_source(TinyListener), language="python")

    # ==========================================
    # TAB 2: ENTRENAMIENTO
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Experimento")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Hiperparámetros")
            epochs = st.number_input("Épocas", 1, 100, 20)
            lr = st.number_input("Learning Rate", value=1e-3, format="%.1e")
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=2)
            
        with col2:
            st.markdown("#### Callbacks & Optimizador")
            st.info("⚡ Gestión automática con PyTorch Lightning.")
            st.markdown("- **ModelCheckpoint**: Guarda el mejor modelo (val_loss).")
            st.markdown("- **EarlyStopping**: Detiene si no mejora en 5 épocas.")
            
        if st.button("🚀 Iniciar Entrenamiento", type="primary"):
            run_training(epochs, lr, batch_size)
            
        with st.expander("💻 Ver Código de Entrenamiento (LightningModule)"):
            st.code(get_function_source(TinyListenerLightning), language="python")

    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📚 Gestión de Modelos")
        checkpoints = list_checkpoints("listener")
        
        if not checkpoints:
            st.info("No hay modelos entrenados.")
        else:
            for ckpt in checkpoints:
                with st.expander(f"🎵 {ckpt['filename']} - {datetime.fromtimestamp(ckpt['timestamp']).strftime('%Y-%m-%d %H:%M')}"):
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
    config = load_master_dataset_config()
    words = config.get("diccionario_seleccionado", {}).get("palabras", [])
    
    if not words:
        st.error("No hay palabras en el diccionario.")
        return

    with st.spinner("Cargando datos..."):
        train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
            batch_size=batch_size, num_workers=0, seed=42
        )
        
    model = TinyListenerLightning(
        class_names=words,
        learning_rate=lr
    )
    
    history_cb = ListenerHistoryCallback()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[history_cb],
        enable_progress_bar=True,
        default_root_dir="lightning_logs/tiny_listener"
    )
    
    progress_bar = st.progress(0)
    with st.spinner(f"Entrenando..."):
        trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
        
    st.success("Entrenamiento completado!")
    
    save_dir = Path("models/listener")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"listener_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    meta_config = {"epochs": epochs, "lr": lr, "batch_size": batch_size}
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, meta_config, final_metrics)
    
    st.info(f"Modelo guardado en {final_path}")
    
    if history_cb.history:
        df = pd.DataFrame(history_cb.history)
        st.line_chart(df[['train_loss', 'val_loss']])
        st.line_chart(df[['train_top1', 'val_top1']])

def run_laboratory():
    checkpoints = list_checkpoints("listener")
    if not checkpoints:
        st.warning("Entrena un modelo primero.")
        return
        
    ckpt_opts = {c['filename']: c['path'] for c in checkpoints}
    sel_ckpt = st.selectbox("Seleccionar Modelo", list(ckpt_opts.keys()))
    
    config = load_master_dataset_config()
    words = config.get("diccionario_seleccionado", {}).get("palabras", [])
    
    if st.button("Cargar Modelo"):
        st.session_state['listener_model'] = TinyListenerLightning.load_from_checkpoint(
            ckpt_opts[sel_ckpt], class_names=words
        )
        st.session_state['listener_model'].eval()
        st.success("Modelo cargado!")
        
    if 'listener_model' in st.session_state:
        # Seleccionar audio del dataset
        audio_dir = Path("data/audios")
        # Buscar audios recursivamente
        audios = list(audio_dir.glob("**/*.wav"))[:20]
        
        if not audios:
            st.error("No se encontraron audios.")
            return
            
        sel_audio_path = st.selectbox("Probar con audio:", [str(p) for p in audios])
        st.audio(sel_audio_path)
        
        if st.button("Analizar Audio"):
            waveform = load_waveform(sel_audio_path)
            
            with torch.no_grad():
                # Forward pass
                # Listener espera lista de tensores o tensor batched
                # load_waveform devuelve (samples,) -> unsqueeze -> (1, samples)
                waveform = waveform.unsqueeze(0)
                logits, _ = st.session_state['listener_model'](waveform)
                probs = torch.softmax(logits, dim=1)
                
            top_probs, top_idxs = torch.topk(probs[0], 3)
            for p, idx in zip(top_probs, top_idxs):
                st.write(f"**{words[idx]}**: {p:.2%}")
                st.progress(float(p))

if __name__ == "__main__":
    main()
