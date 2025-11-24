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
from components.analytics import plot_learning_curves, plot_confusion_matrix, display_classification_report, plot_probability_matrix
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
    list_checkpoints,
    save_model_metadata
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
    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📚 Gestión de Modelos")
        checkpoints = list_checkpoints("listener")
        
        if not checkpoints:
            st.info("No hay modelos entrenados.")
        else:
            ckpt_opts = {f"{c['filename']} ({datetime.fromtimestamp(c['timestamp']).strftime('%Y-%m-%d %H:%M')})": c for c in checkpoints}
            sel_ckpt_key = st.selectbox("Seleccionar Modelo para Detalles", list(ckpt_opts.keys()))
            sel_ckpt = ckpt_opts[sel_ckpt_key]
            
            col_info, col_actions = st.columns([3, 1])
            with col_info:
                st.markdown(f"**Archivo:** `{sel_ckpt['filename']}`")
                meta = sel_ckpt.get('meta', {})
                
                # Configuración
                with st.expander("⚙️ Configuración de Entrenamiento", expanded=True):
                    st.json(meta.get('config', {}))
                
                # Métricas
                metrics = meta.get('metrics', {})
                if metrics:
                    st.markdown("#### 📊 Métricas Finales")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Val Accuracy", f"{metrics.get('val_top1', 0):.2f}%")
                    m_col2.metric("Val Loss", f"{metrics.get('val_loss', 0):.4f}")
                    m_col3.metric("Train Loss", f"{metrics.get('train_loss', 0):.4f}")
                    
            with col_actions:
                st.markdown("### Acciones")
                if st.button("🗑️ Eliminar Modelo", key=f"del_{sel_ckpt['filename']}", type="primary"):
                    Path(sel_ckpt['path']).unlink(missing_ok=True)
                    Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json").unlink(missing_ok=True)
                    Path(sel_ckpt['path']).with_suffix(".csv").unlink(missing_ok=True)
                    st.rerun()
            
            st.divider()
            
            # --- SECCIÓN DE ANALÍTICA AVANZADA ---
            st.markdown("### 📈 Analítica del Modelo")
            
            # 1. Curvas de Aprendizaje
            hist_path = Path(sel_ckpt['path']).with_suffix(".csv")
            if hist_path.exists():
                history_df = pd.read_csv(hist_path)
                plot_learning_curves(history_df)
            else:
                st.info("⚠️ No hay historial de entrenamiento detallado disponible.")

            # 2. Evaluación
            st.markdown("### 🧪 Evaluación Detallada")
            st.markdown("Ejecuta una evaluación completa sobre el conjunto de validación.")
            
            if st.button("🚀 Ejecutar Evaluación Completa", key=f"eval_{sel_ckpt['filename']}"):
                with st.spinner("Cargando modelo y datos..."):
                    try:
                        # Cargar Vocab
                        meta_path = Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json")
                        words = []
                        if meta_path.exists():
                            with open(meta_path) as f:
                                words = json.load(f).get("config", {}).get("vocab", [])
                        if not words:
                            config = load_master_dataset_config()
                            words = config.get("diccionario_seleccionado", {}).get("palabras", [])
                            
                        # Cargar Modelo
                        model = TinyListenerLightning.load_from_checkpoint(sel_ckpt['path'], class_names=words)
                        model.eval()
                        device = encontrar_device()
                        model.to(device)
                        
                        # Cargar Datos
                        _, _, _, loaders = build_audio_dataloaders(batch_size=16, num_workers=0, seed=42)
                        val_loader = loaders['val']
                        
                        # Inferencia
                        all_preds = []
                        all_probs = []
                        all_labels = []
                        progress_bar = st.progress(0)
                        total_batches = len(val_loader)
                        
                        with torch.no_grad():
                            for idx, batch in enumerate(val_loader):
                                # TinyListenerLightning espera waveforms como lista en batch['waveforms']
                                # Pero el dataloader devuelve batch['waveforms'] como lista de tensores
                                waveforms = [w.to(device) for w in batch['waveforms']]
                                labels = batch['label'].to(device)
                                
                                logits = model(waveforms)
                                probs = torch.softmax(logits, dim=1)
                                preds = torch.argmax(probs, dim=1)
                                
                                all_preds.extend(preds.cpu().numpy())
                                all_probs.extend(probs.cpu().numpy())
                                all_labels.extend(labels.cpu().numpy())
                                progress_bar.progress((idx + 1) / total_batches)
                                
                        st.success("Evaluación completada.")
                        
                        # Visualizar
                        st.markdown("#### Mapa de Calor de Probabilidades")
                        display_labels = words if len(words) == len(model.class_names) else [str(i) for i in range(len(model.class_names))]
                        
                        # Usar el nuevo heatmap de probabilidades
                        plot_probability_matrix(all_labels, all_probs, display_labels)
                        
                        # Reporte clásico
                        display_classification_report(all_labels, all_preds, display_labels)
                        
                    except Exception as e:
                        st.error(f"Error durante la evaluación: {e}")
                        st.exception(e)

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
    
    # Guardar vocabulario en metadata para el laboratorio
    meta_config = {
        "epochs": epochs, 
        "lr": lr, 
        "batch_size": batch_size,
        "vocab": words # Guardar palabras entrenadas
    }
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, meta_config, final_metrics)
    
    # Guardar historial completo
    if history_cb.history:
        hist_path = final_path.with_suffix(".csv")
        pd.DataFrame(history_cb.history).to_csv(hist_path, index=False)
    
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
    sel_ckpt_name = st.selectbox("Seleccionar Modelo", list(ckpt_opts.keys()))
    sel_ckpt_path = ckpt_opts[sel_ckpt_name]
    
    # Intentar cargar vocabulario desde metadata
    meta_path = Path(sel_ckpt_path).with_suffix(".ckpt.meta.json")
    words = []
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            words = meta.get("config", {}).get("vocab", [])
        except:
            pass
            
    # Fallback al config global si no hay metadata
    if not words:
        st.warning("⚠️ No se encontró vocabulario en metadata. Usando configuración global (puede haber mismatch).")
        config = load_master_dataset_config()
        words = config.get("diccionario_seleccionado", {}).get("palabras", [])
    
    if st.button("Cargar Modelo"):
        st.session_state['listener_model'] = TinyListenerLightning.load_from_checkpoint(
            sel_ckpt_path, class_names=words
        )
        st.session_state['listener_model'].eval()
        st.session_state['listener_vocab'] = words # Guardar vocabulario en sesión
        st.success(f"Modelo cargado con {len(words)} palabras!")
        
    if 'listener_model' in st.session_state:
        model_vocab = st.session_state.get('listener_vocab', words)
        
        # Seleccionar audio del dataset
        audio_dir = Path("data/audios")
        # Buscar audios recursivamente
        audios = list(audio_dir.glob("**/*.wav"))
        
        if not audios:
            st.error("No se encontraron audios.")
            return
            
        sel_audio_path = st.selectbox("Probar con audio:", [str(p) for p in audios])
        st.audio(sel_audio_path)
        
        if st.button("Analizar Audio"):
            waveform = load_waveform(sel_audio_path)
            
            with torch.no_grad():
                # Forward pass
                waveform = waveform.unsqueeze(0)
                logits = st.session_state['listener_model'](waveform)
                probs = torch.softmax(logits, dim=1)
                
            # Visualización Mejorada
            st.markdown("### 📊 Resultados del Análisis")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.markdown("**Top 3 Predicciones**")
                top_probs, top_idxs = torch.topk(probs[0], 3)
                for p, idx in zip(top_probs, top_idxs):
                    word = model_vocab[idx]
                    st.markdown(f"- **{word}**: `{p:.2%}`")
                    st.progress(float(p))
            
            with col_res2:
                st.markdown("**Distribución de Probabilidad**")
                df_probs = pd.DataFrame({
                    "Palabra": model_vocab,
                    "Probabilidad": probs[0].numpy()
                })
                st.bar_chart(df_probs.set_index("Palabra"))

if __name__ == "__main__":
    main()
