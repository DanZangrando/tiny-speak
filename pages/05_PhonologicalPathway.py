"""
🎵 Phonological Pathway (TinyListener) - Entrenamiento y analítica sobre el dataset de audio.
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_listener_diagram
from components.code_viewer import get_function_source
from components.analytics import plot_learning_curves, plot_confusion_matrix, display_classification_report, plot_probability_matrix
from models import PhonologicalPathway
from training.audio_dataset import build_audio_dataloaders, DEFAULT_AUDIO_SPLIT_RATIOS
from training.audio_module import PhonologicalPathwayLightning
from training.config import load_master_dataset_config
from utils import (
    WAV2VEC_DIM,
    WAV2VEC_SR,
    encontrar_device,
    get_default_words,
    load_waveform,
    list_checkpoints,
    save_model_metadata,
    RealTimePlotCallback
)

# Configurar página
st.set_page_config(
    page_title="Phonological Pathway - Audición",
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
    
    st.markdown('<h1 class="main-header">👂 Phonological Pathway: Reconocimiento de Voz</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("### 👂 Phonological Pathway (Vía Fonológica)")
        
        st.markdown("""
        <div class="card">
            <b>Tarea:</b> Reconocimiento de palabras habladas (ASR).<br>
            <b>Input:</b> Forma de onda de audio (Batch, Samples).<br>
            <b>Output:</b> Logits (Batch, NumClasses).<br>
            <b>Arquitectura:</b> Modelo entrenado desde cero que combina:
            <ul>
                <li><b>Feature Extractor (CNN):</b> Convierte el audio raw en características latentes.</li>
                <li><b>Context Encoder (Transformer):</b> Procesa dependencias temporales.</li>
                <li><b>Classifier:</b> Predice la palabra.</li>
            </ul>
            <b>Justificación:</b>
            <ul>
                <li><b>Mel-Spectrogram:</b> Simula la cóclea del oído interno, descomponiendo el sonido en frecuencias logarítmicas, similar a la percepción humana.</li>
                <li><b>CNN & Transformer:</b> La CNN procesa características locales (timbre) mientras que el Transformer integra la secuencia temporal, permitiendo distinguir palabras con los mismos fonemas en diferente orden (ej. "casa" vs "saca").</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.graphviz_chart(get_listener_diagram())
        
        with st.expander("💻 Ver Código del Modelo (models.py)"):
            st.code(get_function_source(PhonologicalPathway), language="python")

    # ==========================================
    # TAB 2: ENTRENAMIENTO
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Experimento")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Hiperparámetros")
            epochs = st.number_input("Épocas", min_value=1, value=20)
            lr = st.number_input("Learning Rate", value=1e-3, format="%.1e")
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=2)
            
        with col2:
            st.markdown("#### Callbacks & Optimizador")
            st.info("⚡ Gestión automática con PyTorch Lightning.")
            st.markdown("- **ModelCheckpoint**: Guarda el mejor modelo (val_loss).")
            st.markdown("- **EarlyStopping**: Detiene si no mejora.")
            
            patience = st.slider("Patience (Early Stopping)", 1, 20, 10, help="Número de épocas sin mejora antes de detener.")
            min_delta = st.slider("Min Delta (Early Stopping)", 0.0, 0.1, 0.00, step=0.001, format="%.3f", help="Mejora mínima para considerar.")
            
        # Selector de Idioma
        config = load_master_dataset_config()
        exp_config = config.get('experiment_config', {})
        available_langs = exp_config.get('languages', ['es'])
        
        target_lang = st.selectbox("Idioma de Entrenamiento", available_langs, index=0)
            
        if st.button("🚀 Iniciar Entrenamiento", type="primary"):
            run_training(epochs, lr, batch_size, patience, min_delta, target_lang)
            
        with st.expander("💻 Ver Código de Entrenamiento (LightningModule)"):
            st.code(get_function_source(PhonologicalPathwayLightning), language="python")

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
                        # 1. Cargar Metadata (Vocabulario y Idioma)
                        meta_path = Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json")
                        words = []
                        target_lang = None
                        
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta_config = json.load(f).get("config", {})
                                words = meta_config.get("vocab", [])
                                target_lang = meta_config.get("language")
                        
                        # Fallback legacy
                        if not target_lang:
                            for lang in ['es', 'en', 'fr']:
                                if f"_{lang}_" in sel_ckpt['filename']:
                                    target_lang = lang
                                    break
                        
                        if not target_lang:
                            config = load_master_dataset_config()
                            target_lang = config.get('experiment_config', {}).get('languages', ['es'])[0]
                            st.warning(f"⚠️ Idioma no detectado en metadata. Usando '{target_lang}' por defecto.")

                        if not words and target_lang:
                             try:
                                 with st.spinner(f"Cargando vocabulario para {target_lang}..."):
                                     train_ds, _, _, _ = build_audio_dataloaders(target_language=target_lang)
                                     words = train_ds.class_names
                             except Exception as e:
                                 st.warning(f"No se pudo cargar vocabulario del dataset: {e}")
                        
                        if not words:
                            st.error("❌ No se pudo determinar el vocabulario (class_names) para cargar el modelo.")
                            st.stop()

                        # 2. Cargar Modelo
                        model = PhonologicalPathwayLightning.load_from_checkpoint(sel_ckpt['path'], class_names=words)
                        model.eval()
                        device = encontrar_device()
                        model.to(device)
                        
                        # Verificar consistencia
                        num_classes_model = model.model.classifier.out_features
                        if words and len(words) != num_classes_model:
                            st.warning(f"⚠️ Mismatch: Metadata tiene {len(words)} palabras, pero el modelo espera {num_classes_model}. Ignorando metadata.")
                            words = []
                            
                        if not words:
                            words = [f"Class {i}" for i in range(num_classes_model)]
                            st.info(f"Usando etiquetas genéricas para {num_classes_model} clases.")

                        model.class_names = words

                        # 3. Cargar Datos
                        st.info(f"Cargando datos de validación para idioma: {target_lang}")
                        _, _, _, loaders = build_audio_dataloaders(
                            batch_size=16, 
                            num_workers=0, 
                            seed=42, 
                            target_language=target_lang
                        )
                        val_loader = loaders['val']
                        
                        # 4. Inferencia
                        all_preds = []
                        all_probs = []
                        all_labels = []
                        progress_bar = st.progress(0)
                        total_batches = len(val_loader)
                        
                        with torch.no_grad():
                            for idx, batch in enumerate(val_loader):
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
                        
                        # 5. Visualizar
                        st.markdown("#### Mapa de Calor de Probabilidades")
                        display_labels = words
                        max_label = max(all_labels) if all_labels else 0
                        if max_label >= len(display_labels):
                            st.error(f"❌ Error: El dataset contiene etiquetas ({max_label}) fuera del rango del modelo ({len(display_labels)-1}).")
                        else:
                            plot_probability_matrix(all_labels, all_probs, display_labels)
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

def run_training(epochs, lr, batch_size, patience, min_delta, target_language):
    config = load_master_dataset_config()
    
    with st.spinner(f"Cargando datos para idioma '{target_language}'..."):
        train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
            batch_size=batch_size, 
            num_workers=0, 
            seed=42,
            target_language=target_language
        )
        
    words = train_ds.class_names
    
    if not words:
        st.error(f"No hay palabras en el dataset para el idioma {target_language}.")
        return
        
    model = PhonologicalPathwayLightning(
        class_names=words,
        learning_rate=lr
    )
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/listener_checkpoints",
        filename="listener-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    history_cb = ListenerHistoryCallback()
    
    # Placeholders para gráficas en tiempo real
    st.markdown("### 📈 Progreso en Tiempo Real")
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("#### Pérdida (Loss)")
        plot_loss = st.empty()
    with col_plot2:
        st.markdown("#### Precisión (Accuracy)")
        plot_acc = st.empty()
        
    realtime_cb = RealTimePlotCallback(plot_loss, plot_acc)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[history_cb, realtime_cb, early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        default_root_dir="lightning_logs/tiny_listener"
    )
    
    progress_bar = st.progress(0)
    with st.spinner(f"Entrenando..."):
        trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
        
    st.success("Entrenamiento completado!")
    
    # Guardar el mejor modelo
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        st.warning("No se encontró un checkpoint del mejor modelo. Usando el estado final.")
        save_dir = Path("models/listener")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        final_path = save_dir / f"listener_{target_language}_{timestamp}.ckpt"
        trainer.save_checkpoint(final_path)
    else:
        import shutil
        save_dir = Path("models/listener")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        final_path = save_dir / f"listener_{target_language}_{timestamp}.ckpt"
        shutil.copy(best_model_path, final_path)
        st.info(f"Mejor modelo restaurado desde: {Path(best_model_path).name}")
    
    # Guardar vocabulario en metadata
    meta_config = {
        "epochs": epochs, 
        "lr": lr, 
        "batch_size": batch_size,
        "vocab": words,
        "language": target_language
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
    
    meta_path = Path(sel_ckpt_path).with_suffix(".ckpt.meta.json")
    words = []
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            words = meta.get("config", {}).get("vocab", [])
        except:
            pass
            
    if not words:
        st.warning("⚠️ No se encontró vocabulario en metadata. Usando configuración global (puede haber mismatch).")
        config = load_master_dataset_config()
        words = config.get("diccionario_seleccionado", {}).get("palabras", [])
    
    if st.button("Cargar Modelo"):
        st.session_state['listener_model'] = PhonologicalPathwayLightning.load_from_checkpoint(
            sel_ckpt_path, class_names=words
        )
        st.session_state['listener_model'].eval()
        st.session_state['listener_vocab'] = words
        st.success(f"Modelo cargado con {len(words)} palabras!")
        
    if 'listener_model' in st.session_state:
        model_vocab = st.session_state.get('listener_vocab', words)
        
        audio_dir = Path("data/audios")
        audios = list(audio_dir.glob("**/*.wav"))
        
        if not audios:
            st.error("No se encontraron audios.")
            return
            
        sel_audio_path = st.selectbox("Probar con audio:", [str(p) for p in audios])
        st.audio(sel_audio_path)
        
        if st.button("Analizar Audio"):
            waveform = load_waveform(sel_audio_path)
            
            with torch.no_grad():
                # Asegurar que el input esté en el mismo dispositivo que el modelo
                device = next(st.session_state['listener_model'].parameters()).device
                waveform = waveform.to(device)
                
                waveform = waveform.unsqueeze(0)
                logits = st.session_state['listener_model'](waveform)
                probs = torch.softmax(logits, dim=1)
                
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
                    "Probabilidad": probs[0].cpu().numpy()
                })
                st.bar_chart(df_probs.set_index("Palabra"))

if __name__ == "__main__":
    main()
