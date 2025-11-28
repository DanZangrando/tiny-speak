"""
🖼️ Visual Pathway (TinyRecognizer) - Entrenamiento y analítica sobre el dataset visual actual.
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import streamlit as st
import torch
from PIL import Image
from matplotlib import pyplot as plt

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_recognizer_diagram
from components.code_viewer import get_function_source
from models import VisualPathway
from training.visual_dataset import VisualLetterDataset, build_visual_dataloaders, DEFAULT_SPLIT_RATIOS
from training.visual_module import VisualPathwayLightning
from training.config import load_master_dataset_config
from utils import (
    WAV2VEC_DIM,
    encontrar_device,
    get_default_words,
    load_waveform,
    load_waveform,
    list_checkpoints,
    save_model_metadata,
    RealTimePlotCallback
)
from components.analytics import plot_learning_curves, plot_confusion_matrix, display_classification_report

# Configurar página
st.set_page_config(
    page_title="Visual Pathway - Visión",
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
    
    st.markdown('<h1 class="main-header">👁️ Visual Pathway: Visión Artificial</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("### 🧠 Visual Pathway (Vía Visual)")
        
        st.markdown("""
        <div class="card">
            <b>Tarea:</b> Clasificación de imágenes de letras (64x64 RGB).<br>
            <b>Input:</b> Tensor (Batch, 3, 64, 64).<br>
            <b>Output:</b> Logits (Batch, NumClasses).<br>
            <b>Arquitectura:</b> CNN personalizada y ligera.<br>
            <b>Justificación:</b>
            <ul>
                <li><b>CNN Jerárquica:</b> Simula la estructura de la corteza visual (V1, V2, V4, IT), donde las primeras capas detectan bordes simples y las capas profundas reconocen formas complejas (letras completas).</li>
                <li><b>Eficiencia:</b> Al usar una arquitectura ligera y personalizada, evitamos el sobreajuste que tendrían modelos masivos como ResNet en este dataset específico.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.graphviz_chart(get_recognizer_diagram())
        
        with st.expander("💻 Ver Código del Modelo (models.py)"):
            st.code(get_function_source(VisualPathway), language="python")

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
            st.code(get_function_source(VisualPathwayLightning), language="python")

    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📚 Gestión de Modelos")
        checkpoints = list_checkpoints("recognizer")
        
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
            
            # 1. Curvas de Aprendizaje (si existen)
            hist_path = Path(sel_ckpt['path']).with_suffix(".csv")
            if hist_path.exists():
                history_df = pd.read_csv(hist_path)
                plot_learning_curves(history_df)
            else:
                st.info("⚠️ No hay historial de entrenamiento detallado disponible para este modelo (modelos antiguos).")

            # 2. Evaluación en Validation Set
            st.markdown("### 🧪 Evaluación Detallada")
            st.markdown("Ejecuta una evaluación completa sobre el conjunto de validación para generar la Matriz de Confusión.")
            
            if st.button("🚀 Ejecutar Evaluación Completa", key=f"eval_{sel_ckpt['filename']}"):
                with st.spinner("Cargando modelo y datos..."):
                    # Cargar Modelo
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
                            
                        # 2. Cargar Lightning Module
                        # Primero cargamos sin num_classes para ver qué tiene el checkpoint
                        # Pero VisualPathwayLightning requiere num_classes en __init__ si no está en hparams
                        # Intentemos cargar confiando en hparams del checkpoint
                        try:
                            model = VisualPathwayLightning.load_from_checkpoint(sel_ckpt['path'])
                        except:
                            # Si falla, intentamos inferir o usar vocab
                            kwargs = {"num_classes": len(words)} if words else {}
                            model = VisualPathwayLightning.load_from_checkpoint(sel_ckpt['path'], **kwargs)

                        model.eval()
                        device = encontrar_device()
                        model.to(device)
                        
                        # Verificar consistencia
                        num_classes_model = model.hparams.num_classes
                        if words and len(words) != num_classes_model:
                            st.warning(f"⚠️ Mismatch: Metadata tiene {len(words)} clases, pero el modelo espera {num_classes_model}. Ignorando metadata.")
                            words = []
                            
                        if not words:
                            words = [f"Class {i}" for i in range(num_classes_model)]
                            st.info(f"Usando etiquetas genéricas para {num_classes_model} clases.")
                        
                        # 3. Cargar Datos (Validation Set)
                        st.info(f"Cargando datos de validación para idioma: {target_lang}")
                        _, _, _, loaders = build_visual_dataloaders(
                            batch_size=32, 
                            num_workers=0,
                            target_language=target_lang
                        )
                        val_loader = loaders['val']
                        
                        # 4. Inferencia Loop
                        all_preds = []
                        all_labels = []
                        
                        progress_bar = st.progress(0)
                        total_batches = len(val_loader)
                        
                        with torch.no_grad():
                            for idx, batch in enumerate(val_loader):
                                images = batch["image"].to(device)
                                labels = batch["label"].to(device)
                                
                                logits = model(images)
                                preds = torch.argmax(logits, dim=1)
                                
                                all_preds.extend(preds.cpu().numpy())
                                all_labels.extend(labels.cpu().numpy())
                                progress_bar.progress((idx + 1) / total_batches)
                                
                        # Visualizar Resultados
                        st.success("Evaluación completada.")
                        
                        # Matriz de Confusión
                        st.markdown("#### Matriz de Confusión")
                        display_labels = words
                        
                        # Validar rango
                        max_label = max(all_labels) if all_labels else 0
                        if max_label >= len(display_labels):
                             st.error(f"❌ Error: El dataset contiene etiquetas ({max_label}) fuera del rango del modelo ({len(display_labels)-1}).")
                        else:
                            plot_confusion_matrix(all_labels, all_preds, display_labels)
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
    # Cargar datos
    # Usamos build_visual_dataloaders que ya filtra por idioma y nos da las clases correctas
    
    with st.spinner(f"Cargando datos para idioma '{target_language}'..."):
        train_ds, val_ds, test_ds, loaders = build_visual_dataloaders(
            batch_size=batch_size,
            num_workers=0,
            target_language=target_language
        )
        train_loader = loaders['train']
        val_loader = loaders['val']
        
    # Las clases son las letras/grafemas presentes en el dataset filtrado
    words = train_ds.letters
    
    if not words:
        st.error(f"No se encontraron clases visuales (letras) para el idioma {target_language}.")
        return
        
    # Modelo
    model = VisualPathwayLightning(
        num_classes=len(words),
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
        dirpath="models/recognizer_checkpoints",
        filename="recognizer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    # Trainer
    history_cb = RecognizerHistoryCallback()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[history_cb, early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        default_root_dir="lightning_logs/tiny_recognizer"
    )
    
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
    trainer.callbacks.append(realtime_cb)
    
    progress_bar = st.progress(0)
    with st.spinner(f"Entrenando..."):
        trainer.fit(model, train_loader, val_loader)
        
    st.success("Entrenamiento completado!")
    
    # Guardar
    save_dir = Path("models/recognizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"recognizer_{target_language}_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    # Metadata
    meta_config = {
        "epochs": epochs, 
        "lr": lr, 
        "batch_size": batch_size,
        "vocab": words,
        "language": target_language
    }
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, meta_config, final_metrics)
    
    # Guardar historial completo para gráficas
    if history_cb.history:
        hist_path = final_path.with_suffix(".csv")
        pd.DataFrame(history_cb.history).to_csv(hist_path, index=False)
    
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
    
    # Intentar cargar vocabulario desde metadata
    sel_ckpt_path = ckpt_opts[sel_ckpt]
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
        st.warning("⚠️ No se encontró vocabulario en metadata. Intentando cargar con hiperparámetros del checkpoint...")
    
    if st.button("Cargar Modelo"):
        # Si tenemos palabras (metadata), forzamos num_classes. Si no, confiamos en el checkpoint.
        kwargs = {"num_classes": len(words)} if words else {}
        
        try:
            st.session_state['recognizer_model'] = VisualPathwayLightning.load_from_checkpoint(
                sel_ckpt_path, **kwargs
            )
            st.session_state['recognizer_model'].eval()
            
            # Si no había palabras, generar etiquetas genéricas o intentar machear con config global
            if not words:
                model_n = st.session_state['recognizer_model'].hparams.num_classes
                config = load_master_dataset_config()
                visual_cfg = config.get("visual_dataset", {})
                generated = visual_cfg.get("generated_images", {})
                global_words = sorted(generated.keys())
                
                if len(global_words) == model_n:
                    words = global_words
                    st.info(f"✅ Coincidencia de tamaño ({model_n}). Usando clases visuales globales.")
                else:
                    words = [f"Clase {i}" for i in range(model_n)]
                    st.warning(f"⚠️ Tamaño ({model_n}) no coincide con global ({len(global_words)}). Usando etiquetas genéricas.")
            
            st.session_state['recognizer_vocab'] = words
            st.success(f"Modelo cargado con {len(words)} clases!")
            
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")
        
    if 'recognizer_model' in st.session_state:
        model_vocab = st.session_state.get('recognizer_vocab', words)
        
        # Seleccionar imagen del dataset visual
        visual_dir = Path("data/visual")
        if not visual_dir.exists():
            st.error("No hay dataset visual.")
            return
            
        # Listar algunas imágenes (aleatorias para variedad)
        all_images = list(visual_dir.glob("**/*.png")) + list(visual_dir.glob("**/*.jpg"))
        import random
        random.shuffle(all_images)
        images = all_images[:20]
        images = sorted(images) # Ordenar la selección para que el dropdown se vea ordenado
        
        if not images:
            st.error(f"No se encontraron imágenes en {visual_dir}.")
            return
            
        sel_img_path = st.selectbox(f"Probar con imagen (Total encontradas: {len(list(visual_dir.glob('**/*')))})", [str(p) for p in images])
        
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
                # Usar vocabulario del modelo
                if idx < len(model_vocab):
                    w = model_vocab[idx]
                else:
                    w = f"Class {idx}"
                st.write(f"**{w}**: {p:.2%}")
                st.progress(float(p))

if __name__ == "__main__":
    main()
