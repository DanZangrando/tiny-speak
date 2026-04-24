"""
👁️ TinyEyes - Reconocimiento Visual de Grafemas
"""

import streamlit as st
import pytorch_lightning as pl
from pathlib import Path
import torch
import pandas as pd
import json
import numpy as np

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_recognizer_diagram
from components.code_viewer import get_function_source
from components.analytics import plot_learning_curves, plot_confusion_matrix, display_classification_report
from models import TinyEyes
from training.visual_dataset import build_visual_dataloaders
from training.visual_module import TinyEyesLightning
from training.config import load_master_dataset_config, save_master_dataset_config
from utils.device import encontrar_device
from utils.checkpoints import list_checkpoints

def format_accuracy(val):
    """Asegura que el accuracy esté en rango 0-1 antes de aplicar .2%"""
    if val > 1.0:
        return val / 100.0
    return val

st.set_page_config(
    page_title="TinyEyes - Grafemas",
    page_icon="👁️",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #ff9966, #ff5e62);
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
        border-left: 5px solid #ff5e62;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """

def get_active_models():
    ckpts = list_checkpoints("tiny_eyes")
    active = {}
    for c in ckpts:
        lang = c['meta'].get('config', {}).get('language')
        if lang and lang not in active:
            active[lang] = c
    return active

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("tiny_eyes")
    
    st.markdown('<h1 class="main-header">👁️ TinyEyes: Reconocimiento Visual</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs([
        "📉 Entrenamiento Lotes", 
        "🧪 Historial y Resultados", 
        "🔍 Laboratorio Interactivo",
        "📐 Arquitectura Estructural"
    ])
    config = load_master_dataset_config()
    languages = config.get('experiment_config', {}).get('languages', ['es', 'en', 'fr'])
    active_models = get_active_models()

    # ==========================================
    # TAB 1: ENTRENAMIENTO Y MODELOS
    # ==========================================
    with tabs[0]:
        st.markdown("### 📊 Modelos Activos")
        if not active_models:
            st.info("No hay modelos entrenados actualmente. Inicia el entrenamiento por lotes abajo.")
        else:
            cols = st.columns(len(languages))
            for i, lang in enumerate(languages):
                with cols[i]:
                    st.markdown(f"#### Idioma: {lang.upper()}")
                    if lang in active_models:
                        ckpt = active_models[lang]
                        meta = ckpt.get('meta', {})
                        st.success("✅ Modelo Listo")
                        st.json({
                            "Épocas": meta.get('config', {}).get('epochs'),
                            "Val Loss": round(meta.get('metrics', {}).get('val_loss', 0.0), 4),
                            "Actualizado": ckpt.get('date', 'Desconocido')
                        })
                    else:
                        st.warning("⚠️ Pendiente de entrenar")

        st.divider()
        st.markdown("### ⚙️ Iniciar Entrenamiento")
        train_config = config.get("training_params", {}).get("tiny_eyes", {})
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Épocas", min_value=1, max_value=1000, value=train_config.get("epochs", 50))
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=train_config.get("batch_size", 32))
        with col2:
            lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=train_config.get("lr", 1e-3), format="%.5f")
            
        if st.button("🚀 Iniciar Entrenamiento por Lotes (Todos los Idiomas)", type="primary"):
            if "training_params" not in config:
                config["training_params"] = {}
            config["training_params"]["tiny_eyes"] = {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr
            }
            save_master_dataset_config(config)
            
            from training.runner import train_recognizer
            
            st.markdown("### 📈 Progreso de Entrenamiento...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            plots_container = st.container()

            for i, lang in enumerate(languages):
                status_text.markdown(f"**Entrenando TinyEyes para {lang.upper()}... ({i+1}/{len(languages)})**")
                
                visual_data = config.get('visual_dataset', {}).get('generated_images', {})
                if not visual_data:
                    st.warning(f"Saltando {lang}: No se encontraron imágenes generadas.")
                    continue
                
                with plots_container:
                    st.markdown(f"#### Entrenamiento: {lang.upper()}")
                    col_plot1, col_plot2 = st.columns(2)
                    plot_loss = col_plot1.empty()
                    plot_acc = col_plot2.empty()
                plot_placeholders = (plot_loss, plot_acc)
                
                train_conf = {
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size
                }
                
                try:
                    ckpt_path, hist = train_recognizer(lang, train_conf, plot_placeholders=plot_placeholders)
                    st.success(f"✅ {lang.upper()} guardado.")
                except Exception as e:
                    st.error(f"❌ Error en {lang}: {e}")
                
                progress_bar.progress((i + 1) / len(languages))
            
            st.success("🎉 Entrenamientos completados. Actualizando la interfaz en 2 segundos...")
            import time
            time.sleep(2)
            st.rerun()

    # ==========================================
    # TAB 2: RESULTADOS GLOBALES E HISTORIAL
    # ==========================================
    with tabs[1]:
        st.markdown("### 🧪 Historial y Resultados Generales")
        if not active_models:
            st.warning("Debes entrenar los modelos primero.")
        else:
            from components.analytics import plot_training_history, plot_confusion_matrix, display_classification_report, plot_latent_space_pca
            import pickle
            
            st.markdown("#### 🌍 Comparativa de Rendimiento por Idioma")
            eval_cols = st.columns(len(active_models))
            
            for i, (lang_eval, ckpt_info) in enumerate(active_models.items()):
                with eval_cols[i]:
                    st.markdown(f"## 🌍 {lang_eval.upper()}")
                    meta = ckpt_info.get('meta', {})
                    metrics = meta.get('metrics', {})
                    
                    # Métricas Rápidas
                    acc_raw = metrics.get('val_acc') or metrics.get('val_top1') or 0.0
                    st.metric("Val Acc", f"{format_accuracy(acc_raw):.2%}")
                    
                    # 1. Historial
                    st.markdown("#### 📈 Historial")
                    plot_training_history(meta.get('history', []))
                    
                    try:
                        eval_path = Path(ckpt_info['path']).parent / "eval_results.pkl"
                        if eval_path.exists():
                            with open(eval_path, "rb") as f:
                                data = pickle.load(f)
                            
                            # 2. Clasificación
                            conf = data.get("confusion", {})
                            if conf and conf.get("y_true"):
                                st.markdown("#### 🎯 Matriz de Confusión")
                                plot_confusion_matrix(conf["y_true"], conf["y_pred"], conf["class_names"])
                                display_classification_report(conf["y_true"], conf["y_pred"], conf["class_names"])
                            
                            # 3. PCA
                            embs = data.get("embeddings", [])
                            labels = data.get("labels", [])
                            if len(embs) > 0:
                                st.markdown("#### 🌌 Espacio Latente")
                                # Fallback robusto para nombres de clases
                                classes_to_use = conf.get("class_names") or meta.get('config', {}).get('classes') or []
                                if classes_to_use:
                                    plot_latent_space_pca(embs, labels, classes_to_use)
                                else:
                                    st.warning("No se encontraron nombres de clases para el PCA.")
                            
                            # 4. Muestras
                            samples = data.get("samples", [])
                            if samples:
                                st.markdown("#### 📄 Muestras Reales")
                                for s in samples[:5]:
                                    st.markdown(f"""
                                    <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin-bottom: 5px; border-left: 3px solid #ff5e62;">
                                        <b>Real:</b> {s['target']} | <b>Pred:</b> {s['prediction']}<br>
                                        <small>Conf: {s.get('confidence', 0.0):.2%}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Debes re-entrenar para ver validación detallada.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ==========================================
    # TAB 3: LABORATORIO INTERACTIVO
    # ==========================================
    with tabs[2]:
        st.markdown("### 🔍 Laboratorio Interactivo")
        if not active_models:
            st.warning("Debes entrenar los modelos primero.")
        else:
            lang_lab = st.selectbox("Seleccionar Idioma para Pruebas Manuales", list(active_models.keys()), key="lab_lang")
            ckpt_lab = active_models[lang_lab]
            
            if st.button(f"👁️ Cargar 1 Imagen de {lang_lab.upper()} y predecir", type="primary"):
                with st.spinner("Inferiendo..."):
                    try:
                        class_names = ckpt_lab.get('meta', {}).get('config', {}).get('classes', [])
                        meta_config = ckpt_lab.get('meta', {}).get('config', {})
                        model_hparams = {k: v for k, v in meta_config.items() if k in ["hidden_dim"]}
                        if not model_hparams:
                            model_hparams = config.get("architectures", {}).get("tiny_eyes", {})
                        
                        model = TinyEyesLightning.load_from_checkpoint(
                            ckpt_lab['path'],
                            class_names=class_names,
                            **model_hparams
                        )
                        model.eval()
                        device = encontrar_device()
                        model.to(device)
                        
                        _, _, _, loaders = build_visual_dataloaders(
                            batch_size=32, target_language=lang_lab, num_workers=0, seed=42
                        )
                        
                        target_class = st.selectbox("Seleccionar grafema a predecir", class_names, key=f"sel_05_{lang_lab}")
                        target_idx = class_names.index(target_class)
                        
                        val_ds = loaders['val'].dataset
                        found_item = None
                        for i in range(len(val_ds)):
                            item = val_ds[i]
                            if int(item["label"]) == target_idx:
                                found_item = item
                                break
                                
                        if found_item is not None:
                            image = found_item["image"].to(device)
                            label = int(found_item["label"])
                            
                            from torchvision.transforms import ToPILImage
                            img_pil = ToPILImage()(image.cpu())
                            
                            logits, _ = model.model(image.unsqueeze(0).to(device))
                            pred = torch.argmax(logits, dim=1)[0].item()
                            
                            col_img, col_res = st.columns([1, 2])
                            with col_img:
                                st.image(img_pil, caption=f"Verdadero: {class_names[label]}")
                            with col_res:
                                st.markdown(f"**Verdadera Categoría:** `{class_names[label]}`")
                                st.markdown(f"**Categoría Predicha:** `{class_names[pred]}`")
                                
                                if pred == label:
                                    st.success("¡Predicción correcta!")
                                else:
                                    st.warning("Predicción incorrecta.")
                        else:
                            st.warning("No se encontraron muestras en validación.")
                    except Exception as e:
                        st.error(f"Error cargando instancia: {e}")

    # ==========================================
    # TAB 4: ARQUITECTURA
    # ==========================================
    with tabs[3]:
        st.markdown("### 📐 Arquitectura de la Red: TinyEyes")
        st.info("La arquitectura configurada aquí se aplica consistentemente a los tres idiomas al momento de entrenar.")
        
        arch_config = config.get("architectures", {}).get("tiny_eyes", {})
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Parámetros Sistemáticos")
            with st.form("arch_form_tiny_eyes"):
                h_dim = st.number_input("Dimensión Oculta (hidden_dim)", min_value=64, max_value=2048, value=arch_config.get("hidden_dim", 512), step=64)
                
                if st.form_submit_button("💾 Guardar Configuración Arquitectónica", type="primary"):
                    if "architectures" not in config:
                        config["architectures"] = {}
                    config["architectures"]["tiny_eyes"] = {
                        "hidden_dim": h_dim
                    }
                    save_master_dataset_config(config)
                    st.success("Configuración global guardada para todos los idiomas.")
                    st.rerun()
                    
        with col_c2:
            st.markdown("#### Topología Base")
            st.graphviz_chart(get_recognizer_diagram())

        with st.expander("💻 Ver Código del Modelo Base"):
            st.code(get_function_source(TinyEyes), language="python")

if __name__ == "__main__":
    main()
