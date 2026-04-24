"""
📊 Resultados Globales - Visor Translingüístico Premium
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import pickle
import numpy as np

from components.modern_sidebar import display_modern_sidebar
from components.analytics import (
    plot_training_history, 
    plot_confusion_matrix, 
    display_classification_report, 
    plot_latent_space_pca
)
from utils.checkpoints import list_checkpoints

def format_accuracy(val):
    """Asegura que el accuracy esté en rango 0-1 antes de aplicar .2%"""
    if val > 1.0:
        return val / 100.0
    return val

st.set_page_config(
    page_title="Resultados Globales - TinySpeak",
    page_icon="📊",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #6B66FF, #DF66FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(107, 102, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(107, 102, 255, 0.3);
        margin-bottom: 1rem;
    }
    .sample-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 5px;
        border-left: 3px solid #6B66FF;
    }
    </style>
    """

def get_active_models(domain_key: str):
    ckpts = list_checkpoints(domain_key)
    active = {}
    for c in ckpts:
        lang = c['meta'].get('config', {}).get('language')
        if lang and lang not in active:
            active[lang] = c
    return dict(sorted(active.items()))

def render_language_column(lang, ckpt_info):
    st.markdown(f"## 🌍 {lang.upper()}")
    
    # 1. Métricas Rápidas
    meta = ckpt_info.get('meta', {})
    metrics = meta.get('metrics', {})
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Val Loss", f"{metrics.get('val_loss', 0.0):.4f}")
    with col_m2:
        # Intentar obtener precisión (depende del tipo de modelo)
        acc_raw = metrics.get('val_word_acc') or metrics.get('val_phoneme_acc') or metrics.get('val_top1') or metrics.get('val_acc') or 0.0
        st.metric("Accuracy", f"{format_accuracy(acc_raw):.2%}")

    # 2. Historial
    st.markdown("#### 📈 Historial")
    plot_training_history(meta.get('history', []))
    
    # 3. Evaluación Detallada
    try:
        eval_path = Path(ckpt_info['path']).parent / "eval_results.pkl"
        if eval_path.exists():
            with open(eval_path, "rb") as f:
                data = pickle.load(f)
            
            # Matriz y Reporte
            conf = data.get("confusion", {})
            if conf and conf.get("y_true"):
                st.markdown("#### 🎯 Clasificación")
                plot_confusion_matrix(conf["y_true"], conf["y_pred"], conf["class_names"])
                display_classification_report(conf["y_true"], conf["y_pred"], conf["class_names"])
            
            # PCA
            embs = data.get("embeddings", [])
            labels = data.get("labels", [])
            if len(embs) > 0:
                st.markdown("#### 🌌 Espacio Latente")
                # Fallback robusto para nombres de clases para el PCA
                classes_to_use = conf.get("class_names") or meta.get('config', {}).get('classes') or []
                if classes_to_use:
                    plot_latent_space_pca(embs, labels, classes_to_use)
                else:
                    st.warning("No se encontraron nombres de clases para el PCA.")
            
            # Muestras
            samples = data.get("samples", [])
            if samples:
                st.markdown("#### 📄 Muestras Reales")
                for s in samples[:5]: # Mostrar solo 5 en vista global para no saturar
                    st.markdown(f"""
                    <div class="sample-card">
                        <b>Target:</b> {s['target']}<br>
                        <b>Pred:</b> {s['prediction']}<br>
                        <small>Conf: {s.get('confidence', 0.0):.2%}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Falta archivo de evaluación.")
    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("resultados_globales")
    
    st.markdown('<h1 class="main-header">📊 Dashboard de Resultados Globales</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs([
        "👂 TinyEars Phonemes", 
        "🧠 TinyEars Words", 
        "👁️ TinyEyes",
        "✍️ TinySpeller (G2P)",
        "📖 TinyReader (P2W)"
    ])
    
    domains = [
        "tiny_ears_phonemes",
        "tiny_ears_words",
        "tiny_eyes",
        "tiny_speller",
        "tiny_reader"
    ]
    
    for tab, domain in zip(tabs, domains):
        with tab:
            active = get_active_models(domain)
            if not active:
                st.info(f"No hay modelos entrenados para {domain}.")
                continue
                
            cols = st.columns(len(active))
            for i, (lang, info) in enumerate(active.items()):
                with cols[i]:
                    render_language_column(lang, info)

if __name__ == "__main__":
    main()
