"""
📖 TinySpeller - Ensamblaje Fonológico (Stage 1 G2P)
"""

import streamlit as st
import pytorch_lightning as pl
from pathlib import Path
import torch
import pandas as pd
import json
import numpy as np

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_reader_diagram, get_latent_mapping_diagram
from components.code_viewer import get_function_source
from components.analytics import plot_learning_curves, plot_confusion_matrix, plot_dtw_alignment
from models.tiny_speller import TinySpeller
from training.reader_dataset import build_reader_dataloaders
from training.reader_module import TinyReaderLightning
from training.config import load_master_dataset_config, save_master_dataset_config
from utils.device import encontrar_device
from utils.graphemes import get_default_words
from utils.checkpoints import list_checkpoints

# Configurar página
st.set_page_config(
    page_title="TinySpeller - Stage 1",
    page_icon="📖",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #bb66ff, #ff66d9);
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
        border-left: 5px solid #bb66ff;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """

def get_active_models():
    ckpts = list_checkpoints("tiny_speller")
    active = {}
    for c in ckpts:
        lang = c['meta'].get('config', {}).get('language')
        if lang and lang not in active:
            active[lang] = c
    return active

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("tiny_speller")
    
    st.markdown('<h1 class="main-header">📖 TinySpeller: Conversión Grafema-Fonema (G2P)</h1>', unsafe_allow_html=True)
    
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
                        metrics = meta.get('metrics', {})
                        st.json({
                            "Épocas": meta.get('config', {}).get('epochs'),
                            "Val MSE": round(metrics.get('val_g2p_structural_mse', metrics.get('val_loss', 0.0)), 5),
                            "Val CE": round(metrics.get('val_g2p_categorical_ce', 0.0), 4),
                            "Actualizado": ckpt.get('date', 'Desconocido')
                        })
                    else:
                        st.warning("⚠️ Pendiente de entrenar")

        st.divider()
        st.markdown("### ⚙️ Iniciar Entrenamiento")
        train_config = config.get("training_params", {}).get("tiny_speller", {})
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Épocas", min_value=1, max_value=1000, value=train_config.get("epochs", 50))
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=train_config.get("batch_size", 16))
        with col2:
            lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=train_config.get("lr", 1e-3), format="%.5f")
            w_mse = st.slider("Peso de Alineación Estructural (MSE)", 0.0, 2.0, train_config.get("w_mse", 1.0), 0.1)
            w_perceptual = st.slider("Peso Categórico (Cross-Entropy)", 0.0, 2.0, train_config.get("w_perceptual", 0.5), 0.1)
            
        if st.button("🚀 Iniciar Entrenamiento por Lotes (Todos los Idiomas)", type="primary"):
            if "training_params" not in config:
                config["training_params"] = {}
            config["training_params"]["tiny_speller"] = {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "w_mse": w_mse,
                "w_perceptual": w_perceptual
            }
            save_master_dataset_config(config)
            
            from training.runner import train_reader
            
            st.markdown("### 📈 Progreso de Entrenamiento...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            plots_container = st.container()
            prediction_placeholder = st.empty()

            for i, lang in enumerate(languages):
                status_text.markdown(f"**Entrenando TinySpeller (G2P) para {lang.upper()}... ({i+1}/{len(languages)})**")
                
                phoneme_data = config.get('phoneme_samples', {}).get(lang, {})
                if not phoneme_data:
                    st.warning(f"Saltando {lang}: No se encontraron datos de fonemas.")
                    continue
                
                rec_path = f"data/checkpoints/tiny_eyes/{lang}/best_model.ckpt"
                lis_path = f"data/checkpoints/tiny_ears_phonemes/{lang}/best_model.ckpt"
                
                if not Path(rec_path).exists() or not Path(lis_path).exists():
                    st.error(f"❌ Faltan dependencias para {lang} (TinyEyes o TinyEars Fonemas).")
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
                    "batch_size": batch_size,
                    "w_dtw": 0.0,
                    "w_perceptual": w_perceptual,
                    "use_two_stage": True,
                    "training_phase": "g2p"
                }
                
                try:
                    ckpt_path, hist = train_reader(
                        lang, 
                        listener_ckpt=lis_path, 
                        recognizer_ckpt=rec_path, 
                        config=train_conf, 
                        plot_placeholders=plot_placeholders,
                        prediction_placeholder=prediction_placeholder
                    )
                    st.success(f"✅ {lang.upper()} guardado: `{ckpt_path}`")
                except Exception as e:
                    st.error(f"❌ Error en {lang}: {e}")
                
                progress_bar.progress((i + 1) / len(languages))
            
            st.success("🎉 Entrenamientos completados. Por favor refresca la página.")

    # ==========================================
    # TAB 2: RESULTADOS GLOBALES E HISTORIAL
    # ==========================================
    with tabs[1]:
        st.markdown("### 🧪 Historial y Resultados Generales")
        if not active_models:
            st.warning("Debes entrenar los modelos primero.")
        else:
            from components.analytics import plot_training_history
            
            st.markdown("#### Progreso de Entrenamiento Registrado")
            hist_cols = st.columns(len(active_models))
            for i, (lang_eval, ckpt_info) in enumerate(active_models.items()):
                with hist_cols[i]:
                    st.markdown(f"**🌍 {lang_eval.upper()} - Ciclo de Entrenamiento**")
                    history_data = ckpt_info.get('meta', {}).get('history', [])
                    plot_training_history(history_data)
                    
            st.divider()
            
            st.markdown("#### Evaluación Empírica de Validación")
            st.markdown("#### 🌍 Evaluación Empírica por Idioma")
            eval_cols = st.columns(len(active_models))
            
            import pickle
            for i, (lang_eval, ckpt_info) in enumerate(active_models.items()):
                with eval_cols[i]:
                    st.markdown(f"### 🌍 {lang_eval.upper()}")
                    try:
                        eval_path = Path(ckpt_info['path']).parent / "eval_results.pkl"
                        
                        if eval_path.exists():
                            with open(eval_path, "rb") as f:
                                eval_data_raw = pickle.load(f)
                            
                            # Cargar datos estandarizados
                            samples = eval_data_raw.get("samples", [])
                            confusion = eval_data_raw.get("confusion", {})
                            embeddings = eval_data_raw.get("embeddings", [])
                            labels = eval_data_raw.get("labels", [])
                            
                            st.success(f"Evaluación {lang_eval.upper()} cargada.")
                            
                            # 1. Matriz de Confusión
                            if confusion.get("y_true"):
                                st.markdown("#### 🎯 Matriz de Confusión (Fonemas)")
                                plot_confusion_matrix(
                                    confusion["y_true"], 
                                    confusion["y_pred"], 
                                    confusion["class_names"]
                                )
                                
                                # 2. Reporte de Clasificación
                                st.markdown("#### 📋 Métricas de Clasificación")
                                from components.analytics import display_classification_report
                                display_classification_report(
                                    confusion["y_true"], 
                                    confusion["y_pred"], 
                                    confusion["class_names"]
                                )

                            # 3. PCA (Espacio Latente)
                            if len(embeddings) > 0:
                                st.markdown("#### 🌌 Espacio Latente (PCA 3D)")
                                from components.analytics import plot_latent_space_pca
                                # Usar solo una muestra si hay demasiados puntos
                                n_pca = min(len(embeddings), 500)
                                plot_latent_space_pca(
                                    np.array(embeddings)[:n_pca], 
                                    np.array(labels)[:n_pca], 
                                    confusion.get("class_names", [])
                                )
                            
                            # 4. Muestras Individuales
                            st.markdown("#### 📖 Muestras de Predicción")
                            for item in samples:
                                word = item.get("word") or item.get("label_str")
                                pred = item.get("prediction")
                                target = item.get("target")
                                
                                with st.expander(f"Muestra: {word}"):
                                    st.write(f"**Target:** {target}")
                                    st.write(f"**Pred:** {pred}")
                                    conf = item.get("confidence", 0)
                                    st.progress(conf, text=f"Confianza: {conf:.2%}")
                        else:
                            st.warning(f"⚠️ Sin evaluación para {lang_eval}.")
                    except Exception as e:
                        st.error(f"Error en {lang_eval}: {e}")

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
            
            if st.button(f"🎧 Sintetizar Fonemas de Prueba G2P para {lang_lab.upper()}", type="primary"):
                with st.spinner("Decodificando..."):
                    try:
                        meta_config = ckpt_lab.get('meta', {}).get('config', {})
                        model_hparams = {k: v for k, v in meta_config.items() if k in ["hidden_dim", "output_dim", "num_layers"]}
                        if not model_hparams:
                            model_hparams = config.get("architectures", {}).get("tiny_speller", {})
                        
                        model = TinyReaderLightning.load_from_checkpoint(
                            ckpt_lab['path'], 
                            strict=False,
                            **model_hparams
                        )
                        model.eval()
                        device = encontrar_device()
                        model.to(device)
                        
                        _, _, _, loaders = build_reader_dataloaders(
                            batch_size=32, num_workers=0, seed=42, target_language=lang_lab, use_phoneme_targets=True
                        )
                        
                        val_ds = loaders['val'].dataset
                        # Obtener todas las palabras disponibles en el set de validación
                        available_words = sorted(list(set([val_ds[i]["label_str"] for i in range(len(val_ds))])))
                        
                        target_word = st.selectbox("Seleccionar palabra a predecir", available_words, key=f"sel_06_{lang_lab}")
                        
                        found_item = None
                        for i in range(len(val_ds)):
                            item = val_ds[i]
                            if item["label_str"] == target_word:
                                found_item = item
                                break
                                
                        if found_item is not None:
                            images = found_item["image"].to(device).unsqueeze(0)
                            target_audios = found_item["waveform"].to(device)
                            label = found_item["label_str"]
                            
                            pred_audio = model(images)
                            
                            st.markdown(f"#### Palabra Base Generadora: `{label}`")
                            fig = plot_dtw_alignment(pred_audio[0], target_audios)
                            st.pyplot(fig)
                            
                            sr = 16000
                            st.markdown("##### Fonética Sintetizada G2P")
                            st.audio(pred_audio[0].cpu().numpy(), sample_rate=sr)
                            st.markdown("##### Fonética Original de Referencia")
                            st.audio(target_audios.cpu().numpy(), sample_rate=sr)
                        else:
                            st.warning("No se encontraron muestras en validación.")
                        
                    except Exception as e:
                        st.error(f"Error cargando instancia: {e}")

    # ==========================================
    # TAB 4: ARQUITECTURA
    # ==========================================
    with tabs[3]:
        st.markdown("### 📐 Arquitectura de la Red: TinySpeller (G2P)")
        st.info("La arquitectura configurada aquí mapea activaciones visuales IT (512-dim) a imágenes neurales auditivas.")
        
        arch_config = config.get("architectures", {}).get("tiny_speller", {})
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Parámetros Técnicos (MLP Projector)")
            with st.form("arch_form_tiny_speller"):
                h_dim = st.number_input("Dimensión Oculta (hidden_dim)", min_value=64, max_value=2048, value=arch_config.get("hidden_dim", 256), step=64)
                o_dim = st.number_input("Dimensión Salida Auditiva (output_dim)", min_value=64, max_value=2048, value=arch_config.get("output_dim", 256), step=64)
                n_layers = st.number_input("Número de Capas Densas (MLP)", min_value=1, max_value=10, value=arch_config.get("num_layers", 2))
                
                if st.form_submit_button("💾 Guardar Configuración", type="primary"):
                    if "architectures" not in config:
                        config["architectures"] = {}
                    config["architectures"]["tiny_speller"] = {
                        "hidden_dim": h_dim,
                        "output_dim": o_dim,
                        "num_layers": n_layers
                    }
                    save_master_dataset_config(config)
                    st.success("Configuración del Proyector Puntual guardada.")
                    st.rerun()
                    
        with col_c2:
            st.markdown("#### 🧠 Teoría: Mapeo Neural Directo (Pointwise)")
            st.markdown("""
            Esta red implementa una **transducción sensorial directa**:
            
            1. **Entrada (IT Cortex)**: Recibe vectores de 512-dim que representan la abstracción visual del grafema.
            2. **Proyección Puntual**: Cada frame se procesa de forma independiente mediante capas MLP (Linear + GELU + LayerNorm), asegurando que el sonido generado dependa solo de la letra vista.
            3. **Salida (Auditorio)**: Genera la imagen neural auditiva que el oído fonético reconoce como el sonido correspondiente.
            
            Este enfoque evita la fuga de información temporal, obligando al sistema a aprender una **decodificación fonológica pura**.
            """)
            st.graphviz_chart(get_latent_mapping_diagram(module_name="TinySpeller", stage="G2P"))

        with st.expander("💻 Ver Código del Modelo Base"):
            st.code(get_function_source(TinySpeller), language="python")

if __name__ == "__main__":
    main()
