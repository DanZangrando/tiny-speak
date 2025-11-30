"""
👂 TinyEars (Words) - Entrenamiento del Oído para Palabras
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
    page_title="TinyEars - Palabras",
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
    display_modern_sidebar("tiny_ears_words")
    
    st.markdown('<h1 class="main-header">👂 TinyEars: Reconocimiento de Palabras</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("""
        ### 👂 TinyEars: Reconocimiento de Palabras (Lexical Access)

        #### 1. Evolución de la Arquitectura: De Wav2Vec 2.0 a TinyEars
        Al igual que en la vía fonémica, hemos migrado de modelos masivos como **Wav2Vec 2.0** a una arquitectura especializada.
        
        **TinyEars (Words)** se enfoca en el acceso léxico:
        *   **Segmentación y Reconocimiento:** En lugar de depender de representaciones pre-computadas, el modelo aprende a segmentar el flujo continuo de audio en unidades léxicas (palabras) desde cero.
        *   **Transparencia del Lexicón:** Esta arquitectura nos permite visualizar exactamente qué patrones temporales y espectrales activan cada "entrada léxica" en la capa final, ofreciendo una ventana clara al proceso de reconocimiento de palabras.

        #### 2. Arquitectura Cognitiva
        La arquitectura es idéntica a la vía fonológica (Wav2Vec 2.0 Tiny), pero entrenada con un objetivo diferente:
        
        *   **Input (Cóclea):** Espectrograma Mel.
        *   **Feature Extractor (Tronco Encefálico):** Extracción de rasgos acústicos.
        *   **Context Network (Corteza Auditiva Superior):** Integración temporal para formar representaciones de palabras completas.
        *   **Clasificador (Lexicón):** Capa final que mapea la representación auditiva a una entrada léxica específica (palabra).

        #### 3. Input/Output
        *   **Entrada:** Audio de palabra hablada.
        *   **Salida:** Identidad de la palabra (Clase Léxica).
        """)
        
        st.graphviz_chart(get_listener_diagram())

    # ==========================================
    # TAB 2: ENTRENAMIENTO
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Entrenamiento")
        
        config = load_master_dataset_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hiperparámetros")
            epochs = st.number_input("Épocas", min_value=1, max_value=1000, value=50)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
            lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
            
        with col2:
            st.markdown("#### Dataset")
            target_language = st.selectbox("Idioma Objetivo", config.get('experiment_config', {}).get('languages', ['es']))
            
            # Mostrar palabras/fonemas disponibles
            # Para TinyEars Words, buscamos en generated_samples (audio)
            audio_data = config.get('generated_samples', {}).get(target_language, {})
            if audio_data:
                # Filtrar vacíos y ordenar para coincidir con el dataset
                words = sorted([w for w, s in audio_data.items() if s])
            else:
                words = []
                
            st.info(f"Entrenando sobre {len(words)} palabras.")
            with st.expander("Ver Vocabulario (Palabras)"):
                st.write(words)

        if st.button("🚀 Iniciar Entrenamiento de Palabras", type="primary"):
            # Setup
            pl.seed_everything(42)
            
            # 1. Construir Dataloaders PRIMERO para obtener el vocabulario real
            try:
                train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
                    batch_size=batch_size,
                    target_language=target_language,
                    num_workers=4,
                    seed=42,
                    use_phonemes=False
                )
                
                # Actualizar words con lo que realmente hay en el dataset
                words = train_ds.class_names
                st.success(f"Dataset cargado con {len(words)} palabras válidas.")
                
            except Exception as e:
                st.error(f"Error cargando datos: {e}")
                st.stop()
                
            # 2. Inicializar Modelo con el vocabulario CORRECTO
            if not words:
                 st.error("⚠️ No se encontraron palabras válidas en el dataset.")
                 st.stop()
                 
            model = PhonologicalPathwayLightning(
                class_names=words,
                learning_rate=lr
            )
            
            # Callbacks
            history_cb = ListenerHistoryCallback()
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode="min"
            )
            
            checkpoint_callback = ModelCheckpoint(
                dirpath="models/listener_checkpoints",
                filename="word_listener-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min"
            )
            
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator="auto",
                devices=1,
                callbacks=[history_cb, early_stop_callback, checkpoint_callback],
                enable_progress_bar=True,
                default_root_dir="lightning_logs/tiny_ears_words"
            )
            
            # Placeholders
            st.markdown("### 📈 Progreso")
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                plot_loss = st.empty()
            with col_plot2:
                plot_acc = st.empty()
                
            realtime_cb = RealTimePlotCallback(plot_loss, plot_acc)
            trainer.callbacks.append(realtime_cb)
            
            with st.spinner("Entrenando Oído para Palabras..."):
                trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
                
            st.success("Entrenamiento completado!")
            
            # Guardar
            save_dir = Path("models/listener")
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            final_path = save_dir / f"word_listener_{target_language}_{timestamp}.ckpt"
            trainer.save_checkpoint(final_path)
            
            meta_config = {
                "epochs": epochs, "lr": lr, "batch_size": batch_size,
                "language": target_language,
                "vocab": words,
                "type": "word"
            }
            final_metrics = history_cb.history[-1] if history_cb.history else {}
            save_model_metadata(final_path, meta_config, final_metrics)
            
            if history_cb.history:
                pd.DataFrame(history_cb.history).to_csv(final_path.with_suffix(".csv"), index=False)
                
            st.info(f"Modelo guardado en {final_path}")

    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📚 Modelos de Palabras")
        all_ckpts = list_checkpoints("listener")
        # Filtrar solo los que parecen de palabras
        word_ckpts = [c for c in all_ckpts if "word" in c['filename'] or c.get('meta', {}).get('config', {}).get('type') == 'word']
        
        if not word_ckpts:
            st.info("No hay modelos de palabras entrenados.")
        else:
            opts = {c['filename']: c for c in word_ckpts}
            sel_key = st.selectbox("Seleccionar Modelo", list(opts.keys()))
            sel_ckpt = opts[sel_key]
            
            col_info, col_actions = st.columns([3, 1])
            with col_info:
                st.markdown(f"**Archivo:** `{sel_ckpt['filename']}`")
                st.json(sel_ckpt.get('meta', {}))
                
            with col_actions:
                if st.button("🗑️ Eliminar", key="del_wd"):
                    Path(sel_ckpt['path']).unlink(missing_ok=True)
                    st.rerun()

            st.divider()
            st.markdown("### 🧪 Evaluación Profunda")
            if st.button("🚀 Ejecutar Evaluación Completa", key="eval_wd"):
                with st.spinner("Cargando modelo y datos..."):
                    try:
                        # 1. Cargar Modelo
                        meta_path = Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json")
                        class_names = []
                        target_lang = None
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta = json.load(f)
                                config = meta.get('config', {})
                                class_names = config.get('vocab', [])
                                target_lang = config.get('language')
                        
                        if not class_names:
                            st.warning("No se encontraron class_names en metadata. Usando default.")
                            class_names = get_default_words()

                        model = PhonologicalPathwayLightning.load_from_checkpoint(
                            sel_ckpt['path'],
                            class_names=class_names
                        )
                        model.eval()
                        device = encontrar_device()
                        model.to(device)
                        
                        # 2. Cargar Datos (Validation Set)
                        if not target_lang:
                            target_lang = 'es' # Fallback
                            
                        _, _, _, loaders = build_audio_dataloaders(
                            batch_size=32,
                            target_language=target_lang,
                            num_workers=0,
                            use_phonemes=False, # Words!
                            seed=42
                        )
                        val_loader = loaders['val']
                        
                        # 3. Inferencia
                        all_preds = []
                        all_labels = []
                        all_embeddings = []
                        
                        with torch.no_grad():
                            for batch in val_loader:
                                waveforms = [w.to(device) for w in batch["waveforms"]]
                                labels = batch["label"].to(device)
                                
                                # Forward
                                # PhonologicalPathwayLightning.forward solo devuelve logits.
                                # Accedemos al modelo interno para obtener embeddings también.
                                if isinstance(waveforms, list):
                                    from torch.nn.utils.rnn import pad_sequence
                                    waveforms_padded = pad_sequence(waveforms, batch_first=True).to(device)
                                else:
                                    waveforms_padded = waveforms

                                logits, embeddings = model.model(waveforms_padded)
                                
                                preds = torch.argmax(logits, dim=1)
                                
                                all_preds.extend(preds.cpu().numpy())
                                all_labels.extend(labels.cpu().numpy())
                                
                                # Pooling de embeddings para PCA (Mean over time)
                                pooled_emb = embeddings.mean(dim=1)
                                all_embeddings.extend(pooled_emb.cpu().numpy())
                                
                        # 4. Visualización
                        st.success("Evaluación completada.")
                        
                        # Matriz de Confusión
                        st.markdown("#### Matriz de Confusión")
                        plot_confusion_matrix(all_labels, all_preds, class_names)
                        
                        # Reporte
                        st.markdown("#### Reporte de Clasificación")
                        display_classification_report(all_labels, all_preds, class_names)
                        
                        # PCA
                        st.markdown("#### Espacio Latente (PCA)")
                        
                        from components.analytics import plot_latent_space_pca
                        plot_latent_space_pca(np.array(all_embeddings), all_labels, class_names)
                        
                    except Exception as e:
                        st.error(f"Error en evaluación: {e}")
                        st.exception(e)

    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    with tabs[3]:
        st.markdown("### 🧪 Laboratorio de Palabras")
        
        # 1. Seleccionar Modelo
        all_ckpts = list_checkpoints("listener")
        word_ckpts = [c for c in all_ckpts if "word" in c['filename'] or c.get('meta', {}).get('config', {}).get('type') == 'word']
        
        if not word_ckpts:
            st.warning("Entrena un modelo primero.")
        else:
            opts = {c['filename']: c['path'] for c in word_ckpts}
            sel_model_name = st.selectbox("Modelo para Inferencia", list(opts.keys()), key="lab_model_sel")
            
            if st.button("Cargar Modelo", key="load_model_lab"):
                ckpt_path = opts[sel_model_name]
                with st.spinner("Cargando modelo..."):
                    try:
                        meta_path = Path(ckpt_path).with_suffix(".ckpt.meta.json")
                        class_names = []
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta = json.load(f)
                                class_names = meta.get('config', {}).get('vocab', [])
                        
                        if not class_names:
                            class_names = get_default_words()
                            
                        model = PhonologicalPathwayLightning.load_from_checkpoint(
                            ckpt_path,
                            class_names=class_names
                        )
                        model.eval()
                        st.session_state['wd_lab_model'] = model
                        st.success(f"Modelo cargado: {sel_model_name}")
                    except Exception as e:
                        st.error(f"Error cargando modelo: {e}")

            if 'wd_lab_model' in st.session_state:
                model = st.session_state['wd_lab_model']
                device = next(model.parameters()).device
                
                st.divider()
                st.markdown("#### 🎤 Prueba Interactiva")
                
                input_method = st.radio("Método de Entrada", ["Muestra del Dataset", "Subir Archivo WAV"])
                
                waveform = None
                sample_rate = 16000
                label_text = "Desconocido"
                
                if input_method == "Muestra del Dataset":
                    # Cargar dataset de validación
                    if 'val_dataset_words' not in st.session_state:
                         _, val_ds, _, _ = build_audio_dataloaders(batch_size=1, target_language='es', use_phonemes=False, seed=42, num_workers=0)
                         st.session_state['val_dataset_words'] = val_ds
                    
                    val_ds = st.session_state['val_dataset_words']
                    
                    # Selector de clase
                    selected_class = st.selectbox("Selecciona una Palabra", model.class_names)
                    
                    # Filtrar muestras de esa clase
                    class_samples = [s for s in val_ds.samples if s.word == selected_class]
                    
                    if not class_samples:
                        st.warning(f"No hay muestras de validación para la palabra '{selected_class}'.")
                    else:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if st.button("🎲 Cargar Muestra", type="primary"):
                                import random
                                sample = random.choice(class_samples)
                                waveform = sample.waveform
                                label_text = sample.word
                                st.session_state['word_lab_waveform'] = waveform
                                st.session_state['word_lab_label'] = label_text
                        
                        with col2:
                            st.caption(f"Disponibles: {len(class_samples)} muestras")

                    if 'word_lab_waveform' in st.session_state:
                        waveform = st.session_state['word_lab_waveform']
                        label_text = st.session_state['word_lab_label']
                        st.info(f"Muestra cargada: **{label_text}**")
                        
                else:
                    uploaded_file = st.file_uploader("Sube un archivo WAV", type=["wav"])
                    if uploaded_file:
                        import torchaudio
                        wf, sr = torchaudio.load(uploaded_file)
                        if sr != 16000:
                            wf = torchaudio.transforms.Resample(sr, 16000)(wf)
                        if wf.shape[0] > 1:
                            wf = wf.mean(dim=0, keepdim=True)
                        waveform = wf
                        st.audio(uploaded_file, format='audio/wav')

                if waveform is not None:
                    # Visualizar
                    st.markdown("##### Espectrograma")
                    import torchaudio.transforms as T
                    spec_transform = T.MelSpectrogram(sample_rate=16000, n_mels=80)
                    spec = (spec_transform(waveform) + 1e-9).log2()
                    spec_np = spec.numpy()
                    spec_norm = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min() + 1e-6)
                    # Invertir eje Y
                    spec_norm = np.flipud(spec_norm)
                    st.image(spec_norm, caption="Mel Spectrogram", use_container_width=True)
                    
                    if st.button("🧠 Analizar Palabra", type="primary"):
                        try:
                            with torch.no_grad():
                                wf_in = waveform.to(device)
                                
                                logits = model([wf_in])
                                probs = torch.softmax(logits, dim=1)
                                
                                top_probs, top_idxs = torch.topk(probs, 5, dim=1)
                                
                                st.markdown("### 🎯 Predicciones")
                                results = []
                                for i in range(5):
                                    idx = top_idxs[0, i].item()
                                    prob = top_probs[0, i].item()
                                    cls_name = model.class_names[idx] if idx < len(model.class_names) else f"Unknown({idx})"
                                    results.append({"Palabra": cls_name, "Confianza": f"{prob:.2%}"})
                                    
                                st.table(results)
                                
                                top_pred = results[0]["Palabra"]
                                if label_text != "Desconocido":
                                    if top_pred == label_text:
                                        st.balloons()
                                        st.success(f"¡Correcto! Predicción: {top_pred}")
                                    else:
                                        st.error(f"Incorrecto. Predicción: {top_pred} vs Real: {label_text}")
                                else:
                                    st.info(f"Predicción Principal: **{top_pred}**")
                                    
                                # Gráfica de barras
                                chart_data = pd.DataFrame({
                                    "Palabra": [r["Palabra"] for r in results],
                                    "Probabilidad": top_probs[0].cpu().numpy()
                                })
                                st.bar_chart(chart_data.set_index("Palabra"))
                                
                        except Exception as e:
                            st.error(f"Error en análisis: {e}")

if __name__ == "__main__":
    main()
