"""
🧠 TinySpeller (Stage 1) - Aprendiendo a Deletrear (G2P)
"""

import streamlit as st
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import torch
import pandas as pd
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_tiny_reader_diagram
from components.code_viewer import get_function_source
from models import TinyReader
from training.reader_module import TinyReaderLightning
from training.audio_dataset import build_audio_dataloaders, DEFAULT_AUDIO_SPLIT_RATIOS
from training.config import load_master_dataset_config
from utils import list_checkpoints, encontrar_device, save_model_metadata, RealTimePlotCallback, ReaderPredictionCallback, get_phonemes_from_word

# Configurar página
st.set_page_config(
    page_title="TinySpeller - Stage 1",
    page_icon="🧠",
    layout="wide"
)

def get_custom_css():
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #556270);
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
        border-left: 5px solid #FF6B6B;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """

class ReaderHistoryCallback(pl.Callback):
    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in trainer.callback_metrics.items()}
        metrics['epoch'] = trainer.current_epoch
        self.history.append(metrics)

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("tiny_speller")
    
    st.markdown('<h1 class="main-header">🧠 TinySpeller: Aprendiendo a Deletrear (Stage 1)</h1>', unsafe_allow_html=True)

    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("### 🔤 Stage 1: Grapheme-to-Phoneme (G2P)")
        
        st.markdown("""
        ### 🧠 TinySpeller: La Ruta Subléxica (Grapheme-to-Phoneme)

        #### 1. Filosofía de Diseño: Codificación Predictiva
        TinySpeller no aprende a clasificar fonemas directamente (como un clasificador softmax tradicional). En su lugar, aprende a **"imaginar"** cómo suenan las letras.
        
        *   **Perceptual Loss (El "Juez Auditivo"):** Usamos el modelo *TinyEars (Phonemes)* congelado como juez. TinySpeller debe generar vectores que activen las neuronas de TinyEars de la misma manera que lo haría el audio real.
        *   **Justificación Neurocientífica:** Esto simula el aprendizaje mediante la minimización del error de predicción entre las áreas visuales (VWFA) y auditivas. El cerebro aprende a asociar grafemas con fonemas porque la activación visual predice exitosamente la activación auditiva.

        #### 2. Arquitectura Cognitiva
        Implementamos un modelo **Seq2Seq (Encoder-Decoder)** que permite mapeos flexibles (N letras -> M fonemas):
        
        *   **Encoder (VWFA):** Procesa la secuencia visual de letras (embeddings visuales de TinyEyes).
        *   **Decoder (Imaginación):** Genera secuencialmente representaciones en el espacio latente auditivo.
        *   **Mecanismo de Atención (Opcional):** Permite al modelo "mirar" partes específicas de la palabra mientras la deletrea mentalmente.

        #### 3. Input/Output
        *   **Entrada:** Secuencia de Embeddings Visuales (de TinyEyes).
        *   **Salida:** Secuencia de Embeddings Auditivos (compatibles con TinyEars).
        """)
        
        # Usamos el diagrama completo pero nos enfocamos en el Stage 1
        st.graphviz_chart(get_tiny_reader_diagram())

    # ==========================================
    # TAB 2: ENTRENAMIENTO
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Entrenamiento (G2P)")
        
        config = load_master_dataset_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hiperparámetros")
            epochs = st.number_input("Épocas", min_value=1, max_value=1000, value=50)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
            lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
            
            st.markdown("#### Pesos de Loss")
            w_dtw = st.slider("Peso Soft-DTW (Alineación)", 0.0, 2.0, 1.0, 0.1)
            w_perceptual = st.slider("Peso Perceptual (Feature Matching)", 0.0, 2.0, 0.5, 0.1)
            
        with col2:
            st.markdown("#### Componentes Previos")
            
            # 1. Recognizer (Visual)
            rec_ckpts = list_checkpoints("recognizer")
            if not rec_ckpts:
                st.error("❌ No hay modelos TinyEyes (Visual) disponibles.")
                st.stop()
            rec_opts = {c['filename']: c['path'] for c in rec_ckpts}
            sel_rec = st.selectbox("TinyEyes (Visual)", list(rec_opts.keys()))
            
            # 2. Listener (Phonemes)
            lis_ckpts = list_checkpoints("listener")
            # Filtrar fonemas
            phoneme_ckpts = [c for c in lis_ckpts if "phoneme" in c['filename'] or c.get('meta', {}).get('config', {}).get('type') == 'phoneme']
            
            if not phoneme_ckpts:
                st.error("❌ No hay modelos TinyEars (Fonemas) disponibles. Entrena uno en la página 'TinyEars - Fonemas'.")
                st.stop()
                
            lis_opts = {c['filename']: c['path'] for c in phoneme_ckpts}
            sel_lis = st.selectbox("TinyEars (Juez Fonémico)", list(lis_opts.keys()))
            
            # Dataset
            target_language = st.selectbox("Idioma Objetivo", config.get('experiment_config', {}).get('languages', ['es']))
            
            # Para G2P, el output son fonemas, así que el vocabulario debe ser de fonemas
            phoneme_data = config.get('phoneme_samples', {}).get(target_language, {})
            if phoneme_data:
                # Filtrar vacíos y ordenar para coincidir con el dataset
                words = sorted([w for w, s in phoneme_data.items() if s])
            else:
                words = []
                
            st.info(f"Entrenando sobre {len(words)} fonemas.")

        if st.button("🚀 Iniciar Entrenamiento TinySpeller", type="primary"):
            # Setup
            pl.seed_everything(42)
            
            # 1. Construir Dataloaders PRIMERO
            try:
                train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
                    batch_size=batch_size,
                    target_language=target_language,
                    num_workers=4,
                    seed=42,
                    use_phonemes=True
                )
                
                words = train_ds.class_names
                st.success(f"Dataset cargado con {len(words)} ítems.")
                
            except Exception as e:
                st.error(f"Error cargando datos: {e}")
                st.stop()
                
            # 2. Inicializar Modelo
            if not words:
                 st.error("⚠️ No se encontraron datos válidos en el dataset.")
                 st.stop()
                 
            # Cargar checkpoints de dependencias
            if not lis_opts[sel_lis] or not rec_opts[sel_rec]:
                st.error("Se requieren checkpoints de TinyEars (Phonemes) y TinyEyes (Visual).")
                st.stop()
                
            model = TinyReaderLightning(
                class_names=words,
                listener_checkpoint_path=lis_opts[sel_lis], # Phoneme Listener
                recognizer_checkpoint_path=rec_opts[sel_rec],
                learning_rate=lr,
                w_dtw=w_dtw,
                w_perceptual=w_perceptual,
                use_two_stage=True, # Siempre True para usar la arquitectura modular
                training_phase="g2p", # FASE G2P
                phoneme_listener_checkpoint_path=lis_opts[sel_lis] # Pasamos el mismo como phoneme listener
            )
            
            # Callbacks
            history_cb = ReaderHistoryCallback()
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode="min"
            )
            
            checkpoint_callback = ModelCheckpoint(
                dirpath="models/reader_checkpoints",
                filename="tiny_speller-{epoch:02d}-{val_loss:.2f}",
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
                default_root_dir="lightning_logs/tiny_speller"
            )
            
            # Placeholders
            st.markdown("### 📈 Progreso")
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                st.markdown("#### Pérdida (Loss)")
                plot_loss = st.empty()
            with col_plot2:
                st.markdown("#### Precisión (Fonemas)")
                plot_acc = st.empty()
                
            realtime_cb = RealTimePlotCallback(plot_loss, plot_acc)
            trainer.callbacks.append(realtime_cb)
            
            # Prediction Callback
            pred_placeholder = st.empty()
            pred_cb = ReaderPredictionCallback(loaders['val'], pred_placeholder)
            trainer.callbacks.append(pred_cb)
            
            with st.spinner("Entrenando TinySpeller (G2P)..."):
                trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
                
            st.success("Entrenamiento completado!")
            
            # Guardar
            save_dir = Path("models/reader")
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            final_path = save_dir / f"tiny_speller_{target_language}_{timestamp}.ckpt"
            trainer.save_checkpoint(final_path)
            
            meta_config = {
                "epochs": epochs, "lr": lr, "batch_size": batch_size,
                "weights": {"dtw": w_dtw, "perceptual": w_perceptual},
                "listener_ckpt": lis_opts[sel_lis],
                "recognizer_ckpt": rec_opts[sel_rec],
                "language": target_language,
                "vocab": words,
                "training_phase": "g2p",
                "type": "speller"
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
        st.markdown("### 📚 Modelos TinySpeller (G2P)")
        all_ckpts = list_checkpoints("reader")
        speller_ckpts = [c for c in all_ckpts if "speller" in c['filename'] or c.get('meta', {}).get('config', {}).get('training_phase') == 'g2p']
        
        if not speller_ckpts:
            st.info("No hay modelos TinySpeller entrenados.")
        else:
            opts = {c['filename']: c for c in speller_ckpts}
            sel_key = st.selectbox("Seleccionar Modelo", list(opts.keys()))
            sel_ckpt = opts[sel_key]
            
            col_act1, col_act2 = st.columns([1, 3])
            with col_act1:
                if st.button("🗑️ Eliminar", key="del_sp"):
                    Path(sel_ckpt['path']).unlink(missing_ok=True)
                    Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json").unlink(missing_ok=True)
                    Path(sel_ckpt['path']).with_suffix(".csv").unlink(missing_ok=True)
                    st.rerun()
            
            st.divider()
            
            # --- EVALUACIÓN PROFUNDA ---
            st.markdown("### 🔬 Evaluación Profunda")
            
            if st.button("🚀 Cargar y Evaluar Modelo", key="eval_deep"):
                with st.spinner("Cargando modelo y ejecutando evaluación..."):
                    try:
                        # 1. Cargar Modelo
                        # Necesitamos saber las clases para cargar
                        meta_path = Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json")
                        words = []
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta = json.load(f)
                                words = meta.get('config', {}).get('vocab', [])
                        
                        # Si no hay vocab, intentar cargar sin él (si está en hparams) o error
                        kwargs = {"class_names": words} if words else {}
                        
                        model = TinyReaderLightning.load_from_checkpoint(sel_ckpt['path'], **kwargs)
                        model.eval()
                        device = encontrar_device()
                        model.to(device)
                        
                        # Si no había metadata, intentar recuperar vocabulario del modelo
                        if not words and hasattr(model, "class_names"):
                            words = model.class_names
                            st.warning("⚠️ Metadata no encontrada. Usando vocabulario guardado en el checkpoint.")
                        
                        # 2. Cargar Datos (Validation)
                        target_lang = model.hparams.get('target_language', 'es') # Intentar obtener de hparams o default
                        # Si no está en hparams, intentar de meta
                        if meta_path.exists():
                             with open(meta_path) as f:
                                meta = json.load(f)
                                target_lang = meta.get('config', {}).get('language', 'es')

                        _, _, _, loaders = build_audio_dataloaders(
                            batch_size=16,
                            target_language=target_lang,
                            num_workers=0,
                            seed=42,
                            use_phonemes=True # Para obtener el vocabulario correcto si es necesario, aunque aquí usamos words del modelo
                        )
                        val_loader = loaders['val']
                        
                        # 3. Loop de Inferencia
                        all_real_phonemes = []
                        all_pred_phonemes = []
                        all_real_idxs = []
                        all_pred_idxs = []
                        all_similarities = []
                        all_gen_embeddings_list = []
                        all_gen_embeddings_list = []
                        
                        results_data = []
                        
                        from utils import get_phonemes_from_word as get_phonemes_utils
                        from torch.nn.functional import cosine_similarity
                        
                        progress_bar = st.progress(0)
                        total_batches = len(val_loader)
                        
                        with torch.no_grad():
                            for batch_idx, batch in enumerate(val_loader):
                                # Extraer datos
                                batch_words = batch['words']
                                
                                # A. Obtener Logits Visuales (Input)
                                logits_sequences = []
                                for w in batch_words:
                                    images = model._get_word_images(w).to(device)
                                    res = model.recognizer(images)
                                    word_logits = res[0] if isinstance(res, tuple) else res
                                    logits_sequences.append(word_logits)
                                from torch.nn.utils.rnn import pad_sequence
                                padded_logits = pad_sequence(logits_sequences, batch_first=True, padding_value=0.0)
                                
                                # B. Calcular Targets Reales (para longitud y comparación)
                                phoneme_targets_list = []
                                for w in batch_words:
                                    ph = get_phonemes_utils(w)
                                    idxs = [model.phoneme_to_idx.get(p, 0) for p in ph]
                                    if not idxs: idxs = [0]
                                    phoneme_targets_list.append(idxs)
                                    
                                max_len = max(len(t) for t in phoneme_targets_list)
                                
                                # C. Generar Embeddings
                                gen_embeddings = model.reader_g2p(padded_logits, target_length=max_len)
                                # (B, T, D)
                                
                                # D. Clasificar (Predicción)
                                phoneme_logits = model.phoneme_listener.classifier(gen_embeddings)
                                preds = torch.argmax(phoneme_logits, dim=-1) # (B, T)
                                
                                # E. Análisis por palabra
                                for b in range(len(batch_words)):
                                    word = batch_words[b]
                                    real_idxs = phoneme_targets_list[b]
                                    pred_idxs = preds[b][:len(real_idxs)].cpu().tolist()
                                    
                                    real_ph_strs = [model.phoneme_class_names[i] for i in real_idxs]
                                    pred_ph_strs = [model.phoneme_class_names[i] for i in pred_idxs]
                                    
                                    # Guardar para matriz de confusión (flattened)
                                    all_real_phonemes.extend(real_ph_strs)
                                    all_pred_phonemes.extend(pred_ph_strs)
                                    all_real_idxs.extend(real_idxs)
                                    all_pred_idxs.extend(pred_idxs)
                                    
                                    # Calcular similitud de embeddings
                                    # Real embeddings (from bank)
                                    real_emb_tensor = model.phoneme_embeddings_bank[torch.tensor(real_idxs, device=device)]
                                    gen_emb_tensor = gen_embeddings[b][:len(real_idxs)]
                                    
                                    # Guardar embeddings generados para PCA
                                    all_gen_embeddings_list.extend(gen_emb_tensor.cpu().numpy())
                                    
                                    sims = cosine_similarity(real_emb_tensor, gen_emb_tensor, dim=-1).cpu().tolist()
                                    avg_sim = sum(sims) / len(sims) if sims else 0
                                    all_similarities.extend(sims)
                                    
                                    results_data.append({
                                        "Palabra": word,
                                        "Real": " ".join(real_ph_strs),
                                        "Predicción": " ".join(pred_ph_strs),
                                        "Similitud Promedio": f"{avg_sim:.4f}",
                                        "Longitud": len(real_idxs)
                                    })
                                
                                progress_bar.progress((batch_idx + 1) / total_batches)
                                
                        st.success("Evaluación Finalizada")
                        
                        # 4. Visualización
                        
                        # A. Tabla de Resultados
                        st.markdown("#### 📝 Comparación de Secuencias")
                        df_res = pd.DataFrame(results_data)
                        st.dataframe(df_res)
                        
                        # B. Matriz de Confusión
                        st.markdown("#### 😵 Matriz de Confusión (Fonemas)")
                        from components.analytics import plot_confusion_matrix
                        # Pasar índices y nombres de clases completos para evitar errores de etiquetas faltantes
                        plot_confusion_matrix(all_real_idxs, all_pred_idxs, model.phoneme_class_names)
                        
                        # C. Análisis de Embeddings
                        st.markdown("#### 💎 Estructura de Embeddings")
                        col_emb1, col_emb2 = st.columns(2)
                        with col_emb1:
                            st.metric("Total Fonemas Generados", len(all_similarities))
                            st.metric("Dimensión del Embedding", model.phoneme_listener.hidden_dim)
                            
                        with col_emb2:
                            # Histograma de similitud
                            fig, ax = plt.subplots()
                            ax.hist(all_similarities, bins=20, color='skyblue', edgecolor='black')
                            ax.set_title("Distribución de Similitud Cosine (Gen vs Real)")
                            ax.set_xlabel("Similitud")
                            ax.set_ylabel("Frecuencia")
                            st.pyplot(fig)
                            
                        # D. PCA 3D
                        st.markdown("#### 🎨 Proyección PCA 3D (Espacio Latente)")
                        st.info("Visualización 3D interactiva de los embeddings de fonemas generados. Los puntos del mismo color (mismo fonema real) deberían agruparse juntos, demostrando que el modelo ha aprendido una representación robusta e invariante.")
                        
                        if len(all_gen_embeddings_list) > 5: # Mínimo para PCA
                            try:
                                from sklearn.decomposition import PCA
                                import plotly.express as px
                                
                                # Usar 3 componentes para 3D
                                pca = PCA(n_components=3)
                                embeddings_3d = pca.fit_transform(all_gen_embeddings_list)
                                
                                df_pca = pd.DataFrame({
                                    'x': embeddings_3d[:, 0],
                                    'y': embeddings_3d[:, 1],
                                    'z': embeddings_3d[:, 2],
                                    'Fonema Real': all_real_phonemes,
                                    'Fonema Predicho': all_pred_phonemes
                                })
                                
                                fig = px.scatter_3d(
                                    df_pca, 
                                    x='x', y='y', z='z',
                                    color='Fonema Real',
                                    symbol='Fonema Real',
                                    hover_data=['Fonema Predicho'],
                                    title="Espacio Latente de Fonemas (3D)",
                                    opacity=0.7,
                                    size_max=10
                                )
                                
                                fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.warning(f"No se pudo generar PCA 3D: {e}")
                        else:
                            st.warning("No hay suficientes datos para PCA.")

                    except Exception as e:
                        st.error(f"Error en evaluación: {e}")
                        st.exception(e)

    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    with tabs[3]:
        st.markdown("### 🧪 Laboratorio G2P")
        
        # 1. Selección de Modelo
        all_ckpts = list_checkpoints("reader")
        speller_ckpts = [c for c in all_ckpts if "speller" in c['filename'] or c.get('meta', {}).get('config', {}).get('training_phase') == 'g2p']
        
        if not speller_ckpts:
            st.warning("⚠️ No hay modelos TinySpeller disponibles. Entrena uno primero.")
        else:
            opts = {c['filename']: c for c in speller_ckpts}
            col_sel1, col_sel2 = st.columns([3, 1])
            with col_sel1:
                sel_key_lab = st.selectbox("Seleccionar Modelo para Laboratorio", list(opts.keys()), key="sel_lab_model")
            
            sel_ckpt_lab = opts[sel_key_lab]
            
            # 2. Cargar Modelo (Lazy Loading con Session State para no recargar siempre)
            if 'lab_model_path' not in st.session_state or st.session_state.lab_model_path != sel_ckpt_lab['path']:
                with st.spinner("Cargando modelo..."):
                    try:
                        # Intentar cargar metadata para vocabulario
                        meta_path = Path(sel_ckpt_lab['path']).with_suffix(".ckpt.meta.json")
                        words = []
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta = json.load(f)
                                words = meta.get('config', {}).get('vocab', [])
                        
                        kwargs = {"class_names": words} if words else {}
                        model_lab = TinyReaderLightning.load_from_checkpoint(sel_ckpt_lab['path'], **kwargs)
                        model_lab.eval()
                        device = encontrar_device()
                        model_lab.to(device)
                        
                        st.session_state.lab_model = model_lab
                        st.session_state.lab_model_path = sel_ckpt_lab['path']
                    except Exception as e:
                        st.error(f"Error cargando modelo: {e}")
                        st.stop()
            
            model_lab = st.session_state.lab_model
            device = encontrar_device()
            
            # 3. Interfaz de Entrada
            st.markdown("#### ✍️ Entrada")
            col_in1, col_in2 = st.columns([3, 1])
            with col_in1:
                input_word = st.text_input("Escribe una palabra:", value="mama").strip().lower()
            with col_in2:
                if st.button("🎲 Palabra Aleatoria"):
                    if model_lab.class_names:
                        import random
                        input_word = random.choice(model_lab.class_names)
                        st.info(f"Palabra seleccionada: {input_word}")
                    else:
                        input_word = "test"
            
            if input_word:
                st.divider()
                try:
                    from utils import tokenize_graphemes
                    # Obtener grafemas disponibles del modelo visual
                    available_graphemes = list(model_lab.visual_config.keys())
                    tokens = tokenize_graphemes(input_word, available_graphemes)
                    
                    # Usar tokens para obtener imágenes
                    # _get_word_images ya usa tokenize_graphemes internamente, pero necesitamos los tokens para iterar
                    images = model_lab._get_word_images(input_word).to(device)
                    # images: (L, 3, 64, 64)
                    
                    # B. Imaginación Fonológica (Serial)
                    st.markdown("#### Imaginación Fonológica (TinySpeller)")
                    st.caption("El modelo procesa cada grafema individualmente para revelar su 'voz interna'.")
                    
                    serial_pred_phonemes = []
                    
                    # Iterar sobre tokens
                    cols_serial = st.columns(min(len(tokens), 5))
                    
                    import plotly.express as px

                    with torch.no_grad():
                        for i, token in enumerate(tokens):
                            col_idx = i % 5
                            if col_idx == 0 and i > 0:
                                cols_serial = st.columns(min(len(tokens) - i, 5))
                            
                            with cols_serial[col_idx]:
                                st.markdown(f"**{token}**")
                                
                                # 1. Imagen
                                if i < len(images):
                                    img_tensor = images[i] # (3, 64, 64)
                                    st.image(img_tensor.permute(1, 2, 0).cpu().numpy(), use_container_width=True)
                                    
                                    # 2. Inferencia Individual
                                    # (1, 3, 64, 64)
                                    img_batch = img_tensor.unsqueeze(0)
                                    
                                    # Recognizer
                                    res = model_lab.recognizer(img_batch)
                                    w_logits = res[0] if isinstance(res, tuple) else res
                                    # (1, VDim) -> (1, 1, VDim)
                                    w_logits = w_logits.unsqueeze(1)
                                    
                                    # Reader G2P
                                    g_emb = model_lab.reader_g2p(w_logits, target_length=1)
                                    # g_emb: (1, 1, 256)
                                    
                                    # Listener Classifier
                                    p_logits = model_lab.phoneme_listener.classifier(g_emb)
                                    p_idx = torch.argmax(p_logits, dim=-1).item()
                                    p_str = model_lab.phoneme_class_names[p_idx]
                                    
                                    serial_pred_phonemes.append(p_str)
                                    
                                    st.markdown(f"🡆 **/{p_str}/**")

                                    # 3. Visualización de Embedding Individual
                                    emb_vis = g_emb.squeeze(0).squeeze(0).cpu().numpy() # (256,)
                                    # Reshape para heatmap vertical (256, 1)
                                    emb_vis = emb_vis.reshape(-1, 1)
                                    
                                    fig_emb = px.imshow(
                                        emb_vis, 
                                        labels=dict(x="", y="Dim", color="Act"),
                                        aspect="auto",
                                        color_continuous_scale="Viridis"
                                    )
                                    fig_emb.update_layout(
                                        height=150, 
                                        margin=dict(l=0, r=0, t=0, b=0),
                                        xaxis=dict(showticklabels=False),
                                        yaxis=dict(showticklabels=False)
                                    )
                                    st.plotly_chart(fig_emb, use_container_width=True)
                                    
                    # C. Comparación
                    from utils import get_phonemes_from_word as get_phonemes_utils_lab
                    real_phonemes = get_phonemes_utils_lab(input_word)
                    
                    st.markdown("#### Verificación")
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.info(f"**Real:** /{ ' - '.join(real_phonemes) }/")
                        
                    with col_res2:
                        # Calcular precisión simple
                        min_len = min(len(real_phonemes), len(serial_pred_phonemes))
                        matches = sum(1 for i in range(min_len) if real_phonemes[i] == serial_pred_phonemes[i])
                        acc = matches / max(len(real_phonemes), 1)
                        
                        if acc == 1.0:
                            st.success(f"**Precisión:** {acc:.0%}")
                        else:
                            st.warning(f"**Precisión:** {acc:.0%}")


                            
                except Exception as e:
                    st.error(f"Error en inferencia: {e}")

if __name__ == "__main__":
    main()
