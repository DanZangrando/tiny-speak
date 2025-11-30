"""
🧠 TinyReader (Stage 2) - Aprendiendo a Leer (P2W)
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
import os
import matplotlib.pyplot as plt

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_tiny_reader_diagram
from components.code_viewer import get_function_source
from models import TinyReader
from training.reader_module import TinyReaderLightning
from training.audio_dataset import build_audio_dataloaders, DEFAULT_AUDIO_SPLIT_RATIOS
from training.config import load_master_dataset_config
from utils import list_checkpoints, encontrar_device, save_model_metadata, RealTimePlotCallback, ReaderPredictionCallback
from components.analytics import plot_learning_curves, plot_confusion_matrix, display_classification_report, plot_probability_matrix, plot_latent_space_pca

# Configurar página
st.set_page_config(
    page_title="TinyReader - Stage 2",
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
    display_modern_sidebar("tiny_reader")
    
    st.markdown('<h1 class="main-header">🧠 TinyReader: Aprendiendo a Leer (Stage 2)</h1>', unsafe_allow_html=True)

    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("### 📖 Stage 2: Phoneme-to-Word (P2W)")
        
        st.markdown("""
        ### 📖 TinyReader: Acceso Léxico (Phoneme-to-Word)

        #### 1. Filosofía de Diseño: De Sonidos a Significados
        Esta etapa completa la ruta de lectura. Una vez que TinySpeller ha convertido las letras en sonidos (fonemas), TinyReader debe ensamblar esos sonidos para acceder al significado (la palabra).
        
        *   **Simulación del Reconocimiento Auditivo de Palabras:** El cerebro no "lee" fonema por fonema aisladamente; integra la secuencia fonológica para activar una representación léxica.
        *   **Juez Léxico:** Usamos *TinyEars (Words)* como el estándar de oro. TinyReader debe producir una representación que active el "área de la palabra" de la misma forma que lo haría escuchar la palabra hablada.

        #### 2. Arquitectura Cognitiva
        El modelo actúa como un integrador temporal:
        
        *   **Input:** Secuencia de embeddings fonémicos (generados por TinySpeller o ideales).
        *   **Procesamiento:** Red Recurrente (LSTM) o Transformer que acumula evidencia a lo largo de la secuencia fonémica.
        *   **Output:** Un único vector de palabra (Word Embedding) que se compara con el espacio latente de TinyEars (Words).

        #### 3. Input/Output
        *   **Entrada:** Secuencia de Embeddings Fonémicos.
        *   **Salida:** Embedding Léxico y clasificación de la palabra.
        """)
        
        st.graphviz_chart(get_tiny_reader_diagram())

    # ==========================================
    # TAB 2: ENTRENAMIENTO
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Entrenamiento (P2W)")
        
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
            
            # 1. TinySpeller (Stage 1)
            speller_ckpts = list_checkpoints("reader")
            # Filtrar spellers
            valid_spellers = [c for c in speller_ckpts if "speller" in c['filename'] or c.get('meta', {}).get('config', {}).get('training_phase') == 'g2p']
            
            if not valid_spellers:
                st.error("❌ No hay modelos TinySpeller (Stage 1) disponibles. Entrena uno en 'TinySpeller'.")
                st.stop()
                
            speller_opts = {c['filename']: c['path'] for c in valid_spellers}
            sel_speller = st.selectbox("TinySpeller (G2P - Congelado)", list(speller_opts.keys()))
            
            # 2. Listener (Words)
            lis_ckpts = list_checkpoints("listener")
            # Filtrar palabras
            word_ckpts = [c for c in lis_ckpts if "word" in c['filename'] or c.get('meta', {}).get('config', {}).get('type') == 'word']
            
            if not word_ckpts:
                st.error("❌ No hay modelos TinyEars (Palabras) disponibles. Entrena uno en 'TinyEars - Palabras'.")
                st.stop()
                
            lis_opts = {c['filename']: c['path'] for c in word_ckpts}
            sel_lis = st.selectbox("TinyEars (Juez Léxico)", list(lis_opts.keys()))
            
            # Dataset
            target_language = st.selectbox("Idioma Objetivo", config.get('experiment_config', {}).get('languages', ['es']))
            
            # Para P2W, el output son palabras (audio)
            audio_data = config.get('generated_samples', {}).get(target_language, {})
            if audio_data:
                # Filtrar vacíos y ordenar para coincidir con el dataset
                words = sorted([w for w, s in audio_data.items() if s])
            else:
                words = []
                
            st.info(f"Entrenando sobre {len(words)} palabras.")

        if st.button("🚀 Iniciar Entrenamiento P2W (Stage 2)", type="primary"):
            # Setup
            pl.seed_everything(42)
            
            # 1. Construir Dataloaders PRIMERO
            try:
                train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
                    batch_size=batch_size,
                    target_language=target_language,
                    num_workers=4,
                    seed=42,
                    use_phonemes=False # P2W uses words as targets, not phonemes directly from dataset
                )
                
                words = train_ds.class_names
                st.success(f"Dataset cargado con {len(words)} palabras.")
                
            except Exception as e:
                st.error(f"Error cargando datos: {e}")
                st.stop()
                
            # 2. Inicializar Modelo
            if not words:
                 st.error("⚠️ No se encontraron datos válidos en el dataset.")
                 st.stop()
                 
            # Cargar metadata del speller para obtener el recognizer usado
            speller_path = speller_opts[sel_speller]
            # Intentar leer metadata
            meta_path = Path(speller_path).with_suffix(".ckpt.meta.json")
            rec_ckpt_path = None
            phoneme_listener_path = None
            
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    rec_ckpt_path = meta.get('config', {}).get('recognizer_ckpt')
                    phoneme_listener_path = meta.get('config', {}).get('listener_ckpt')
            
            if not rec_ckpt_path:
                st.warning("⚠️ No se encontró el Recognizer en la metadata del Speller. Intentando buscar uno compatible...")
                # Fallback logic...
                rec_ckpts = list_checkpoints("recognizer")
                if rec_ckpts:
                    rec_ckpt_path = rec_ckpts[0]['path']
                else:
                    st.error("No se encontró ningún Recognizer.")
                    st.stop()
                    
            if not phoneme_listener_path:
                 st.warning("⚠️ No se encontró el Phoneme Listener en la metadata del Speller.")
                 # Fallback logic...
                 lis_ckpts = list_checkpoints("listener")
                 phoneme_ckpts = [c for c in lis_ckpts if "phoneme" in c['filename']]
                 if phoneme_ckpts:
                     phoneme_listener_path = phoneme_ckpts[0]['path']
                 else:
                     st.error("No se encontró ningún Phoneme Listener.")
                     st.stop()

            # Modelo
            # Aquí cargamos el modelo base pero le decimos que use el checkpoint del speller como base
            # TinyReaderLightning manejará la carga de pesos si se le pasa pretrained_reader_ckpt?
            # No, TinyReaderLightning no tiene argumento 'pretrained_reader_ckpt' en __init__.
            # Debemos cargar el checkpoint del speller y luego modificarlo para Stage 2?
            # O inicializar uno nuevo y cargar los pesos del G2P?
            
            # Opción A: Inicializar nuevo y cargar pesos.
            model = TinyReaderLightning(
                class_names=words,
                listener_checkpoint_path=lis_opts[sel_lis], # Word Listener
                recognizer_checkpoint_path=rec_ckpt_path,
                learning_rate=lr,
                w_dtw=w_dtw,
                w_perceptual=w_perceptual,
                use_two_stage=True,
                training_phase="p2w", # FASE P2W
                phoneme_listener_checkpoint_path=phoneme_listener_path
            )
            
            # Cargar pesos del Speller (G2P)
            # El Speller tiene weights para 'reader_g2p'.
            # Cargamos el checkpoint
            speller_ckpt = torch.load(speller_path, map_location=encontrar_device())
            state_dict = speller_ckpt['state_dict']
            
            # Filtrar solo las keys de reader_g2p
            g2p_weights = {k: v for k, v in state_dict.items() if "reader_g2p" in k}
            
            # Cargar en el modelo actual
            missing, unexpected = model.load_state_dict(g2p_weights, strict=False)
            
            # Verificar si faltan pesos CRÍTICOS (del propio G2P)
            missing_g2p = [k for k in missing if "reader_g2p" in k]
            
            if missing_g2p:
                st.warning(f"⚠️ Alerta: Faltan {len(missing_g2p)} pesos del módulo G2P: {missing_g2p[:5]}...")
            else:
                st.success(f"✅ Pesos de G2P cargados correctamente.")

            # TAMBIÉN cargar el Phoneme Listener del Speller (si fue fine-tuned)
            # El Speller (TinyReaderLightning en fase g2p) entrena todo, incluido el listener.
            # Si no cargamos su estado, el G2P generará embeddings para un listener distinto.
            ph_listener_weights = {k.replace("phoneme_listener.", ""): v for k, v in state_dict.items() if "phoneme_listener" in k}
            if ph_listener_weights:
                try:
                    model.phoneme_listener.load_state_dict(ph_listener_weights)
                    st.success(f"✅ Phoneme Listener actualizado desde el checkpoint del Speller ({len(ph_listener_weights)} pesos).")
                except Exception as e:
                    st.warning(f"⚠️ No se pudo actualizar Phoneme Listener desde Speller: {e}")
            else:
                st.info("ℹ️ El checkpoint del Speller no contiene pesos de Phoneme Listener (usando original).")
            
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
                filename="tiny_reader_stage2-{epoch:02d}-{val_loss:.2f}",
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
                default_root_dir="lightning_logs/tiny_reader_stage2"
            )
            
            # Placeholders
            st.markdown("### 📈 Progreso")
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                st.markdown("#### Pérdida (Loss)")
                plot_loss = st.empty()
            with col_plot2:
                st.markdown("#### Precisión (Palabras)")
                plot_acc = st.empty()
                
            realtime_cb = RealTimePlotCallback(plot_loss, plot_acc)
            trainer.callbacks.append(realtime_cb)
            
            # Prediction Callback
            pred_placeholder = st.empty()
            pred_cb = ReaderPredictionCallback(loaders['val'], pred_placeholder)
            trainer.callbacks.append(pred_cb)
            
            with st.spinner("Entrenando TinyReader (Stage 2 - P2W)..."):
                trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
                
            st.success("Entrenamiento completado!")
            
            # Guardar
            save_dir = Path("models/reader")
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            final_path = save_dir / f"tiny_reader_stage2_{target_language}_{timestamp}.ckpt"
            trainer.save_checkpoint(final_path)
            
            meta_config = {
                "epochs": epochs, "lr": lr, "batch_size": batch_size,
                "weights": {"dtw": w_dtw, "perceptual": w_perceptual},
                "speller_ckpt": speller_opts[sel_speller],
                "listener_ckpt": lis_opts[sel_lis],
                "language": target_language,
                "vocab": words,
                "training_phase": "p2w",
                "type": "reader"
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
        st.markdown("### 📚 Modelos TinyReader (P2W / End-to-End)")
        all_ckpts = list_checkpoints("reader")
        # Filtrar modelos que sean de stage 2 o end_to_end
        reader_ckpts = [c for c in all_ckpts if "reader" in c['filename'] and "speller" not in c['filename']]
        
        if not reader_ckpts:
            st.info("No hay modelos TinyReader entrenados.")
        else:
            opts = {c['filename']: c for c in reader_ckpts}
            sel_key = st.selectbox("Seleccionar Modelo", list(opts.keys()))
            sel_ckpt = opts[sel_key]
            
            col_act1, col_act2 = st.columns([1, 3])
            with col_act1:
                if st.button("🗑️ Eliminar", key="del_rd"):
                    Path(sel_ckpt['path']).unlink(missing_ok=True)
                    Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json").unlink(missing_ok=True)
                    Path(sel_ckpt['path']).with_suffix(".csv").unlink(missing_ok=True)
                    st.rerun()
            
            st.divider()
            
            # --- EVALUACIÓN PROFUNDA ---
            st.markdown("### 🔬 Evaluación Profunda (Full Pipeline)")
            st.info("Esta evaluación prueba la cadena completa: Texto -> TinyEyes -> TinySpeller -> TinyReader -> Significado.")
            
            if st.button("🚀 Cargar y Evaluar Modelo", key="eval_deep_reader"):
                with st.spinner("Cargando modelo y ejecutando evaluación..."):
                    try:
                        # 1. Cargar Modelo
                        meta_path = Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json")
                        words = []
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta = json.load(f)
                                words = meta.get('config', {}).get('vocab', [])
                        
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
                        target_lang = model.hparams.get('target_language', 'es')
                        if meta_path.exists():
                             with open(meta_path) as f:
                                meta = json.load(f)
                                target_lang = meta.get('config', {}).get('language', 'es')

                        _, _, _, loaders = build_audio_dataloaders(
                            batch_size=16,
                            target_language=target_lang,
                            num_workers=0,
                            seed=42,
                            use_phonemes=False # Usamos palabras completas
                        )
                        val_loader = loaders['val']
                        
                        # 3. Loop de Inferencia
                        all_real_words = []
                        all_pred_words = []
                        all_similarities = []
                        all_gen_embeddings_list = []
                        all_labels_idx = []
                        all_preds_idx = []
                        
                        results_data = []
                        
                        from utils import get_phonemes_from_word
                        from torch.nn.functional import cosine_similarity
                        from torch.nn.utils.rnn import pad_sequence
                        
                        progress_bar = st.progress(0)
                        total_batches = len(val_loader)
                        
                        correct_count = 0
                        total_count = 0
                        
                        with torch.no_grad():
                            for batch_idx, batch in enumerate(val_loader):
                                batch_words = batch['words']
                                waveforms = [w.to(device) for w in batch['waveforms']]
                                
                                # A. Pipeline Visual -> G2P
                                logits_sequences = []
                                for w in batch_words:
                                    images = model._get_word_images(w).to(device)
                                    res = model.recognizer(images)
                                    word_logits = res[0] if isinstance(res, tuple) else res
                                    logits_sequences.append(word_logits)
                                padded_logits = pad_sequence(logits_sequences, batch_first=True, padding_value=0.0)
                                
                                # Calcular longitudes de fonemas esperadas (para G2P)
                                phoneme_targets_list = []
                                for w in batch_words:
                                    ph = get_phonemes_from_word(w)
                                    idxs = [model.phoneme_to_idx.get(p, 0) for p in ph]
                                    if not idxs: idxs = [0]
                                    phoneme_targets_list.append(idxs)
                                max_len_phonemes = max(len(t) for t in phoneme_targets_list)
                                
                                # Ejecutar G2P
                                gen_phoneme_embs = model.reader_g2p(padded_logits, target_length=max_len_phonemes)
                                
                                # B. Pipeline P2W (TinyReader)
                                # Necesitamos la longitud del audio real para saber cuánto generar?
                                # O TinyReader genera una longitud fija/aprendida?
                                # En _shared_step usa max_len del audio real.
                                # Para inferencia pura, deberíamos no depender del audio real si es "lectura silenciosa".
                                # Pero TinyReader P2W genera una secuencia que imita al audio.
                                # Vamos a usar la longitud del audio real como "duración de lectura" por ahora para comparar.
                                
                                waveforms_padded = pad_sequence(waveforms, batch_first=True)
                                real_embeddings = model.listener.extract_hidden_activations(waveforms_padded)
                                real_embeddings, lengths = model.listener.mask_hidden_activations(real_embeddings)
                                real_embeddings, lengths = model.listener.downsample_hidden_activations(real_embeddings, lengths, factor=7)
                                real_embeddings = real_embeddings.squeeze(0) # (1, B, T, D) -> (B, T, D)
                                max_len_audio = real_embeddings.size(1)
                                
                                gen_word_seq = model.reader_p2w(gen_phoneme_embs, target_length=max_len_audio)
                                
                                # C. Pooling y Clasificación
                                # Global Average Pooling (masked)
                                mask = torch.arange(max_len_audio, device=device).expand(len(batch_words), max_len_audio) < lengths.unsqueeze(1)
                                mask_float = mask.float().unsqueeze(-1)
                                pooled_gen = (gen_word_seq * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
                                
                                listener_logits = model.listener.classifier(pooled_gen)
                                preds = torch.argmax(listener_logits, dim=-1)
                                
                                # D. Ground Truth (Audio Real)
                                # Pooled Real Embeddings
                                pooled_real = (real_embeddings * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
                                
                                # E. Análisis
                                for b in range(len(batch_words)):
                                    word = batch_words[b]
                                    pred_idx = preds[b].item()
                                    pred_word = model.class_names[pred_idx] if pred_idx < len(model.class_names) else "Unknown"
                                    
                                    all_real_words.append(word)
                                    all_pred_words.append(pred_word)
                                    
                                    # Guardar índices para matriz de confusión
                                    # IMPORTANTE: Usar el índice del vocabulario del MODELO, no del dataloader
                                    # Esto evita errores si el dataset ha cambiado (off-by-one)
                                    try:
                                        true_idx = model.class_names.index(word)
                                    except ValueError:
                                        true_idx = -1 # Palabra no está en el vocabulario del modelo
                                        
                                    all_labels_idx.append(true_idx)
                                    all_preds_idx.append(pred_idx)
                                    
                                    # Similitud Cosine (Word Embedding Level)
                                    sim = cosine_similarity(pooled_real[b].unsqueeze(0), pooled_gen[b].unsqueeze(0)).item()
                                    all_similarities.append(sim)
                                    
                                    # Guardar embedding para PCA
                                    all_gen_embeddings_list.append(pooled_gen[b].cpu().numpy())
                                    
                                    results_data.append({
                                        "Palabra Real": word,
                                        "Palabra Predicha": pred_word,
                                        "Similitud": f"{sim:.4f}",
                                        "Acierto": "✅" if word == pred_word else "❌"
                                    })
                                    
                                    if word == pred_word:
                                        correct_count += 1
                                    total_count += 1
                                
                                progress_bar.progress((batch_idx + 1) / total_batches)
                                
                        st.success("Evaluación Finalizada")
                        
                        # 4. Visualización
                        
                        # A. Métricas Globales
                        acc = correct_count / total_count if total_count > 0 else 0
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Exactitud (Accuracy)", f"{acc:.2%}")
                        col_m2.metric("Similitud Promedio", f"{sum(all_similarities)/len(all_similarities):.4f}")
                        col_m3.metric("Total Palabras", total_count)
                        
                        # B. Tabla
                        st.markdown("#### 📝 Resultados Detallados")
                        st.dataframe(pd.DataFrame(results_data))
                        
                        # C. Matriz de Confusión
                        if all_labels_idx and all_preds_idx:
                            st.markdown("#### 😵 Matriz de Confusión")
                            try:
                                plot_confusion_matrix(all_labels_idx, all_preds_idx, model.class_names)
                            except Exception as e:
                                st.warning(f"No se pudo generar la matriz de confusión: {e}")
                        
                        # C. Histograma Similitud
                        st.markdown("#### 📊 Distribución de Similitud (Semántica/Auditiva)")
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.hist(all_similarities, bins=20, color='lightgreen', edgecolor='black')
                        ax_hist.set_title("Similitud entre 'Lectura Silenciosa' y 'Escucha Real'")
                        st.pyplot(fig_hist)
                        
                        # D. PCA 3D
                        st.markdown("#### 🎨 Espacio Léxico (PCA 3D)")
                        st.info("Visualización de los embeddings de palabras generados por TinyReader. Cada punto es una palabra leída.")
                        
                        if len(all_gen_embeddings_list) > 5:
                            try:
                                from sklearn.decomposition import PCA
                                import plotly.express as px
                                
                                pca = PCA(n_components=3)
                                embeddings_3d = pca.fit_transform(all_gen_embeddings_list)
                                
                                df_pca = pd.DataFrame({
                                    'x': embeddings_3d[:, 0],
                                    'y': embeddings_3d[:, 1],
                                    'z': embeddings_3d[:, 2],
                                    'Palabra Real': all_real_words,
                                    'Palabra Predicha': all_pred_words
                                })
                                
                                fig = px.scatter_3d(
                                    df_pca, 
                                    x='x', y='y', z='z',
                                    color='Palabra Real',
                                    symbol='Palabra Real',
                                    hover_data=['Palabra Predicha'],
                                    title="Espacio Léxico Generado (3D)",
                                    opacity=0.7,
                                    size_max=10
                                )
                                fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"No se pudo generar PCA: {e}")

                    except Exception as e:
                        st.error(f"Error en evaluación: {e}")
                        st.exception(e)

    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    with tabs[3]:
        st.markdown("### 🧪 Laboratorio Unificado")
        st.markdown("Prueba el modelo TinyReader completo: **Texto -> Visión -> Fonología -> Significado**.")
        
        # 1. Selector de Modelo
        # Usar list_checkpoints para consistencia y filtrado
        all_ckpts = list_checkpoints("reader")
        # Filtrar modelos que sean de stage 2 o end_to_end (igual que en Evaluación)
        reader_ckpts = [c for c in all_ckpts if "reader" in c['filename'] and "speller" not in c['filename']]
        
        if not reader_ckpts:
            st.warning("⚠️ No hay modelos TinyReader (Stage 2) disponibles. Entrena uno en la pestaña 'Entrenamiento'.")
        else:
            ckpt_opts = {c['filename']: c['path'] for c in reader_ckpts}
            
            sel_ckpt_name = st.selectbox("Seleccionar Modelo Entrenado", list(ckpt_opts.keys()), key="lab_ckpt")
            
            if sel_ckpt_name:
                ckpt_path = ckpt_opts[sel_ckpt_name]
                
                if st.button("🚀 Cargar Modelo para Laboratorio", key="load_lab"):
                    with st.spinner("Cargando modelo..."):
                        try:
                            # Cargar metadata si existe para saber vocabulario
                            meta_path = Path(ckpt_path).with_suffix(".ckpt.meta.json")
                            vocab = []
                            if meta_path.exists():
                                with open(meta_path) as f:
                                    meta = json.load(f)
                                    vocab = meta.get('config', {}).get('vocab', [])
                            
                            kwargs = {"class_names": vocab} if vocab else {}
                            model = TinyReaderLightning.load_from_checkpoint(ckpt_path, **kwargs)
                            model.eval()
                            device = encontrar_device()
                            model.to(device)
                            st.session_state['lab_model'] = model
                            st.success(f"Modelo cargado: {sel_ckpt_name}")
                        except Exception as e:
                            st.error(f"Error cargando modelo: {e}")
                
                # 2. Interfaz de Prueba
                if 'lab_model' in st.session_state:
                    model = st.session_state['lab_model']
                    device = next(model.parameters()).device
                    
                    st.divider()
                    test_word = st.text_input("Escribe una palabra para leer:", value="gato").strip().lower()
                    
                    if st.button("🧠 Leer e Imaginar", type="primary"):
                        if not test_word:
                            st.warning("Escribe una palabra.")
                        else:
                            with st.spinner(f"Procesando '{test_word}'..."):
                                try:
                                    # A. Tokenización Visual (Chancho Logic)
                                    from utils import tokenize_graphemes
                                    available_graphemes = list(model.visual_config.keys())
                                    tokens = tokenize_graphemes(test_word, available_graphemes)
                                    
                                    st.write(f"**Tokenización Visual:** {tokens}")
                                    
                                    # Mostrar imágenes de entrada
                                    images = model._get_word_images(test_word).to(device) # (L, 3, 64, 64)
                                    
                                    cols = st.columns(len(tokens))
                                    for i, col in enumerate(cols):
                                        with col:
                                            # Convertir tensor a imagen para mostrar
                                            img_tensor = images[i].cpu()
                                            # Des-normalizar si es necesario (asumimos que transform lo hizo)
                                            # Por simplicidad, mostramos lo que hay
                                            img_np = img_tensor.permute(1, 2, 0).numpy()
                                            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                                            st.image(img_np, caption=tokens[i], use_container_width=True)
                                    
                                    # B. TinyEyes (Reconocimiento)
                                    # (L, 3, 64, 64) -> Procesamos como batch de letras
                                    res = model.recognizer(images)
                                    logits_flat = res[0] # (L, num_chars)
                                    logits = logits_flat.unsqueeze(0) # (1, L, num_chars)
                                    
                                    # C. TinySpeller (G2P) - Generación Serial (Uno a Uno)
                                    # El usuario prefiere que se pase grafema por grafema para evitar repeticiones
                                    gen_phoneme_embs_list = []
                                    pred_phonemes = []
                                    
                                    # Iterar sobre cada "letra" (token visual)
                                    for i in range(logits.size(1)):
                                        # Tomar el logit de la letra i
                                        # logits: (1, L, num_chars) -> (1, 1, num_chars)
                                        char_logit = logits[:, i:i+1, :]
                                        
                                        # Generar 1 embedding de fonema para esta letra
                                        # target_length=1 fuerza al modelo a generar un solo paso
                                        g_emb = model.reader_g2p(char_logit, target_length=1) # (1, 1, D)
                                        gen_phoneme_embs_list.append(g_emb)
                                        
                                        # Decodificar fonema (para visualización)
                                        p_logits = model.phoneme_listener.classifier(g_emb)
                                        p_idx = torch.argmax(p_logits, dim=-1).item()
                                        if p_idx < len(model.phoneme_class_names):
                                            pred_phonemes.append(model.phoneme_class_names[p_idx])
                                        else:
                                            pred_phonemes.append("?")
                                            
                                    # Concatenar embeddings para formar la secuencia completa
                                    # (1, L, D)
                                    # gen_phoneme_embs = torch.cat(gen_phoneme_embs_list, dim=1) # ESTO ROMPE EL CONTEXTO
                                    
                                    # SOLUCIÓN HÍBRIDA:
                                    # 1. Usamos la generación iterativa SOLO para visualizar (Speller Prediction) - Ya hecho arriba
                                    # 2. Usamos generación Full Sequence para pasar al Reader (P2W), PERO restringiendo la longitud
                                    #    exactamente al número de tokens visuales para evitar repeticiones.
                                    
                                    # Recalcular logits completos
                                    # padded_logits: (1, L, num_chars)
                                    
                                    # IMPORTANTE: Para que coincida con el entrenamiento/evaluación,
                                    # la longitud objetivo debe ser la cantidad de FONEMAS, no de grafemas.
                                    # En el Lab conocemos la palabra, así que podemos calcularlo.
                                    from utils import get_phonemes_from_word
                                    real_phonemes_list = get_phonemes_from_word(test_word)
                                    target_len_exact = len(real_phonemes_list)
                                    if target_len_exact == 0: target_len_exact = 1 # Fallback
                                    
                                    gen_phoneme_embs_contextual = model.reader_g2p(logits, target_length=target_len_exact)
                                    
                                    st.info(f"**Fonemas Predichos (TinySpeller):** {' - '.join(pred_phonemes)}")
                                    
                                    # D. TinyReader (P2W) -> Imaginación Auditiva
                                    # Generar embedding de palabra usando los embeddings CONTEXTUALES
                                    # Asumimos una duración de audio estándar (ej. 100 frames)
                                    target_len_audio = 100
                                    gen_word_seq = model.reader_p2w(gen_phoneme_embs_contextual, target_length=target_len_audio)
                                    
                                    # Pooling
                                    pooled_gen = gen_word_seq.mean(dim=1) # (1, D)
                                    
                                    # E. Interpretación (Búsqueda de Vecinos)
                                    # Comparar con embeddings de palabras conocidas (si tenemos un banco)
                                    # Si no, usamos el clasificador del Listener
                                    listener_logits = model.listener.classifier(pooled_gen)
                                    probs = torch.softmax(listener_logits, dim=-1)
                                    top_probs, top_idxs = torch.topk(probs, 5, dim=-1)
                                    
                                    st.markdown("### 🎯 Interpretación (Top 5)")
                                    
                                    results = []
                                    for i in range(5):
                                        idx = top_idxs[0, i].item()
                                        prob = top_probs[0, i].item()
                                        word_cls = model.class_names[idx] if idx < len(model.class_names) else "Unknown"
                                        results.append({"Palabra": word_cls, "Confianza": f"{prob:.2%}"})
                                        
                                    st.table(results)
                                    
                                    if results[0]["Palabra"] == test_word:
                                        st.balloons()
                                        st.success(f"¡Leído correctamente como '{test_word}'!")
                                    else:
                                        st.warning(f"Confusión: Leído como '{results[0]['Palabra']}' en lugar de '{test_word}'.")
                                        
                                except Exception as e:
                                    st.error(f"Error en inferencia: {e}")
                                    st.exception(e)

if __name__ == "__main__":
    main()
