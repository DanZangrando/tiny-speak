"""
🧠 TinyReader - La Voz Interior
Modelo generativo que aprende a imaginar audio (embeddings) a partir de conceptos.
"""

import streamlit as st
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import time
from datetime import datetime
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from components.modern_sidebar import display_modern_sidebar
from components.diagrams import get_reader_diagram
from components.code_viewer import get_function_source
from models import TinyReader
from training.reader_module import TinyReaderLightning
from training.audio_dataset import build_audio_dataloaders, DEFAULT_AUDIO_SPLIT_RATIOS
from training.config import load_master_dataset_config
from utils import list_checkpoints, encontrar_device, load_waveform, save_model_metadata, RealTimePlotCallback
from components.analytics import plot_learning_curves, plot_confusion_matrix, display_classification_report, plot_probability_matrix, plot_latent_space_pca

# Configurar página
st.set_page_config(
    page_title="TinyReader - Voz Interior",
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

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    display_modern_sidebar("tiny_reader")
    
    st.markdown('<h1 class="main-header">👁️👂 TinyReader: La Voz Interior</h1>', unsafe_allow_html=True)

    tabs = st.tabs(["📐 Arquitectura", "🏃‍♂️ Entrenamiento", "💾 Modelos Guardados", "🧪 Laboratorio"])

    # ==========================================
    # TAB 1: ARQUITECTURA
    # ==========================================
    with tabs[0]:
        st.markdown("### 💭 Imaginación Generativa (Top-Down)")
        
        st.markdown("""
        <div class="card">
            <b>Tarea:</b> Generación de embeddings auditivos a partir de conceptos.<br>
            <b>Input:</b> Vector de concepto (Logits/One-Hot).<br>
            <b>Output:</b> Secuencia de embeddings (Batch, Time, 768).<br>
            <b>Justificación:</b> <b>TinyReader</b> simula la "voz interior". Recibe un concepto abstracto (la idea de una palabra) y genera una representación auditiva interna. 
            Utiliza una <b>Pérdida Híbrida</b> inspirada en Predictive Coding: minimiza el error de reconstrucción (MSE/Coseno) y maximiza la inteligibilidad según el "Oído Interno" (Pérdida Perceptiva).
        </div>
        """, unsafe_allow_html=True)
        
        st.graphviz_chart(get_reader_diagram())
        
        with st.expander("💻 Ver Código del Modelo (models.py)"):
            st.code(get_function_source(TinyReader), language="python")

    # ==========================================
    # TAB 2: ENTRENAMIENTO
    # ==========================================
    with tabs[1]:
        st.markdown("### ⚙️ Configuración del Experimento")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Hiperparámetros")
            epochs = st.number_input("Épocas", 1, 200, 50)
            lr = st.number_input("Learning Rate", value=1e-3, format="%.1e")
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=2)
            
            st.markdown("#### Pesos de Pérdida")
            w_mse = st.slider("MSE (Reconstrucción)", 0.0, 10.0, 1.0)
            w_cos = st.slider("Coseno (Estructura)", 0.0, 10.0, 1.0)
            w_perceptual = st.slider("Perceptual (Inteligibilidad)", 0.0, 10.0, 0.5)
            
        with col2:
            st.markdown("#### Oído Interno & Callbacks")
            
            # Selector de Idioma
            config = load_master_dataset_config()
            exp_config = config.get('experiment_config', {})
            available_langs = exp_config.get('languages', ['es'])
            target_lang = st.selectbox("Idioma de Entrenamiento", available_langs, index=0)
            
            # Selector de TinyListener (Oído Interno)
            checkpoints = list_checkpoints("listener")
            if not checkpoints:
                st.error("❌ Necesitas un TinyListener entrenado.")
            else:
                # Filtrar checkpoints por idioma
                filtered_ckpts = [c for c in checkpoints if c.get('meta', {}).get('language') == target_lang or f"_{target_lang}_" in c['filename']]
                if not filtered_ckpts:
                    st.warning(f"No se encontraron modelos Listener específicos para '{target_lang}'. Mostrando todos.")
                    filtered_ckpts = checkpoints

                ckpt_opts = {f"{c['filename']}": c['path'] for c in filtered_ckpts}
                sel_ckpt_name = st.selectbox("Modelo Perceptivo (Phonological Pathway)", list(ckpt_opts.keys()))
                selected_ckpt_path = ckpt_opts[sel_ckpt_name]
                
                # Selector de TinyRecognizer (Ojo)
                rec_ckpts = list_checkpoints("recognizer")
                if not rec_ckpts:
                    st.error("❌ Necesitas un TinyRecognizer entrenado.")
                else:
                    # Filtrar checkpoints por idioma
                    filtered_rec_ckpts = [c for c in rec_ckpts if c.get('meta', {}).get('language') == target_lang or f"_{target_lang}_" in c['filename']]
                    if not filtered_rec_ckpts:
                        st.warning(f"No se encontraron modelos Recognizer específicos para '{target_lang}'. Mostrando todos.")
                        filtered_rec_ckpts = rec_ckpts
                        
                    rec_opts = {f"{c['filename']}": c['path'] for c in filtered_rec_ckpts}
                    sel_rec_name = st.selectbox("Modelo Visual (Visual Pathway)", list(rec_opts.keys()))
                    selected_rec_path = rec_opts[sel_rec_name]
                
                    st.info("⚡ Callbacks: ModelCheckpoint (val_loss) & EarlyStopping.")
                    patience = st.slider("Patience (Early Stopping)", 1, 20, 10, help="Número de épocas sin mejora antes de detener.")
                    min_delta = st.slider("Min Delta (Early Stopping)", 0.0, 0.1, 0.00, step=0.001, format="%.3f", help="Mejora mínima para considerar.")

                    if st.button("🚀 Iniciar Entrenamiento", type="primary"):
                        run_training(selected_ckpt_path, selected_rec_path, epochs, lr, batch_size, w_mse, w_cos, w_perceptual, patience, min_delta, target_lang)
            
        with st.expander("💻 Ver Código de Entrenamiento (LightningModule)"):
            st.code(get_function_source(TinyReaderLightning), language="python")

    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📚 Gestión de Modelos")
        reader_checkpoints = list_checkpoints("reader")
        
        if not reader_checkpoints:
            st.info("No hay modelos entrenados.")
        else:
            ckpt_opts = {f"{c['filename']} ({datetime.fromtimestamp(c['timestamp']).strftime('%Y-%m-%d %H:%M')})": c for c in reader_checkpoints}
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
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Val Loss", f"{metrics.get('val_loss', 0):.4f}")
                    m_col2.metric("Train Loss", f"{metrics.get('train_loss', 0):.4f}")
                    
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

            # 2. Evaluación Perceptual
            st.markdown("### 🧠 Evaluación Perceptual (Oído Interno)")
            st.markdown("Genera 'imaginaciones' para cada concepto y verifica si el Phonological Pathway las entiende correctamente.")
            
            if st.button("🚀 Ejecutar Evaluación Perceptual", key=f"eval_{sel_ckpt['filename']}"):
                with st.spinner("Cargando modelos y generando imaginaciones..."):
                    try:
                        # 1. Cargar Metadata del Reader
                        meta_path = Path(sel_ckpt['path']).with_suffix(".ckpt.meta.json")
                        words = []
                        target_lang = None
                        listener_ckpt_path = None
                        
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta_config = json.load(f).get("config", {})
                                words = meta_config.get("vocab", [])
                                target_lang = meta_config.get("language")
                                listener_ckpt_path = meta_config.get("listener_ckpt")
                        
                        # Fallback Language
                        if not target_lang:
                            for lang in ['es', 'en', 'fr']:
                                if f"_{lang}_" in sel_ckpt['filename']:
                                    target_lang = lang
                                    break
                        
                        if not target_lang:
                            config = load_master_dataset_config()
                            target_lang = config.get('experiment_config', {}).get('languages', ['es'])[0]
                            st.warning(f"⚠️ Idioma no detectado. Asumiendo '{target_lang}'.")

                        # 2. Buscar Listener Compatible
                        # Si tenemos path en metadata y existe, genial. Si no, buscar uno del mismo idioma.
                        final_listener_path = None
                        
                        if listener_ckpt_path and Path(listener_ckpt_path).exists():
                            final_listener_path = listener_ckpt_path
                        else:
                            # Buscar el mejor listener para este idioma
                            listener_ckpts = list_checkpoints("listener")
                            candidates = []
                            for c in listener_ckpts:
                                # Chequear idioma en metadata o nombre
                                c_lang = c.get('meta', {}).get('language')
                                if c_lang == target_lang:
                                    candidates.append(c)
                                    continue
                                if f"_{target_lang}_" in c['filename']:
                                    candidates.append(c)
                            
                            if candidates:
                                # Usar el más reciente (asumiendo ordenados por fecha o nombre con timestamp)
                                final_listener_path = candidates[0]['path']
                            elif listener_ckpts:
                                # Fallback desesperado: usar cualquiera (pero avisar)
                                final_listener_path = listener_ckpts[0]['path']
                                st.warning(f"⚠️ No se encontró un Listener específico para '{target_lang}'. Usando {Path(final_listener_path).name}.")
                        
                        if not final_listener_path:
                            st.error("❌ No se encontró ningún modelo TinyListener. Entrena uno primero.")
                            st.stop()

                        # 3. Cargar Modelo Reader
                        device = encontrar_device()
                        
                        # Si no hay words en metadata, intentar cargarlas del listener o config
                        if not words:
                            # Intentar leer metadata del listener
                            l_meta_path = Path(final_listener_path).with_suffix(".ckpt.meta.json")
                            if l_meta_path.exists():
                                with open(l_meta_path) as f:
                                    words = json.load(f).get("config", {}).get("vocab", [])
                            
                        if not words:
                             config = load_master_dataset_config()
                             words = config.get("diccionario_seleccionado", {}).get("palabras", [])

                        # Buscar un Recognizer compatible (mismo idioma)
                        rec_ckpts = list_checkpoints("recognizer")
                        compatible_rec = [c for c in rec_ckpts if c.get('meta', {}).get('language') == target_lang or f"_{target_lang}_" in c['filename']]
                        # Si no hay específico, usar cualquiera (fallback)
                        final_rec_path = compatible_rec[0]["path"] if compatible_rec else (rec_ckpts[0]["path"] if rec_ckpts else None)
                        
                        if not final_rec_path:
                            st.error("No se encontró ningún modelo TinyRecognizer para la evaluación.")
                            st.stop()

                        model = TinyReaderLightning.load_from_checkpoint(
                            sel_ckpt['path'],
                            class_names=words,
                            listener_checkpoint_path=final_listener_path,
                            recognizer_checkpoint_path=final_rec_path,
                            map_location=device
                        )
                        model.eval()
                        model.to(device)
                        
                        # Verificar consistencia
                        # El Reader debe tener num_classes igual a len(words)
                        # Y el Listener interno también.
                        
                        # 4. Inferencia Loop
                        all_preds = []
                        all_probs = []
                        all_labels = []
                        all_embeddings = []
                        
                        progress_bar = st.progress(0)
                        
                        # Generar N ejemplos por clase
                        samples_per_class = 5
                        total_steps = len(words) * samples_per_class
                        step = 0
                        
                        with torch.no_grad():
                            for i, word in enumerate(words):
                                # Label real
                                label_idx = i
                                
                                # Input: Secuencia de imágenes (Spelling)
                                # (L_word, C, H, W)
                                images = model._get_word_images(word).to(device)
                                
                                # Recognizer: Imágenes -> Logits de Letras
                                # (L_word, NumLetters)
                                res = model.recognizer(images)
                                if isinstance(res, tuple):
                                    word_logits = res[0]
                                else:
                                    word_logits = res
                                    
                                # Preparar batch (repetir para samples_per_class)
                                # (samples, L_word, NumLetters)
                                batch_logits = word_logits.unsqueeze(0).expand(samples_per_class, -1, -1)
                                
                                # Generar
                                # (samples, T, D)
                                generated_embeddings = model.reader(batch_logits, target_length=100)
                                
                                # Guardar embeddings para PCA (promedio temporal para tener 1 vector por sample)
                                # (B, L, D) -> (B, D)
                                avg_embeddings = generated_embeddings.mean(dim=1)
                                all_embeddings.extend(avg_embeddings.cpu().numpy())
                                
                                # Escuchar (Listener)
                                # PhonologicalPathway usa Mean Pooling + Classifier
                                # generated_embeddings: (samples, T, D)
                                pooled_gen = generated_embeddings.mean(dim=1)
                                listener_logits = model.listener.classifier(pooled_gen)
                                probs = torch.softmax(listener_logits, dim=1)
                                preds = torch.argmax(probs, dim=1)
                                
                                all_preds.extend(preds.cpu().numpy())
                                all_probs.extend(probs.cpu().numpy())
                                all_labels.extend([label_idx] * samples_per_class)
                                
                                step += samples_per_class
                                progress_bar.progress(min(step / total_steps, 1.0))
                                    
                            st.success("Evaluación perceptual completada.")
                            
                            # Visualizar
                            st.markdown("#### Mapa de Calor de Probabilidades (Perceptual)")
                            st.caption("Eje Y: Concepto Imaginado | Eje X: Probabilidad asignada por el Listener")
                            plot_probability_matrix(all_labels, all_probs, words, title="Mapa de Calor Perceptual")
                            
                            st.markdown("#### Espacio Latente Imaginado (PCA 3D)")
                            st.caption("Visualización de cómo se agrupan las 'imaginaciones' en el espacio vectorial.")
                            plot_latent_space_pca(all_embeddings, all_labels, words)
                            
                            display_classification_report(all_labels, all_preds, words)
                            
                    except Exception as e:
                        st.error(f"Error durante la evaluación: {e}")
                        st.exception(e)

    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    with tabs[3]:
        st.markdown("### 🧪 Laboratorio de Imaginación")
        run_laboratory()

def run_training(ckpt_path, rec_path, epochs, lr, batch_size, w_mse, w_cos, w_perceptual, patience, min_delta, target_language):
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
        
    model = TinyReaderLightning(
        class_names=words,
        listener_checkpoint_path=str(ckpt_path),
        recognizer_checkpoint_path=str(rec_path),
        learning_rate=lr,
        w_mse=w_mse,
        w_cos=w_cos,
        w_perceptual=w_perceptual
    )
    
    history_cb = ReaderHistoryCallback()
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/reader_checkpoints",
        filename="reader-{epoch:02d}-{val_loss:.2f}",
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
        default_root_dir="lightning_logs/tiny_reader"
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
        trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
        
    st.success("Entrenamiento completado!")
    
    save_dir = Path("models/reader")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"reader_{target_language}_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    meta_config = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "weights": {"mse": w_mse, "cos": w_cos, "perceptual": w_perceptual},
        "listener_ckpt": str(ckpt_path),
        "language": target_language,
        "vocab": words
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

def run_laboratory():
    reader_ckpts = list_checkpoints("reader")
    listener_ckpts = list_checkpoints("listener")
    
    if not reader_ckpts or not listener_ckpts:
        st.warning("Necesitas entrenar ambos modelos (Reader y Listener).")
        return
        
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        r_opts = {c['filename']: c['path'] for c in reader_ckpts}
        sel_reader = st.selectbox("TinyReader (Imaginador)", list(r_opts.keys()))
    with col_sel2:
        l_opts = {c['filename']: c['path'] for c in listener_ckpts}
        sel_listener = st.selectbox("Phonological Pathway (Juez)", list(l_opts.keys()))
        
    # Intentar cargar vocabulario del modelo seleccionado
    selected_reader_path = r_opts[sel_reader]
    model_vocab = []
    
    try:
        meta_path = Path(selected_reader_path).with_suffix('.ckpt.meta.json')
        if not meta_path.exists():
            meta_path = Path(selected_reader_path).with_suffix('.json')
            
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                model_vocab = meta.get('config', {}).get('vocab', [])
                if not model_vocab:
                    model_vocab = meta.get('vocab', [])
                    
            if model_vocab:
                st.success(f"✅ Vocabulario cargado del modelo ({len(model_vocab)} palabras).")
    except Exception as e:
        print(f"Error loading metadata: {e}")

    # Si encontramos vocabulario en el modelo, lo usamos. Si no, fallback a config.
    if model_vocab:
        words = model_vocab
    else:
        config = load_master_dataset_config()
        words = config.get("diccionario_seleccionado", {}).get("palabras", [])
        
        if not words:
            st.warning("⚠️ No se encontraron palabras en la configuración ni en el modelo. Usando lista por defecto.")
            from utils import get_default_words
            words = get_default_words()
            
    if not words:
        st.error("❌ No hay palabras disponibles para evaluar.")
        return

    target_word = st.selectbox("Palabra a imaginar:", words)
    
    if not target_word:
        st.warning("Selecciona una palabra para continuar.")
        return
    
    # Visualización de Grafemas (Input Visual Conceptual)
    st.markdown("#### 👁️ Estímulo Visual (Concepto)")
    st.markdown("Estas son las letras que forman el concepto que el modelo va a 'leer' e imaginar.")
    
    # Buscar imágenes de las letras en data/visual
    # Asumimos estructura data/visual/<letra>/...
    visual_dir = Path("data/visual")
    cols = st.columns(len(target_word))
    
    for i, char in enumerate(target_word):
        # Intentar encontrar el directorio de la letra
        char_lower = char.lower()
        char_upper = char.upper()
        
        # Posibles rutas
        candidates = [
            visual_dir / char_lower,
            visual_dir / char_upper,
            visual_dir / f"tiny_emnist_26/{char_lower}", # Estructura anidada común
            visual_dir / f"tiny_emnist_26/{char_upper}"
        ]
        
        found_img = None
        for char_dir in candidates:
            if char_dir.exists():
                # Buscar imágenes recursivamente
                imgs = list(char_dir.glob("**/*.png")) + list(char_dir.glob("**/*.jpg"))
                if imgs:
                    found_img = imgs[0] # Tomar la primera
                    break
        
        if found_img:
            img = Image.open(found_img)
            cols[i].image(img, caption=char, width=64)
        else:
            cols[i].warning(f"No img for {char}")

    if st.button("🧠 Imaginar y Escuchar", type="primary"):
        evaluate_imagination(r_opts[sel_reader], l_opts[sel_listener], target_word, words)

def evaluate_imagination(reader_path, listener_path, target_word, current_words):
    device = encontrar_device()
    
    # 1. Cargar Metadata del Reader para obtener el vocabulario correcto
    # Esto evita el error de size mismatch en el Listener interno
    try:
        ckpt = torch.load(reader_path, map_location='cpu')
        # Intentar leer metadata guardada en el checkpoint (si existe)
        # O usar el archivo .json asociado si existe
        # El formato de guardado es .ckpt.meta.json
        meta_path = Path(reader_path).with_suffix('.ckpt.meta.json')
        if not meta_path.exists():
             # Fallback antiguo
             meta_path = Path(reader_path).with_suffix('.json')
             
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                # La estructura es data['config']['vocab']
                vocab = meta.get('config', {}).get('vocab', [])
                if not vocab:
                     # Fallback si estaba en root (legacy)
                     vocab = meta.get('vocab', current_words)
                     
                target_lang = meta.get('config', {}).get('language', 'es')
        else:
            vocab = current_words
            target_lang = 'es' # Fallback
            st.warning("⚠️ No se encontró metadata del modelo. Usando vocabulario actual (puede fallar).")
            
    except Exception as e:
        st.error(f"Error leyendo checkpoint: {e}")
        return

    if target_word not in vocab:
        st.error(f"La palabra '{target_word}' no está en el vocabulario del modelo entrenado.")
        st.info(f"Vocabulario del modelo: {vocab}")
        return

    # 2. Buscar Recognizer compatible
    rec_ckpts = list_checkpoints("recognizer")
    compatible_rec = [c for c in rec_ckpts if c.get('meta', {}).get('language') == target_lang or f"_{target_lang}_" in c['filename']]
    final_rec_path = compatible_rec[0]["path"] if compatible_rec else (rec_ckpts[0]["path"] if rec_ckpts else None)
    
    if not final_rec_path:
        st.error("No se encontró ningún modelo TinyRecognizer para la imaginación.")
        return

    with st.spinner(f"Imaginando en {device}..."):
        # Cargar Reader (LightningModule)
        # Esto carga internamente el Listener y el Recognizer
        try:
            reader_module = TinyReaderLightning.load_from_checkpoint(
                reader_path,
                class_names=vocab, # USAR VOCAB DEL MODELO
                listener_checkpoint_path=listener_path,
                recognizer_checkpoint_path=final_rec_path,
                map_location=device
            )
        except RuntimeError as e:
            if "size mismatch" in str(e):
                st.error("❌ Error de compatibilidad: El Listener seleccionado no coincide con el usado durante el entrenamiento del Reader (diferente vocabulario).")
                st.info("Intenta seleccionar el mismo Listener que usaste para entrenar, o uno con el mismo diccionario.")
            raise e
            
        reader_module.to(device)
        reader_module.eval()
        
        # Usamos los modelos internos del módulo
        reader = reader_module.reader
        listener = reader_module.listener
        recognizer = reader_module.recognizer
        
        # Input Concepto: Secuencia de imágenes
        images = reader_module._get_word_images(target_word).to(device)
        
        # Recognizer: Imágenes -> Logits
        with torch.no_grad():
            res = recognizer(images)
            if isinstance(res, tuple):
                word_logits = res[0]
            else:
                word_logits = res
                
            # (1, L, NumLetters)
            batch_logits = word_logits.unsqueeze(0)
            
            # Reader: Logits -> Audio Embeddings
            # (1, T, D)
            generated_embeddings = reader(batch_logits, target_length=100)
            
        # Listener: Embeddings -> Predicción
        # PhonologicalPathway usa Mean Pooling + Classifier
        with torch.no_grad():
            # (1, T, D) -> (1, D)
            pooled_gen = generated_embeddings.mean(dim=1)
            listener_logits = listener.classifier(pooled_gen)
            probs = torch.softmax(listener_logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_word = vocab[pred_idx]
            confidence = probs[0, pred_idx].item()
            
        st.success(f"Imaginación completada.")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Palabra Objetivo", target_word)
        with col_res2:
            st.metric("Interpretación del Oído Interno", pred_word, delta=f"{confidence:.2%}")
            
        if pred_word == target_word:
            st.balloons()
        else:
            st.warning("El oído interno no reconoció lo que la mente imaginó.")
            
    # Resultados Detallados
    st.markdown("### 🧠 Probabilidades (Top 3)")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        probs_cpu = probs[0].cpu()
        top_probs, top_idxs = torch.topk(probs_cpu, min(3, len(vocab)))
        for p, idx in zip(top_probs, top_idxs):
            w = vocab[idx]
            st.write(f"**{w}**: {p:.2%}")
            st.progress(float(p))
            
    with col2:
        fig, ax = plt.subplots(figsize=(10, 3))
        emb_vis = generated_embeddings[0, :, :50].cpu().numpy().T 
        ax.imshow(emb_vis, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f"Imaginación de '{target_word}'")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
