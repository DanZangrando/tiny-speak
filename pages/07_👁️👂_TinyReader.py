"""
🧠 TinyReader - La Voz Interior
Modelo generativo que aprende a imaginar audio (embeddings) a partir de conceptos.
"""

import streamlit as st
import pytorch_lightning as pl
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
from utils import list_checkpoints, encontrar_device

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
            checkpoints = list_checkpoints("listener")
            if not checkpoints:
                st.error("❌ Necesitas un TinyListener entrenado.")
            else:
                ckpt_opts = {f"{c['filename']}": c['path'] for c in checkpoints}
                sel_ckpt_name = st.selectbox("Modelo Perceptivo", list(ckpt_opts.keys()))
                selected_ckpt_path = ckpt_opts[sel_ckpt_name]
                
                if st.button("🚀 Iniciar Entrenamiento", type="primary"):
                    run_training(selected_ckpt_path, epochs, lr, batch_size, w_mse, w_cos, w_perceptual)
            
            st.info("⚡ Callbacks: ModelCheckpoint (val_loss).")
            
        with st.expander("💻 Ver Código de Entrenamiento (LightningModule)"):
            st.code(get_function_source(TinyReaderLightning), language="python")

    # ==========================================
    # TAB 3: MODELOS GUARDADOS
    # ==========================================
    with tabs[2]:
        st.markdown("### 📚 Gestión de Modelos")
        reader_checkpoints = list_checkpoints("reader")
        
        if not reader_checkpoints:
            st.info("No hay modelos entrenados.")
        else:
            for ckpt in reader_checkpoints:
                with st.expander(f"🧠 {ckpt['filename']} - {datetime.fromtimestamp(ckpt['timestamp']).strftime('%Y-%m-%d %H:%M')}"):
                    col_info, col_actions = st.columns([3, 1])
                    with col_info:
                        meta = ckpt.get('meta', {})
                        st.json(meta.get('config', {}), expanded=False)
                        metrics = meta.get('metrics', {})
                        if metrics:
                            st.markdown("**Métricas:**")
                            st.write(metrics)
                    with col_actions:
                        if st.button("🗑️ Eliminar", key=f"del_{ckpt['filename']}"):
                            Path(ckpt['path']).unlink(missing_ok=True)
                            Path(ckpt['path']).with_suffix(".ckpt.meta.json").unlink(missing_ok=True)
                            st.rerun()

    # ==========================================
    # TAB 4: LABORATORIO
    # ==========================================
    with tabs[3]:
        st.markdown("### 🧪 Laboratorio de Imaginación")
        run_laboratory()

def run_training(ckpt_path, epochs, lr, batch_size, w_mse, w_cos, w_perceptual):
    config = load_master_dataset_config()
    words = config.get("diccionario_seleccionado", {}).get("palabras", [])
    
    if not words:
        st.error("Diccionario vacío.")
        return

    with st.spinner("Preparando..."):
        train_ds, val_ds, test_ds, loaders = build_audio_dataloaders(
            batch_size=batch_size, num_workers=0, seed=42
        )
        
        model = TinyReaderLightning(
            class_names=words,
            listener_checkpoint_path=str(ckpt_path),
            learning_rate=lr,
            w_mse=w_mse,
            w_cos=w_cos,
            w_perceptual=w_perceptual
        )
        
        history_cb = ReaderHistoryCallback()
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            callbacks=[history_cb],
            enable_progress_bar=True,
            default_root_dir="lightning_logs/tiny_reader"
        )
        
    progress_bar = st.progress(0)
    with st.spinner(f"Entrenando..."):
        trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
        
    st.success("Entrenamiento completado!")
    
    save_dir = Path("models/reader")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_dir / f"reader_{timestamp}.ckpt"
    trainer.save_checkpoint(final_path)
    
    train_config = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "weights": {"mse": w_mse, "cos": w_cos, "perceptual": w_perceptual},
        "listener_ckpt": str(ckpt_path)
    }
    final_metrics = history_cb.history[-1] if history_cb.history else {}
    save_model_metadata(final_path, train_config, final_metrics)
    
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
        sel_listener = st.selectbox("TinyListener (Juez)", list(l_opts.keys()))
        
    config = load_master_dataset_config()
    words = config.get("diccionario_seleccionado", {}).get("palabras", [])
    target_word = st.selectbox("Palabra a imaginar:", words)
    
    # Visualización de Grafemas (Input Visual Conceptual)
    st.markdown("#### 👁️ Estímulo Visual (Concepto)")
    st.markdown("Estas son las letras que forman el concepto que el modelo va a 'leer' e imaginar.")
    
    # Buscar imágenes de las letras en data/visual
    # Asumimos estructura data/visual/<letra>/...
    visual_dir = Path("data/visual")
    cols = st.columns(len(target_word))
    
    for i, char in enumerate(target_word):
        char_dir = visual_dir / char.upper() # Asumiendo mayúsculas en carpetas
        # Si no existe, probar minúscula
        if not char_dir.exists():
            char_dir = visual_dir / char.lower()
            
        if char_dir.exists():
            # Tomar la primera imagen disponible
            imgs = list(char_dir.glob("*.png"))
            if imgs:
                img = Image.open(imgs[0])
                cols[i].image(img, caption=char, width=64)
            else:
                cols[i].warning(f"No img for {char}")
        else:
            cols[i].warning(f"No dir for {char}")

    if st.button("🧠 Imaginar y Escuchar", type="primary"):
        evaluate_imagination(r_opts[sel_reader], l_opts[sel_listener], target_word, words)

def evaluate_imagination(reader_path, listener_path, target_word, class_names):
    device = encontrar_device()
    
    with st.spinner(f"Imaginando en {device}..."):
        # Cargar Reader
        reader_module = TinyReaderLightning.load_from_checkpoint(
            reader_path,
            class_names=class_names,
            listener_checkpoint_path=listener_path,
            map_location=device
        )
        reader = reader_module.reader
        reader.to(device)
        reader.eval()
        
        # Cargar Listener
        from training.audio_module import TinyListenerLightning
        listener_module = TinyListenerLightning.load_from_checkpoint(
            listener_path,
            class_names=class_names,
            map_location=device
        )
        listener = listener_module.listener
        listener.to(device)
        listener.eval()
        
        # Input Concepto
        word_idx = class_names.index(target_word)
        concept_logits = torch.zeros(1, len(class_names), device=device)
        concept_logits[0, word_idx] = 1.0
        
        # Generar
        with torch.no_grad():
            generated_embeddings = reader(concept_logits, target_length=100)
            
        # Escuchar
        lengths = torch.tensor([100], device=device)
        packed_gen = torch.nn.utils.rnn.pack_padded_sequence(
            generated_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        with torch.no_grad():
            listener_logits, _ = listener.tiny_speak(packed_gen)
            probs = F.softmax(listener_logits, dim=-1)
            
    # Resultados
    st.markdown("### 🧠 Resultado")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        probs_cpu = probs[0].cpu()
        top_probs, top_idxs = torch.topk(probs_cpu, 3)
        for p, idx in zip(top_probs, top_idxs):
            w = class_names[idx]
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
