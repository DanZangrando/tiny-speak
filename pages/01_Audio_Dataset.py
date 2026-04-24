import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from components.modern_sidebar import display_modern_sidebar
from training.config import load_master_dataset_config

# Configurar página
st.set_page_config(
    page_title="Explorador de Audio - TinySpeak",
    page_icon="📦",
    layout="wide"
)

def display_audio_stats(samples_dict):
    """Muestra estadísticas rápidas del dataset de audio."""
    total_words = len(samples_dict)
    total_files = sum(len(variations) for variations in samples_dict.values())
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Palabras Únicas", total_words)
    c2.metric("Archivos Totales", total_files)
    c3.metric("Promedio Variaciones", f"{total_files/total_words:.1f}" if total_words > 0 else 0)

def main():
    display_modern_sidebar("audio_dataset")
    
    st.title("📦 Explorador de Dataset de Audio")
    st.markdown("Inspecciona y reproduce las muestras generadas para los 3 idiomas del experimento.")
    
    config = load_master_dataset_config()
    all_words = config.get("generated_samples", {})
    all_phonemes = config.get("phoneme_samples", {})
    
    if not all_words and not all_phonemes:
        st.warning("⚠️ No se han encontrado muestras generadas. Ve a la página de Configuración para generar el dataset.")
        return

    # Selector de Idioma y Tipo
    langs = sorted(list(set(all_words.keys()) | set(all_phonemes.keys())))
    if not langs:
        st.error("Estructura de dataset no reconocida o vacía.")
        return
        
    c1, c2 = st.columns(2)
    with c1:
        selected_lang = st.segmented_control(
            "📍 Idioma", 
            options=langs, 
            format_func=lambda x: {'es': '🇪🇸 Español', 'en': '🇺🇸 Inglés', 'fr': '🇫🇷 Francés'}.get(x, x.upper()),
            default=langs[0]
        )
    with c2:
        dataset_type = st.segmented_control(
            "📁 Categoría",
            options=["Palabras", "Fonemas"],
            default="Palabras"
        )
    
    samples_source = all_words if dataset_type == "Palabras" else all_phonemes
    lang_samples = samples_source.get(selected_lang, {})
    
    # Estadísticas del idioma
    display_audio_stats(lang_samples)
    
    st.markdown("---")
    
    # Reproductor e Inspección
    col1, col2 = st.columns([1, 2])
    
    with col1:
        words = sorted(list(lang_samples.keys()))
        label_search = "🔍 Buscar Palabra" if dataset_type == "Palabras" else "🔍 Buscar Fonema"
        selected_word = st.selectbox(label_search, words)
        
        if selected_word:
            variations = lang_samples[selected_word]
            st.write(f"**Variaciones disponibles:** {len(variations)}")
            
            for i, var in enumerate(variations):
                with st.expander(f"Muestra {i+1}: {var.get('tipo', 'original').title()}"):
                    st.json(var)
    
    with col2:
        if selected_word:
            st.subheader(f"Reproductor: **{selected_word}** ({selected_lang.upper()})")
            variations = lang_samples[selected_word]
            
            for i, var in enumerate(variations):
                file_path = Path(__file__).parent.parent / var.get('file_path', '')
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.write(f"**{var.get('tipo', 'original').title()}**")
                    st.caption(f"{var.get('duracion_ms', 0)}ms")
                with col_b:
                    if file_path.exists():
                        st.audio(str(file_path))
                    else:
                        st.error("Archivo no encontrado")
                st.markdown("---")

if __name__ == "__main__":
    main()