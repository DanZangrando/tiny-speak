"""
Componente de Sidebar Moderna para TinySpeak
Sidebar persistente y reutilizable en todas las páginas
"""
import streamlit as st
from pathlib import Path

from utils.serialization import load_dataset_config
from utils.device import encontrar_device


def get_palabras_vocabulario_actual():
    """Obtiene las palabras del vocabulario configurado actualmente"""
    try:
        from utils.vocabulary import get_diccionario_predefinido
        
        master_config = load_dataset_config("data/master_dataset_config.json")
        if master_config and 'diccionario_seleccionado' in master_config:
            dic_sel = master_config['diccionario_seleccionado']
            dic_tipo = dic_sel.get('tipo', '')
            if dic_tipo == 'predefinido':
                dic_nombre = dic_sel['nombre']
                dic_data = get_diccionario_predefinido(dic_nombre)
                if dic_data:
                    return dic_data['palabras']
            elif dic_tipo == 'personalizado':
                return dic_sel.get('palabras', [])
        
        from utils.graphemes import get_default_words
        return get_default_words()
    except Exception:
        return []

def apply_sidebar_css():
    """Aplica CSS global para la sidebar moderna"""
    st.markdown("""
    <style>
    /* Estilo para botones en sidebar - sutil */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Función eliminada - mini galería del sidebar removida

def display_modern_sidebar(page_prefix="default"):
    """Sidebar modernizada y persistente para todas las páginas"""
    
    # Aplicar CSS moderno
    apply_sidebar_css()
    
    with st.sidebar:
        # Device info
        try:
            device = encontrar_device()
            device_name = str(device).upper()
            if 'cuda' in device_name:
                device_color = "rgba(76, 175, 80, 0.3)"
                device_icon = "🚀"
            else:
                device_color = "rgba(33, 150, 243, 0.3)"
                device_icon = "💻"
        except:
            device_color = "rgba(244, 67, 54, 0.3)"
            device_icon = "❌"
            device_name = "ERROR"
        
        st.markdown(f"""
        <div style="background: {device_color};
                    padding: 0.6rem; border-radius: 8px; margin: 0.3rem 0;
                    color: #FAFAFA; text-align: center; font-weight: 500;
                    border: 1px solid rgba(255, 255, 255, 0.1);">
            {device_icon} {device_name}
        </div>
        """, unsafe_allow_html=True)

        # Dataset status
        master_config = load_dataset_config("data/master_dataset_config.json")
        audio_config = master_config.get('generated_samples', {}) if master_config else {}
        visual_config = master_config.get('visual_dataset', {}) if master_config else {}
        
        # Audio Dataset Card
        if audio_config:
            audio_samples = 0
            audio_words = 0
            
            # Detectar estructura (plana vs anidada)
            is_nested = False
            if audio_config:
                first_val = next(iter(audio_config.values()))
                if isinstance(first_val, dict) and not isinstance(first_val, list):
                    is_nested = True
            
            # Estructura anidada: idioma -> palabra -> lista
            if is_nested:
                unique_words = set()
                for lang_data in audio_config.values():
                    if isinstance(lang_data, dict):
                        unique_words.update(lang_data.keys())
                        for word_variations in lang_data.values():
                            audio_samples += len(word_variations)
                audio_words = len(unique_words)
            else:
                # Estructura plana antigua
                audio_words = len(audio_config)
                for word_variations in audio_config.values():
                    if isinstance(word_variations, list):
                        audio_samples += len(word_variations)
                        
            audio_color = "rgba(76, 175, 80, 0.15)" 
            audio_status = f"✅ {audio_samples:,} muestras ({audio_words} palabras)"
        else:
            audio_color = "rgba(255, 193, 7, 0.15)"
            audio_status = "⚙️ No generado"
        
        st.markdown(f"""
        <div style="background: {audio_color};
                    padding: 0.6rem; border-radius: 8px; margin: 0.3rem 0;
                    color: #FAFAFA; text-align: left; font-weight: 400;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    font-size: 0.9rem;">
            🎵 {audio_status}
        </div>
        """, unsafe_allow_html=True)
        
        # Visual Dataset Card
        if visual_config and visual_config.get('generated_images'):
            visual_samples = sum(len(v) for v in visual_config['generated_images'].values())
            visual_letters = len(visual_config['generated_images'])
            visual_color = "rgba(103, 58, 183, 0.15)" 
            visual_status = f"✅ {visual_samples:,} imágenes ({visual_letters} grafemas)"
        else:
            visual_color = "rgba(255, 193, 7, 0.15)"
            visual_status = "⚙️ No generado"
        
        st.markdown(f"""
        <div style="background: {visual_color};
                    padding: 0.6rem; border-radius: 8px; margin: 0.3rem 0;
                    color: #FAFAFA; text-align: left; font-weight: 400;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    font-size: 0.9rem;">
            🖼️ {visual_status}
        </div>
        """, unsafe_allow_html=True)

        # Phoneme Dataset Card
        phoneme_config = master_config.get('phoneme_samples', {}) if master_config else {}
        if phoneme_config:
            phoneme_samples = 0
            unique_phonemes = set()
            
            # Estructura: idioma -> fonema -> lista
            for lang_data in phoneme_config.values():
                if isinstance(lang_data, dict):
                    unique_phonemes.update(lang_data.keys())
                    for p_list in lang_data.values():
                        phoneme_samples += len(p_list)
            
            phoneme_count = len(unique_phonemes)
            phoneme_color = "rgba(233, 30, 99, 0.15)" 
            phoneme_status = f"✅ {phoneme_samples:,} muestras ({phoneme_count} fonemas)"
        else:
            phoneme_color = "rgba(255, 193, 7, 0.15)"
            phoneme_status = "⚙️ No generado"
            
        st.markdown(f"""
        <div style="background: {phoneme_color};
                    padding: 0.6rem; border-radius: 8px; margin: 0.3rem 0;
                    color: #FAFAFA; text-align: left; font-weight: 400;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    font-size: 0.9rem;">
            🗣️ {phoneme_status}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="font-size: 0.85rem; color: #AAAAAA;">
        <b>1. TinyEyes (Visual):</b> CNN que aprende a "ver" grafemas (Visual Embeddings).<br>
        <b>2. TinyEars (Phonemes):</b> CNN+Transformer que aprende a "oír" fonemas (Phoneme Embeddings).<br>
        <b>3. TinyEars (Words):</b> CNN+Transformer que aprende a "oír" palabras (Word Embeddings).<br>
        <b>4. TinySpeller (Stage 1):</b> Aprende Grapheme-to-Phoneme (G2P).<br>
        <b>5. TinyReader (Stage 2):</b> Aprende Phoneme-to-Word (P2W).
        </div>
        """, unsafe_allow_html=True)
        
        # Información técnica detallada
        with st.expander("🤖 Arquitectura del Sistema", expanded=True):
            st.markdown("""
            <div style="font-size: 0.85rem;">
            
            **👂 TinyEars (Phonemes)**
            *   **Input:** Audio (Phoneme)
            *   **Output:** Phoneme Class
            *   **Role:** Auditory Judge for Stage 1
            
            **👂 TinyEars (Words)**
            *   **Input:** Audio (Word)
            *   **Output:** Word Class
            *   **Role:** Auditory Judge for Stage 2
            
            **🧠 TinySpeller (Stage 1)**
            *   **Input:** Text (Graphemes)
            *   **Output:** Phoneme Embeddings
            *   **Goal:** Learn G2P mapping
            
            **🧠 TinyReader (Stage 2)**
            *   **Input:** Phoneme Embeddings
            *   **Output:** Word Embeddings
            *   **Goal:** Learn P2W mapping (Reading)
            
            </div>
            """, unsafe_allow_html=True)
        
        # Footer removed