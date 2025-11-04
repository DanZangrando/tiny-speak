"""
Componente de Sidebar Moderna para TinySpeak
Sidebar persistente y reutilizable en todas las páginas
"""
import streamlit as st
import random
import base64
import io
from datetime import datetime
from PIL import Image
import json
from pathlib import Path

def load_dataset_config(config_path):
    """Carga configuración de dataset de manera segura"""
    try:
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def get_palabras_vocabulario_actual():
    """Obtiene las palabras del vocabulario configurado actualmente"""
    try:
        from diccionarios import get_diccionario_predefinido
        
        master_config = load_dataset_config("master_dataset_config.json")
        if master_config and 'diccionario_seleccionado' in master_config:
            dic_tipo = master_config['diccionario_seleccionado']['tipo']
            if dic_tipo == 'predefinido':
                dic_nombre = master_config['diccionario_seleccionado']['nombre']
                dic_data = get_diccionario_predefinido(dic_nombre)
                if dic_data:
                    return dic_data['palabras']
            elif dic_tipo == 'personalizado':
                return master_config['diccionario_seleccionado'].get('palabras', [])
        
        # Fallback a palabras por defecto
        from utils import get_default_words
        return get_default_words()
    except Exception:
        return []

def encontrar_device():
    """Detecta el dispositivo disponible"""
    try:
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        return "cpu"

def apply_sidebar_css():
    """Aplica CSS global para la sidebar moderna"""
    st.markdown("""
    <style>
    /* Modernización completa de la sidebar */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%) !important;
    }
    
    /* Ocultar elementos por defecto de Streamlit en sidebar */
    .css-1d391kg .stRadio, .css-1d391kg .stSelectbox {
        display: none !important;
    }
    
    /* Estilizar todos los elementos de texto en sidebar */
    section[data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Estilizar contenedor principal de sidebar */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%) !important;
        padding: 1rem !important;
    }
    
    /* Estilo para botones en sidebar */
    .css-1d391kg .stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .css-1d391kg .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Función eliminada - mini galería del sidebar removida

def display_modern_sidebar():
    """Sidebar modernizada y persistente para todas las páginas"""
    
    # Aplicar CSS moderno
    apply_sidebar_css()
    
    with st.sidebar:
        # Header principal más sutil
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; margin-bottom: 1rem;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 15px; color: white; 
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h1 style="margin: 0; font-size: 2.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🎤 TinySpeak
            </h1>
            <p style="margin: 0.5rem 0 0 0; font-style: italic; font-size: 1rem; opacity: 0.8;">
                Sistema Multimodal Avanzado
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Estado del sistema más sutil
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.15);
                    padding: 0.8rem; border-radius: 10px; margin: 1rem 0;
                    color: white; text-align: center;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(10px);">
            <h3 style="margin: 0; font-size: 1rem; font-weight: 600;">🔧 Estado del Sistema</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Device info más sutil
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
                    color: white; text-align: center; font-weight: 500;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(5px);">
            {device_icon} {device_name}
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset status - usando configuración unificada (desde root del proyecto)
        # El sidebar se ejecuta desde el directorio raíz, no desde pages/
        master_config = load_dataset_config("master_dataset_config.json")
        audio_config = master_config.get('generated_samples', {}) if master_config else {}
        visual_config = master_config.get('visual_dataset', {}) if master_config else {}
        
        # Audio Dataset Card más sutil
        if audio_config:
            audio_samples = sum(len(v) for v in audio_config.values())
            audio_words = len(audio_config)
            audio_color = "rgba(76, 175, 80, 0.2)"
            audio_status = f"✅ {audio_samples:,} muestras ({audio_words} palabras)"
        else:
            audio_color = "rgba(255, 193, 7, 0.2)"
            audio_status = "⚙️ No generado"
        
        st.markdown(f"""
        <div style="background: {audio_color};
                    padding: 0.6rem; border-radius: 8px; margin: 0.3rem 0;
                    color: white; text-align: left; font-weight: 400;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(5px); font-size: 0.9rem;">
            🎵 {audio_status}
        </div>
        """, unsafe_allow_html=True)
        
        # Visual Dataset Card más sutil
        if visual_config and visual_config.get('generated_images'):
            visual_samples = sum(len(v) for v in visual_config['generated_images'].values())
            visual_letters = len(visual_config['generated_images'])
            visual_color = "rgba(103, 58, 183, 0.2)"
            visual_status = f"✅ {visual_samples:,} imágenes ({visual_letters} letras)"
        else:
            visual_color = "rgba(255, 193, 7, 0.2)"
            visual_status = "⚙️ No generado"
        
        st.markdown(f"""
        <div style="background: {visual_color};
                    padding: 0.6rem; border-radius: 8px; margin: 0.3rem 0;
                    color: white; text-align: left; font-weight: 400;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(5px); font-size: 0.9rem;">
            🖼️ {visual_status}
        </div>
        """, unsafe_allow_html=True)
        
        # Sección removida - Mini galería y navegación eliminadas
        
        st.markdown("---")
        
        # Navegación funcional con botones de Streamlit
        st.markdown("**📄 Páginas disponibles:**")
        
        # Botones que realmente funcionan con Streamlit
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎵 Audio", key="nav_audio_global", help="Ir a Audio Dataset", width='stretch'):
                st.switch_page("pages/00_🎵_Audio_Dataset.py")
        
        with col2:
            if st.button("🖼️ Visual", key="nav_visual_global", help="Ir a Visual Dataset Manager", width='stretch'):
                st.switch_page("pages/00_🖼️_Visual_Dataset_Manager.py")
        
        # Analytics navegación
        col3, col4 = st.columns(2)
        with col3:
            if st.button("🎵 Analytics", key="nav_audio_analytics_global", help="Audio Analytics", width='stretch'):
                st.switch_page("pages/01_🎵_Audio_Analytics.py")
        
        with col4:
            if st.button("📊 Analytics", key="nav_visual_analytics_global", help="Visual Analytics", width='stretch'):
                st.switch_page("pages/01_🖼️_Visual_Analytics.py")
        
        # Más botones de navegación
        if st.button("� TinyListener", key="nav_listener_global", width='stretch'):
            st.switch_page("pages/01_�_TinyListener.py")
        
        if st.button("�️ TinyRecognizer", key="nav_recognizer_global", width='stretch'):
            st.switch_page("pages/02_�️_TinyRecognizer.py")
        
        if st.button("🔗 TinySpeller", key="nav_speller_global", width='stretch'):
            st.switch_page("pages/03_🔗_TinySpeller.py")
        
        # Botón para ir al Dashboard principal
        if st.button("🏠 Dashboard", key="nav_dashboard_global", width='stretch'):
            st.switch_page("app.py")
        
        st.markdown("---")
        
        # Información del sistema más sutil
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1);
                    padding: 0.6rem; border-radius: 8px; margin: 0.5rem 0;
                    color: white; text-align: center;
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h4 style="margin: 0; font-size: 0.9rem; font-weight: 500;">📱 Información del Sistema</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Vocabulario activo más sutil
        try:
            palabras = get_palabras_vocabulario_actual()
            vocab_count = len(palabras)
        except:
            vocab_count = 0
        
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.1);
                    padding: 0.4rem; border-radius: 6px; margin: 0.2rem 0;
                    color: rgba(255,255,255,0.9); text-align: left; font-size: 0.8rem;
                    border-left: 3px solid {'#4CAF50' if vocab_count > 0 else '#FFA726'};">
            📚 Vocabulario: {vocab_count} palabras
        </div>
        """, unsafe_allow_html=True)
        
        # Timestamp más sutil
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05);
                    padding: 0.3rem; border-radius: 4px; margin: 0.2rem 0;
                    color: rgba(255,255,255,0.6); text-align: center; font-size: 0.7rem;">
            🕐 {timestamp}
        </div>
        """, unsafe_allow_html=True)
        
        # Información técnica compacta
        with st.expander("🤖 Arquitectura del Sistema", expanded=False):
            st.markdown("""
            **🎵 TinyListener**
            - Wav2Vec2 + LSTM
            - ~2.1M parámetros
            
            **🖼️ TinyRecognizer** 
            - CORnet-Z inspirado
            - ~850K parámetros
            
            **🔗 TinySpeller**
            - Fusión multimodal
            - Consenso inteligente
            """)
        
        # Footer minimalista
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 0.8rem;
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 8px; color: rgba(255,255,255,0.7); 
                    margin-top: 1rem; border: 1px solid rgba(255,255,255,0.1);">
            <p style="margin: 0; font-size: 0.8rem; font-weight: 500;">
                🚀 TinySpeak v2.0
            </p>
            <p style="margin: 0; font-size: 0.6rem; opacity: 0.6;">
                Sistema Multimodal
            </p>
        </div>
        """, unsafe_allow_html=True)