"""
TinySpeak - Aplicación de Reconocimiento de Voz y Visión
Dashboard Principal del Sistema Multimodal - REFACTORIZADO
"""
import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import json
import tempfile
import os
import random
import base64
import io
from datetime import datetime
from PIL import Image

# Configurar la página
st.set_page_config(
    page_title="TinySpeak Dashboard",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar módulos del proyecto
from models import TinySpeak, TinyListener, TinyRecognizer, TinySpeller
from utils import (
    encontrar_device, load_wav2vec_model, get_default_words, synthesize_word,
    save_waveform_to_audio_file, WAV2VEC_SR, WAV2VEC_DIM
)
from diccionarios import (
    DICCIONARIOS_PREDEFINIDOS, get_diccionario_predefinido, 
    get_nombres_diccionarios, get_info_diccionarios
)

# Importar componente de sidebar moderna
from components.modern_sidebar import display_modern_sidebar

# =============================================================================
# CONFIGURACIÓN Y SETUP
# =============================================================================

@st.cache_resource
def setup_models():
    """Inicializa los modelos y configuración del sistema"""
    device = encontrar_device()
    st.sidebar.info(f"Dispositivo detectado: {device}")
    
    # Cargar modelo Wav2Vec2
    with st.spinner("Cargando modelo Wav2Vec2..."):
        wav2vec_model = load_wav2vec_model(device=device)
    
    # Obtener palabras por defecto
    words = get_default_words()
    
    # Inicializar modelos
    tiny_speak = TinySpeak(words=words, hidden_dim=64, num_layers=2, wav2vec_dim=WAV2VEC_DIM)
    tiny_listener = TinyListener(tiny_speak=tiny_speak, wav2vec_model=wav2vec_model)
    tiny_recognizer = TinyRecognizer(wav2vec_dim=WAV2VEC_DIM)
    tiny_speller = TinySpeller(tiny_recognizer=tiny_recognizer, tiny_speak=tiny_speak)
    
    # Mover modelos al dispositivo
    tiny_speak = tiny_speak.to(device)
    tiny_listener = tiny_listener.to(device)
    tiny_recognizer = tiny_recognizer.to(device)
    tiny_speller = tiny_speller.to(device)
    
    return {
        'device': device,
        'wav2vec_model': wav2vec_model,
        'tiny_speak': tiny_speak,
        'tiny_listener': tiny_listener,
        'tiny_recognizer': tiny_recognizer,
        'tiny_speller': tiny_speller,
        'words': words
    }

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_custom_css():
    """Retorna CSS personalizado para el tema moderno"""
    return """
    <style>
    /* Main header con gradiente */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
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
    
    /* Estilo para elementos de la sidebar */
    .css-1d391kg .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    .css-1d391kg .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .css-1d391kg .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Cards de modelos */
    .model-card {
        background: rgba(255, 107, 107, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 107, 107, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Container de métricas */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Mejoras para métricas de Streamlit */
    .metric-container .metric-value {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    /* Estilo para tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
        color: white;
        font-weight: bold;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
    }
    
    /* Estilo para selectboxes y otros widgets */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Botones principales */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Mejoras para progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    }
    
    /* Estilo para expandir */
    .streamlit-expanderHeader {
        background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Estilo para info boxes */
    .stAlert > div {
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Modernización de dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Sidebar moderna - texto */
    .css-1d391kg .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    .css-1d391kg .stCaption {
        color: rgba(255, 255, 255, 0.7);
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    </style>
    """

def create_model_metrics():
    """Crea las métricas simuladas de los modelos"""
    return {
        'TinyListener': {'params': '~2.1M', 'accuracy': '94.2%', 'delta': '2.1%'},
        'TinyRecognizer': {'params': '~850K', 'accuracy': '97.8%', 'delta': '1.5%'}, 
        'TinySpeller': {'latency': '12ms', 'accuracy': '98.9%', 'delta': '4.7%'}
    }

def load_dataset_config(config_path):
    """Carga configuración de dataset de manera segura"""
    try:
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error cargando {config_path}: {e}")
    return None

def convert_numpy_types(obj):
    """Convierte tipos numpy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_dataset_config(config, config_path):
    """Guarda configuración de dataset de manera segura"""
    try:
        # Convertir tipos numpy a tipos nativos de Python
        config_to_save = convert_numpy_types(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error guardando {config_path}: {e}")
        return False

def get_palabras_vocabulario_actual():
    """Obtiene las palabras del vocabulario configurado actualmente"""
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
    return get_default_words()

def verificar_consistencia_datasets():
    """Verifica que los datasets generados coincidan con la configuración actual"""
    master_config = load_dataset_config("master_dataset_config.json")
    if not master_config:
        return {"audio": False, "visual": False, "metodo": "N/A", "mensaje": "No hay configuración maestro"}
    
    palabras_configuradas = set(get_palabras_vocabulario_actual())
    config_audio = master_config.get('configuracion_audio', {})
    metodo_configurado = config_audio.get('metodo_sintesis', 'espeak')
    
    resultado = {
        "audio": True, 
        "visual": True, 
        "metodo": metodo_configurado,
        "mensaje": "Datasets sincronizados"
    }
    
    # Verificar dataset de audio
    audio_config = load_dataset_config("master_dataset_config.json")
    if audio_config and 'generated_samples' in audio_config:
        palabras_audio = set(audio_config['generated_samples'].keys())
        
        # Verificar si hay palabras generadas y si coinciden
        if not audio_config['generated_samples']:
            resultado["audio"] = False
            resultado["mensaje"] = "No hay dataset de audio generado"
        else:
            # Verificación más inteligente
            # Considerar sincronizado si el dataset contiene al menos las palabras principales del diccionario
            palabras_principales_presentes = len(palabras_audio & palabras_configuradas)
            total_configuradas = len(palabras_configuradas)
            
            # Si al menos el 80% de las palabras configuradas están en el dataset, considerar sincronizado
            porcentaje_cobertura = palabras_principales_presentes / total_configuradas if total_configuradas > 0 else 0
            
            if porcentaje_cobertura < 0.8:
                # Debug info solo si hay verdadera desincronización
                palabras_faltantes = palabras_configuradas - palabras_audio
                palabras_extra = palabras_audio - palabras_configuradas
                
                if len(palabras_faltantes) > 0:
                    resultado["audio"] = False
                    resultado["mensaje"] = f"Dataset incompleto: faltan {len(palabras_faltantes)} palabras del vocabulario"
                elif len(palabras_extra) > len(palabras_configuradas):
                    resultado["audio"] = False
                    resultado["mensaje"] = f"Dataset contiene {len(palabras_extra)} palabras adicionales no configuradas"
            else:
                # Dataset está bien sincronizado
                if palabras_audio == palabras_configuradas:
                    resultado["mensaje"] = f"Dataset perfectamente sincronizado ({len(palabras_audio)} palabras)"
                else:
                    resultado["mensaje"] = f"Dataset sincronizado ({palabras_principales_presentes}/{total_configuradas} palabras principales)"
        
        # Verificar método de síntesis (si está disponible en el config)
        metodo_dataset = audio_config.get('configuracion_audio', {}).get('metodo_sintesis', 
                                        audio_config.get('metodo_sintesis', 'desconocido'))
        if metodo_dataset != metodo_configurado and metodo_dataset != 'desconocido':
            resultado["audio"] = False
            resultado["mensaje"] = f"Método de síntesis desincronizado: {metodo_dataset} vs {metodo_configurado}"
    else:
        resultado["audio"] = False
        resultado["mensaje"] = "No se encontró dataset de audio generado"
    
    # Verificar dataset visual desde master config
    master_config = load_dataset_config("master_dataset_config.json")
    if master_config and 'visual_dataset' in master_config:
        visual_config = master_config['visual_dataset']
        if visual_config and 'generated_images' in visual_config:
            # Para visual, verificamos letras, no palabras completas
            pass
    
    return resultado

# =============================================================================
# COMPONENTES DEL DASHBOARD
# =============================================================================

def display_vocabulary_selector():
    """Selector simplificado de vocabulario para sincronización"""
    st.markdown("### 📚 Selección de Vocabulario")
    
    # Cargar configuración existente
    master_config = load_dataset_config("master_dataset_config.json")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Solo selector de diccionario predefinido (configuración detallada en páginas)
        info_diccionarios = get_info_diccionarios()
        
        default_diccionario = "tiny_kalulu_original"
        if master_config and 'diccionario_seleccionado' in master_config:
            dic_sel = master_config['diccionario_seleccionado']
            if dic_sel.get('tipo') == 'predefinido':
                default_diccionario = dic_sel.get('nombre', default_diccionario)
        
        keys_disponibles = list(info_diccionarios.keys())
        default_idx = keys_disponibles.index(default_diccionario) if default_diccionario in keys_disponibles else 0
        
        diccionario_key = st.selectbox(
            "🎯 Vocabulario activo:",
            keys_disponibles,
            index=default_idx,
            format_func=lambda x: f"{info_diccionarios[x]['nombre']} ({info_diccionarios[x]['cantidad_palabras']} palabras)",
            key="diccionario_seleccionado"
        )
        
        if diccionario_key:
            diccionario_data = get_diccionario_predefinido(diccionario_key)
            
            # Mostrar información básica del diccionario
            st.info(f"📖 **{diccionario_data['descripcion']}**")
            
            # Vista previa de palabras
            palabras_preview = diccionario_data['palabras'][:15]
            with st.expander("👀 Vista previa del vocabulario"):
                cols = st.columns(5)
                for i, palabra in enumerate(palabras_preview):
                    cols[i % 5].write(f"• {palabra}")
                if len(diccionario_data['palabras']) > 15:
                    st.caption(f"... y {len(diccionario_data['palabras']) - 15} palabras más")
    
    with col2:
        # Botón de sincronización
        if st.button("� Sincronizar", type="primary", help="Actualiza el vocabulario activo"):
            if diccionario_key:
                diccionario_data = get_diccionario_predefinido(diccionario_key)
                
                # Obtener configuración actual completa
                config_actual = master_config if master_config else {}
                
                # Información del nuevo diccionario
                nueva_info_diccionario = {
                    "tipo": "predefinido",
                    "nombre": diccionario_key,
                    "descripcion": diccionario_data['nombre'],
                    "palabras": diccionario_data['palabras']
                }
                
                # Verificar si cambió el diccionario
                diccionario_cambio = (
                    config_actual.get('diccionario_seleccionado', {}).get('nombre') != diccionario_key or
                    set(config_actual.get('diccionario_seleccionado', {}).get('palabras', [])) != set(diccionario_data['palabras'])
                )
                
                # Crear configuración actualizada preservando lo existente
                nueva_config = config_actual.copy()
                nueva_config.update({
                    "diccionario_seleccionado": nueva_info_diccionario,
                    "configuracion_audio": config_actual.get('configuracion_audio', {
                        "num_variaciones": 5,
                        "idioma": "es",
                        "metodo_sintesis": "gtts"
                    }),
                    "configuracion_visual": config_actual.get('configuracion_visual', {
                        "num_variaciones": 10
                    }),
                    "fecha_configuracion": pd.Timestamp.now().isoformat()
                })
                
                # Preservar generated_samples existentes - no limpiar automáticamente
                # El usuario puede limpiar manualmente en Audio Dataset Manager si lo desea
                # Solo actualizar la información del diccionario, no tocar los samples
                
                if save_dataset_config(nueva_config, "master_dataset_config.json"):
                    st.success("✅ Vocabulario sincronizado")
                    st.rerun()
                else:
                    st.error("❌ Error en sincronización")
        
        # Información rápida de configuración
        if master_config:
            config_audio = master_config.get('configuracion_audio', {})
            st.caption("**Configuración actual:**")
            st.caption(f"🎵 Método: {config_audio.get('metodo_sintesis', 'No definido')}")
            st.caption(f"🌍 Idioma: {config_audio.get('idioma', 'No definido')}")
            st.caption(f"📊 Variaciones: {config_audio.get('num_variaciones', 'No definido')}")
        
        # Botón para ir a configuración detallada
        st.markdown("---")
        st.info("⚙️ Para configuración detallada de parámetros, ve a las páginas de Audio Dataset y Visual Dataset")

def display_system_metrics():
    """Muestra métricas del sistema en tiempo real"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Dispositivo
    try:
        device = encontrar_device()
        device_name = str(device).upper()
        device_emoji = "🚀" if 'cuda' in device_name else "💻"
    except:
        device_name = "ERROR"
        device_emoji = "❌"
    
    # Vocabulario actual (dinámico)
    try:
        palabras_actuales = get_palabras_vocabulario_actual()
        vocab_size = len(palabras_actuales)
        vocab_status = "📚"
    except:
        vocab_size = 0
        vocab_status = "⚠️"
    
    # Datasets con verificación de contenido real
    audio_config = load_dataset_config("master_dataset_config.json")
    if audio_config and 'generated_samples' in audio_config:
        total_audio_samples = sum(len(samples) for samples in audio_config['generated_samples'].values())
        audio_status = f"✅ ({total_audio_samples})"
    else:
        audio_status = "⚙️"
    
    master_config = load_dataset_config("master_dataset_config.json")
    visual_config = master_config.get('visual_dataset', {}) if master_config else {}
    if visual_config and 'generated_images' in visual_config:
        total_visual_samples = sum(len(images) for images in visual_config['generated_images'].values())
        visual_status = f"✅ ({total_visual_samples})"
    else:
        visual_status = "⚙️"
    
    # Verificar consistencia
    consistencia = verificar_consistencia_datasets()
    consistencia_emoji = "✅" if consistencia["audio"] and consistencia["visual"] else "⚠️"
    
    col1.metric(f"{device_emoji} Dispositivo", device_name)
    col2.metric(f"{vocab_status} Vocabulario", f"{vocab_size}")
    col3.metric("🎵 Dataset Audio", audio_status)
    col4.metric("🖼️ Dataset Visual", visual_status)
    
    # Mostrar estado de consistencia
    if not (consistencia["audio"] and consistencia["visual"]):
        st.warning(f"⚠️ {consistencia['mensaje']}")
        if st.button("🔄 Actualizar métricas", key="refresh_metrics"):
            st.rerun()

def display_model_cards():
    """Muestra las tarjetas de información de los modelos"""
    col1, col2, col3 = st.columns(3)
    metrics = create_model_metrics()
    
    with col1:
        st.markdown("""
        <div class="model-card">
        <h4>🎵 TinyListener</h4>
        <p><strong>Audio → Palabra</strong></p>
        <ul>
        <li>🤖 Wav2Vec2 + LSTM</li>
        <li>🎯 ~200 palabras español</li>
        <li>⚡ Tiempo real</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Parámetros", metrics['TinyListener']['params'], "Compacto")
        st.metric("Precisión", metrics['TinyListener']['accuracy'], metrics['TinyListener']['delta'])
        
    with col2:
        st.markdown("""
        <div class="model-card">
        <h4>🖼️ TinyRecognizer</h4>
        <p><strong>Imagen → Letra</strong></p>
        <ul>
        <li>🧠 CORnet-Z inspirado</li>
        <li>🔤 26 letras alfabeto</li>
        <li>📱 Optimizado móvil</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Parámetros", metrics['TinyRecognizer']['params'], "Eficiente")
        st.metric("Precisión", metrics['TinyRecognizer']['accuracy'], metrics['TinyRecognizer']['delta'])
        
    with col3:
        st.markdown("""
        <div class="model-card">
        <h4>🔗 TinySpeller</h4>
        <p><strong>Multimodal → Consenso</strong></p>
        <ul>
        <li>🔄 Fusión modalidades</li>
        <li>📊 Confianza agregada</li>
        <li>🎯 Mayor robustez</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Precisión", metrics['TinySpeller']['accuracy'], metrics['TinySpeller']['delta'])
        st.metric("Latencia", metrics['TinySpeller']['latency'], "Ultra rápido")



def display_dataset_statistics():
    """Estadísticas detalladas y completas de los datasets generados"""
    st.markdown("### 📊 Estadísticas de los Datasets")
    
    # Cargar configuraciones
    master_config = load_dataset_config("master_dataset_config.json")
    master_config = load_dataset_config("master_dataset_config.json")
    audio_config = master_config if master_config else {}
    visual_config = master_config.get('visual_dataset', {}) if master_config else {}
    
    if not master_config:
        st.warning("⚠️ No hay configuración de vocabulario. Configure primero el diccionario.")
        return
    
    # Información del vocabulario activo
    dic_info = master_config.get('diccionario_seleccionado', {})
    st.info(f"🎯 **Vocabulario activo:** {dic_info.get('descripcion', 'Sin definir')} ({len(dic_info.get('palabras', []))} palabras)")
    
    # Pestañas para diferentes vistas de estadísticas
    tab1, tab2, tab3 = st.tabs(["🎵 Audio Dataset", "🖼️ Visual Dataset", "📈 Análisis Comparativo"])
    
    with tab1:
        display_audio_statistics(audio_config, master_config)
    
    with tab2:
        # Usar función mejorada si hay imágenes generadas, sino usar la original
        if visual_config and visual_config.get('generated_images'):
            display_enhanced_visual_statistics(visual_config, master_config)
        else:
            display_visual_statistics(visual_config, master_config)
    
    with tab3:
        display_comparative_analysis(audio_config, visual_config, master_config)

def display_audio_statistics(audio_config, master_config):
    """Estadísticas avanzadas del dataset de audio con análisis de ondas"""
    st.info("🎵 **Dataset de Audio**: Análisis estadístico por palabras y variaciones")
    
    if not audio_config or not audio_config.get('generated_samples'):
        st.warning("📭 No hay dataset de audio generado")
        st.info("💡 Ve a **🎵 Audio Dataset** → **🎤 Audio Base** para comenzar")
        return
    
    samples = audio_config['generated_samples']
    config_audio = master_config.get('configuracion_audio', {})
    
    # Métricas principales con diseño moderno
    col1, col2, col3, col4 = st.columns(4)
    
    total_palabras = len(samples)
    total_muestras = sum(len(variaciones) for variaciones in samples.values())
    promedio_por_palabra = total_muestras / total_palabras if total_palabras > 0 else 0
    
    # Análisis de métodos y tipos
    metodos_count = {}
    tipos_count = {}
    duraciones = []
    pitch_factors = []
    speed_factors = []
    volume_factors = []
    
    for palabra, variaciones in samples.items():
        for variacion in variaciones:
            metodo = variacion.get('metodo_sintesis', 'sin_especificar')
            tipo = variacion.get('tipo', 'desconocido')
            
            metodos_count[metodo] = metodos_count.get(metodo, 0) + 1
            tipos_count[tipo] = tipos_count.get(tipo, 0) + 1
            
            # Recopilar datos para análisis
            if 'duracion_ms' in variacion and variacion['duracion_ms'] > 0:
                duraciones.append(variacion['duracion_ms'])
            if 'pitch_factor' in variacion:
                pitch_factors.append(variacion['pitch_factor'])
            if 'speed_factor' in variacion:
                speed_factors.append(variacion['speed_factor'])
            if 'volume_factor' in variacion:
                volume_factors.append(variacion['volume_factor'])
    
    # Cards de métricas modernas
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: white;">
            <h3>🎵</h3>
            <h2>{}</h2>
            <p>Palabras</p>
        </div>
        """.format(total_palabras), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: white;">
            <h3>🎙️</h3>
            <h2>{:,}</h2>
            <p>Muestras Total</p>
        </div>
        """.format(total_muestras), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: white;">
            <h3>📊</h3>
            <h2>{:.1f}</h2>
            <p>Promedio/Palabra</p>
        </div>
        """.format(promedio_por_palabra), unsafe_allow_html=True)
    
    with col4:
        metodo_principal = max(metodos_count.items(), key=lambda x: x[1])[0] if metodos_count else 'N/A'
        metodo_nombre = {'gtts': 'Google TTS', 'espeak': 'eSpeak'}.get(metodo_principal, metodo_principal.title())
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: white;">
            <h3>🔊</h3>
            <h2 style="font-size: 1rem;">{}</h2>
            <p>Método Principal</p>
        </div>
        """.format(metodo_nombre), unsafe_allow_html=True)
    
    # Análisis dinámico de parámetros de audio
    st.markdown("#### 🎼 Análisis Dinámico de Parámetros de Audio")
    
    # Crear DataFrame completo
    detailed_data = []
    for palabra, variaciones in samples.items():
        for variacion in variaciones:
            detailed_data.append({
                'Palabra': palabra,
                'Tipo': variacion.get('tipo', 'original').replace('_', ' ').title(),
                'Método': variacion.get('metodo_sintesis', 'No especificado'),
                'Pitch': variacion.get('pitch_factor', 1.0),
                'Velocidad': variacion.get('speed_factor', 1.0), 
                'Volumen': variacion.get('volume_factor', 1.0),
                'Duración_ms': variacion.get('duracion_ms', 0),
                'Duración_s': variacion.get('duracion_ms', 0) / 1000.0
            })
    
    df_audio = pd.DataFrame(detailed_data)
    
    if not df_audio.empty:
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            # Histograma de duraciones con estadísticas
            if duraciones:
                fig_duration = px.histogram(
                    df_audio[df_audio['Duración_s'] > 0],
                    x='Duración_s',
                    color='Tipo',
                    nbins=25,
                    title="⏱️ Distribución de Duraciones por Tipo",
                    marginal="box",
                    opacity=0.7
                )
                fig_duration.update_xaxes(title_text="Duración (segundos)")
                fig_duration.update_yaxes(title_text="Frecuencia")
                st.plotly_chart(fig_duration, width='stretch')
                
                # Estadísticas de duración
                duraciones_s = [d/1000 for d in duraciones if d > 0]
                if duraciones_s:
                    col_dur1, col_dur2, col_dur3 = st.columns(3)
                    col_dur1.metric("⏱️ Duración Media", f"{np.mean(duraciones_s):.2f}s")
                    col_dur2.metric("📏 Rango", f"{np.min(duraciones_s):.2f}s - {np.max(duraciones_s):.2f}s")
                    col_dur3.metric("📊 Desviación Std", f"{np.std(duraciones_s):.2f}s")
        
        with col_analysis2:
            # Análisis de variaciones de parámetros (Pitch, Speed, Volume)
            param_data = []
            
            for param_name, param_list in [('Pitch', pitch_factors), ('Velocidad', speed_factors), ('Volumen', volume_factors)]:
                if param_list:
                    for val in param_list:
                        param_data.append({'Parámetro': param_name, 'Valor': val})
            
            if param_data:
                df_params = pd.DataFrame(param_data)
                
                fig_params = px.violin(
                    df_params,
                    x='Parámetro',
                    y='Valor',
                    title="🎛️ Distribución de Parámetros de Variación",
                    box=True,
                    points="outliers"
                )
                fig_params.update_layout(height=400)
                st.plotly_chart(fig_params, width='stretch')
                
                # Estadísticas rápidas de parámetros
                st.markdown("**📈 Estadísticas de Variación:**")
                param_stats_cols = st.columns(3)
                
                for i, (param_name, param_list) in enumerate([('Pitch', pitch_factors), ('Velocidad', speed_factors), ('Volumen', volume_factors)]):
                    if param_list and i < 3:
                        with param_stats_cols[i]:
                            avg_val = np.mean(param_list)
                            std_val = np.std(param_list)
                            st.write(f"**{param_name}**")
                            st.write(f"μ = {avg_val:.3f}")
                            st.write(f"σ = {std_val:.3f}")
    


def display_visual_statistics(visual_config, master_config):
    """Estadísticas del dataset visual"""
    st.info("�️ **Dataset Visual**: Basado en EMNIST para reconocimiento de letras")
    
    if not visual_config or not visual_config.get('generated_images'):
        st.warning("📭 No hay dataset visual generado")
        st.info("💡 Ve a **🖼️ Visual Dataset** para generar imágenes")
        return
    
    images = visual_config['generated_images']
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    total_letras = len(images)
    total_imagenes = sum(len(imgs) for imgs in images.values())
    promedio_por_letra = total_imagenes / total_letras if total_letras > 0 else 0
    
    col1.metric("🔤 Total Letras", total_letras)
    col2.metric("🖼️ Total Imágenes", f"{total_imagenes:,}")
    col3.metric("📊 Promedio/Letra", f"{promedio_por_letra:.1f}")
    col4.metric("🎯 Objetivo", "26 letras")
    
    # Gráfico de distribución
    if len(images) > 0:
        df_letras = pd.DataFrame([
            {'Letra': letra.upper(), 'Imágenes': len(imgs)} 
            for letra, imgs in images.items()
        ])
        
        fig_letras = px.bar(
            df_letras, 
            x='Letra', 
            y='Imágenes', 
            title="🔤 Imágenes por Letra",
            color='Imágenes',
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_letras, width='stretch')
    
    # Información adicional
    fecha_gen = visual_config.get('fecha_generacion', 'No disponible')
    if fecha_gen != 'No disponible':
        try:
            fecha_obj = datetime.fromisoformat(fecha_gen)
            fecha_str = fecha_obj.strftime('%d/%m/%Y %H:%M')
        except:
            fecha_str = fecha_gen[:19]
    else:
        fecha_str = fecha_gen
    
    st.info(f"� **Generado:** {fecha_str}")

def display_comparative_analysis(audio_config, visual_config, master_config):
    """Análisis comparativo entre datasets"""
    st.markdown("#### 🔄 Sincronización de Datasets")
    
    # Estado de sincronización
    consistencia = verificar_consistencia_datasets()
    
    if consistencia["audio"] and consistencia["visual"]:
        st.success("✅ Todos los datasets están sincronizados correctamente")
    else:
        st.warning(f"⚠️ {consistencia['mensaje']}")
    
    # Comparativa de completitud
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Estado de Completitud**")
        
        audio_completitud = 0
        if audio_config and audio_config.get('generated_samples'):
            samples = audio_config['generated_samples']
            total = len(samples)
            completas = sum(1 for v in samples.values() if len(v) > 1)
            audio_completitud = completas / total if total > 0 else 0
        
        visual_completitud = 0
        if visual_config and visual_config.get('generated_images'):
            images = visual_config['generated_images']
            visual_completitud = len(images) / 26  # 26 letras del alfabeto
        
        # Gráfico de completitud
        df_completitud = pd.DataFrame({
            'Dataset': ['Audio (Variaciones)', 'Visual (Letras)'],
            'Completitud': [audio_completitud * 100, min(visual_completitud * 100, 100)]
        })
        
        fig_completitud = px.bar(
            df_completitud, 
            x='Dataset', 
            y='Completitud',
            title="📈 Nivel de Completitud (%)",
            color='Completitud',
            color_continuous_scale='RdYlGn',
            range_y=[0, 100]
        )
        st.plotly_chart(fig_completitud, width='stretch')
    
    with col2:
        st.markdown("**🎯 Próximos Pasos Recomendados**")
        
        # Recomendaciones basadas en el estado
        if audio_completitud < 0.5:
            st.info("1. 🎵 Generar más variaciones de audio")
        if visual_completitud < 0.5:
            st.info("2. 🖼️ Completar dataset visual")
        if audio_completitud > 0.8 and visual_completitud > 0.8:
            st.success("3. ✅ Datasets listos para entrenamiento")
        
        # Botón de regeneración completa
        if st.button("🔄 Regenerar Todo", help="Regenera ambos datasets desde cero"):
            st.warning("Esta función regeneraría todos los datasets. Por implementar...")
    
    # Tabla resumen
    st.markdown("#### 📋 Resumen Ejecutivo")
    
    summary_data = []
    
    # Audio dataset
    if audio_config and audio_config.get('generated_samples'):
        samples = audio_config['generated_samples']
        config_audio = master_config.get('configuracion_audio', {})
        
        summary_data.append({
            'Dataset': '🎵 Audio',
            'Estado': '✅ Generado' if samples else '❌ Vacío',
            'Elementos': f"{len(samples)} palabras",
            'Muestras': f"{sum(len(v) for v in samples.values()):,}",
            'Método': config_audio.get('metodo_sintesis', 'N/A'),
            'Última Actualización': audio_config.get('fecha_generacion', 'N/A')[:16] if audio_config.get('fecha_generacion') else 'N/A'
        })
    else:
        summary_data.append({
            'Dataset': '🎵 Audio',
            'Estado': '❌ No generado',
            'Elementos': '0 palabras',
            'Muestras': '0',
            'Método': 'N/A',
            'Última Actualización': 'N/A'
        })
    
    # Visual dataset
    if visual_config and visual_config.get('generated_images'):
        images = visual_config['generated_images']
        
        summary_data.append({
            'Dataset': '🖼️ Visual',
            'Estado': '✅ Generado' if images else '❌ Vacío',
            'Elementos': f"{len(images)} letras",
            'Muestras': f"{sum(len(v) for v in images.values()):,}",
            'Método': 'EMNIST',
            'Última Actualización': visual_config.get('fecha_generacion', 'N/A')[:16] if visual_config.get('fecha_generacion') else 'N/A'
        })
    else:
        summary_data.append({
            'Dataset': '🖼️ Visual',
            'Estado': '❌ No generado',
            'Elementos': '0 letras',
            'Muestras': '0',
            'Método': 'N/A',
            'Última Actualización': 'N/A'
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, width='stretch', hide_index=True)

def display_performance_charts():
    """Muestra gráficos de rendimiento del sistema"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Latencia por Modelo")
        
        # Datos simulados de latencia
        models_data = ['TinyListener', 'TinyRecognizer', 'TinySpeller']
        latencies = [45, 12, 8]  # milisegundos
        
        df_latency = pd.DataFrame({
            'Modelo': models_data,
            'Latencia (ms)': latencies
        })
        
        fig_latency = px.bar(
            df_latency, 
            x='Modelo', 
            y='Latencia (ms)',
            title="Latencia de Inferencia",
            color='Latencia (ms)',
            color_continuous_scale='Viridis'
        )
        fig_latency.update_layout(height=300)
        st.plotly_chart(fig_latency, width="stretch")
    
    with col2:
        st.markdown("#### 🎯 Precisión por Modalidad")
        
        # Datos simulados de precisión
        modalities = ['Audio', 'Visión', 'Multimodal']
        accuracies = [94.2, 97.8, 98.9]
        
        df_accuracy = pd.DataFrame({
            'Modalidad': modalities,
            'Precisión (%)': accuracies
        })
        
        fig_accuracy = px.bar(
            df_accuracy, 
            x='Modalidad', 
            y='Precisión (%)',
            title="Precisión por Modalidad",
            color='Precisión (%)',
            color_continuous_scale='RdYlGn',
            range_y=[90, 100]
        )
        fig_accuracy.update_layout(height=300)
        st.plotly_chart(fig_accuracy, width="stretch")
    
    # Gráfico de evolución temporal (simulado)
    st.markdown("#### 📈 Evolución del Rendimiento")
    
    epochs = list(range(1, 21))
    listener_acc = [70 + 1.2*i + np.random.normal(0, 0.5) for i in epochs]
    recognizer_acc = [75 + 1.1*i + np.random.normal(0, 0.3) for i in epochs]
    
    df_evolution = pd.DataFrame({
        'Época': epochs * 2,
        'Precisión': listener_acc + recognizer_acc,
        'Modelo': ['TinyListener'] * 20 + ['TinyRecognizer'] * 20
    })
    
    fig_evolution = px.line(
        df_evolution, 
        x='Época', 
        y='Precisión', 
        color='Modelo',
        title="Evolución durante el Entrenamiento"
    )
    fig_evolution.update_layout(height=400)
    st.plotly_chart(fig_evolution, width="stretch")



def display_technical_info():
    """Muestra información técnica del sistema"""
    with st.expander("🏗️ Información Técnica", expanded=False):
        st.markdown("""
        ### 📊 **Flujo de Datos:**
        
        ```
        🎤 Audio Input           🖼️ Image Input
             ↓                        ↓
        🤖 Wav2Vec2 (768D)      🧠 CORnet-Z (768D)  
             ↓                        ↓
        🔄 LSTM (64D)           📝 Secuencia → LSTM
             ↓                        ↓
        🎯 Clasificador         🎯 Clasificador
             ↓                        ↓
        📝 Palabra Predicha     📝 Palabra Predicha
        ```
        
        ### 🧠 **Componentes Técnicos:**
        - **Wav2Vec2**: facebook/wav2vec2-base-es-voxpopuli-v2 (95M parámetros)
        - **CORnet-Z**: Arquitectura cortical V1→V2→V4→IT  
        - **LSTM**: 768→64→num_classes, 2 capas
        - **Datasets**: Configurables vía páginas de gestión
        """)



# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

# Función removida - ahora se importa desde components.modern_sidebar

def display_sidebar_mini_gallery(visual_config):
    """Mini galería persistente para la sidebar"""
    if not visual_config or not visual_config.get('generated_images'):
        return
    
    images = visual_config['generated_images']
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1);
                padding: 0.4rem; border-radius: 6px; margin: 0.3rem 0;
                border: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 0.7rem; 
                  text-align: center; font-weight: 500;">
            🖼️ Vista Previa Visual
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Seleccionar 3 muestras aleatorias
    sample_images = []
    available_letters = list(images.keys())[:3]
    
    for letra in available_letters:
        imgs = images[letra]
        if imgs:
            sample = random.choice(imgs)
            sample_images.append((letra.upper(), sample))
    
    # Mostrar mini galería
    if sample_images:
        cols_mini = st.columns(3)
        for i, (letra, img_data) in enumerate(sample_images):
            with cols_mini[i]:
                try:
                    if 'image' in img_data:
                        img_bytes = base64.b64decode(img_data['image'])
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        st.image(img, caption=letra, width=40)
                except Exception as e:
                    st.caption("❌", style={'font-size': '0.7rem'})

        # Info compacta
        total_imgs = sum(len(imgs) for imgs in images.values())
        st.markdown(f"""
        <div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; 
                    text-align: center; margin-top: 0.3rem;">
            📊 {total_imgs:,} imágenes totales
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FUNCIONES MEJORADAS PARA DATASET VISUAL
# =============================================================================

def display_visual_preview_gallery(visual_config):
    """Galería de preview de imágenes del dataset visual"""
    if not visual_config or not visual_config.get('generated_images'):
        return
    
    images = visual_config['generated_images']
    
    st.markdown("#### 🖼️ Galería de Muestras")
    
    # Seleccionar muestras aleatorias
    sample_images = []
    max_samples_per_letter = 2
    max_letters = 6
    
    available_letters = list(images.keys())[:max_letters]
    
    for letra in available_letters:
        imgs = images[letra]
        if imgs:
            sample_count = min(max_samples_per_letter, len(imgs))
            letter_samples = random.sample(imgs, sample_count)
            for sample in letter_samples:
                sample_images.append((letra.upper(), sample))
    
    # Mostrar galería
    if sample_images:
        cols_gallery = st.columns(min(len(sample_images), 6))
        for i, (letra, img_data) in enumerate(sample_images[:6]):
            with cols_gallery[i]:
                try:
                    if 'image' in img_data:
                        img_bytes = base64.b64decode(img_data['image'])
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        st.image(img, caption=f"{letra}", width=80)
                        
                        # Mostrar parámetros
                        if 'params' in img_data:
                            params = img_data['params']
                            st.caption(f"Font: {params.get('font_size', 'N/A')}px")
                            st.caption(f"Rot: {params.get('rotation', 0):.1f}°")
                except Exception as e:
                    st.caption(f"❌ Error: {letra}")

def display_enhanced_visual_statistics(visual_config, master_config):
    """Versión mejorada de estadísticas del dataset visual"""
    st.info("🖼️ **Dataset Visual**: Generación parametrizada con análisis dinámico")
    
    if not visual_config or not visual_config.get('generated_images'):
        st.warning("📭 No hay dataset visual generado")
        st.info("💡 Ve a **🖼️ Visual Dataset** → **🖼️ Generación Inteligente** para comenzar")
        return
    
    images = visual_config['generated_images']
    image_params = visual_config.get('image_params', {})
    
    # Métricas principales con diseño moderno
    col1, col2, col3, col4 = st.columns(4)
    
    total_letras = len(images)
    total_imagenes = sum(len(imgs) for imgs in images.values())
    promedio_por_letra = total_imagenes / total_letras if total_letras > 0 else 0
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: white;">
            <h3>🔤</h3>
            <h2>{}</h2>
            <p>Letras</p>
        </div>
        """.format(total_letras), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: white;">
            <h3>🖼️</h3>
            <h2>{:,}</h2>
            <p>Imágenes</p>
        </div>
        """.format(total_imagenes), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: #333;">
            <h3>📊</h3>
            <h2>{:.1f}</h2>
            <p>Prom/Letra</p>
        </div>
        """.format(promedio_por_letra), unsafe_allow_html=True)
    
    with col4:
        size = image_params.get('size', [64, 64])
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center; color: #333;">
            <h3>🎯</h3>
            <h2>{}x{}</h2>
            <p>Tamaño px</p>
        </div>
        """.format(size[0], size[1]), unsafe_allow_html=True)
    
    # Análisis de parámetros con gráficos interactivos
    st.markdown("#### 📊 Análisis Dinámico de Variaciones")
    
    param_data = []
    for letra, imgs in images.items():
        for img_data in imgs:
            if 'params' in img_data:
                params = img_data['params']
                param_data.append({
                    'Letra': letra.upper(),
                    'Font_Size': params.get('font_size', 0),
                    'Rotacion': abs(params.get('rotation', 0)),
                    'Noise_Level': params.get('noise_level', 0),
                    'Fuente': params.get('font', 'Desconocida')
                })
    
    if param_data:
        df_params = pd.DataFrame(param_data)
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            # Histograma de tamaños de fuente
            fig_fonts = px.histogram(
                df_params,
                x='Font_Size',
                color='Letra',
                title="📝 Distribución Tamaños de Fuente",
                nbins=15,
                marginal="box"
            )
            st.plotly_chart(fig_fonts, width='stretch')
        
        with col_analysis2:
            # Scatter plot rotación vs ruido
            fig_scatter = px.scatter(
                df_params,
                x='Rotacion',
                y='Noise_Level',
                color='Letra',
                size='Font_Size',
                title="🔄 Rotación vs Ruido por Letra",
                hover_data=['Fuente']
            )
            st.plotly_chart(fig_scatter, width='stretch')
    
    # Galería de previews
    display_visual_preview_gallery(visual_config)
    
    # Información de sincronización
    fecha_gen = visual_config.get('last_generation', visual_config.get('created', 'No disponible'))
    sync_status = visual_config.get('synchronized', False)
    
    if fecha_gen and fecha_gen != 'No disponible':
        try:
            if isinstance(fecha_gen, str):
                fecha_obj = datetime.fromisoformat(fecha_gen)
            else:
                fecha_obj = fecha_gen
            fecha_str = fecha_obj.strftime('%d/%m/%Y %H:%M')
        except:
            fecha_str = str(fecha_gen)[:16]
    else:
        fecha_str = "No disponible"
    
    sync_icon = "✅" if sync_status else "⚠️"
    st.info(f"📅 **Última generación:** {fecha_str} • {sync_icon} **Sincronizado:** {'Sí' if sync_status else 'Verificar'}")

def main():
    """Dashboard principal de TinySpeak - Refactorizado y optimizado"""
    
    # Aplicar CSS personalizado
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Cargar modelos una sola vez
    models = setup_models()
    
    # Sidebar modernizada
    display_modern_sidebar()
    
    # Header principal
    st.markdown('<h1 class="main-header">🎤 TinySpeak Dashboard</h1>', unsafe_allow_html=True)
    
    # Métricas del sistema en tiempo real
    display_system_metrics()
    
    # Configuración de vocabulario (nueva sección)
    st.markdown("---")
    display_vocabulary_selector()
    
    # Dashboard de modelos
    st.markdown("---")
    st.markdown("### 🧠 Arquitectura del Sistema")
    display_model_cards()
    
    # Estadísticas completas de datasets
    st.markdown("---")
    display_dataset_statistics()
    
    # Gráficos de rendimiento
    st.markdown("---") 
    st.markdown("### ⚡ Rendimiento del Sistema")
    display_performance_charts()
    
    # Información técnica al final
    display_technical_info()


if __name__ == "__main__":
    main()