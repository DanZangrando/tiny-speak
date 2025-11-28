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
# Importar módulos del proyecto
from models import PhonologicalPathway, VisualPathway
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

# =============================================================================
# CONFIGURACIÓN Y SETUP
# =============================================================================

from training.config import load_master_dataset_config, save_master_dataset_config

def render_experiment_config():
    """Renderiza la configuración global del experimento en la página principal."""
    st.markdown("### 🛠️ Configuración del Experimento")
    st.info("Define aquí los parámetros globales para el experimento de transparencia.")
    
    # Cargar configuración actual
    try:
        config = load_master_dataset_config()
    except FileNotFoundError:
        config = {}
        
    exp_config = config.get("experiment_config", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selector de Idiomas
        st.markdown("#### 1. Idiomas del Experimento")
        langs = st.multiselect(
            "Selecciona los idiomas a comparar:",
            ['es', 'en', 'fr'],
            default=exp_config.get('languages', ['es']),
            format_func=lambda x: {'es': '🇪🇸 Español', 'en': '🇺🇸 English', 'fr': '🇫🇷 Français'}[x]
        )
        
    with col2:
        # Selector de Diccionario Base
        st.markdown("#### 2. Diccionario Base")
        dict_names = get_nombres_diccionarios()
        current_dict = exp_config.get('base_dictionary', dict_names[0] if dict_names else None)
        if current_dict not in dict_names and dict_names:
            current_dict = dict_names[0]
            
        base_dict = st.selectbox(
            "Selecciona el diccionario base:",
            dict_names,
            index=dict_names.index(current_dict) if current_dict in dict_names else 0
        )

    # Validación
    st.markdown("#### 3. Validación de Recursos")
    if not langs:
        st.warning("⚠️ Debes seleccionar al menos un idioma.")
        return

    all_valid = True
    status_container = st.container()
    
    with status_container:
        cols = st.columns(len(langs))
        for i, lang in enumerate(langs):
            with cols[i]:
                # Verificar diccionario
                dic_data = get_diccionario_predefinido(base_dict, idioma=lang)
                if not dic_data:
                    st.error(f"❌ {lang.upper()}: Falta diccionario")
                    all_valid = False
                else:
                    words = dic_data['palabras']
                    st.success(f"✅ {lang.upper()} ({len(words)})")
                    with st.expander(f"📖 Ver Vocabulario ({lang.upper()})"):
                        st.write(", ".join(words))

    if st.button("💾 Guardar Configuración Global", type="primary", disabled=not all_valid):
        new_config = config.copy()
        new_config["experiment_config"] = {
            "languages": langs,
            "base_dictionary": base_dict,
            "last_updated": datetime.now().isoformat()
        }
        # También actualizamos el diccionario seleccionado por defecto para compatibilidad
        # Cargamos el diccionario base (español por defecto o el primero seleccionado) para el uso general
        default_lang = langs[0] if 'es' not in langs else 'es'
        dic_def = get_diccionario_predefinido(base_dict, idioma=default_lang)
        if dic_def:
            # Añadir 'tipo' para compatibilidad con código legado
            dic_def['tipo'] = 'predefinido'
            new_config["diccionario_seleccionado"] = dic_def
            
        save_master_dataset_config(new_config)
        st.toast("Configuración guardada exitosamente!", icon="🎉")
        st.rerun()

@st.cache_resource
def setup_models(autoload: bool = False):

    """Inicializa (o prepara) los modelos y configuración del sistema.

    Para evitar cargas pesadas en el dashboard principal, por defecto
    `autoload=False` y la función retorna la información ligera (device, words)
    sin cargar pesos grandes. Las páginas de entrenamiento/inferencia pueden
    llamar `setup_models(autoload=True)` para forzar la carga completa.
    """
    device = encontrar_device()
    # st.sidebar.info(f"Dispositivo detectado: {device}")

    # Obtener palabras por defecto
    words = get_default_words()

    if not autoload:
        # Retornar estructura mínima (sin pesos cargados)
        return {
            'device': device,
            'wav2vec_model': None,
            'phonological_pathway': None,
            'visual_pathway': None,
            'tiny_speller': None,
            'words': words
        }

    # Si se solicita autoload, cargar los modelos pesados
    with st.spinner("Cargando modelo Wav2Vec2 y modelos locales (esto puede tardar)..."):
        wav2vec_model = load_wav2vec_model(device=device)

        # Inicializar modelos
        # PhonologicalPathway ya no necesita TinySpeak, es autocontenido
        phonological_pathway = PhonologicalPathway(num_classes=len(words))
        visual_pathway = VisualPathway(num_classes=len(words))

        # Mover modelos al dispositivo
        phonological_pathway = phonological_pathway.to(device)
        visual_pathway = visual_pathway.to(device)

    return {
        'device': device,
        'wav2vec_model': wav2vec_model,
        'phonological_pathway': phonological_pathway,
        'visual_pathway': visual_pathway,
        'words': words
    }

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_custom_css():
    """Retorna CSS personalizado para el tema moderno"""
    return """
    <style>
    /* Estilos mínimos para contenedores */
    .stDataFrame {
        border-radius: 10px;
    }
    </style>
    """

def create_model_metrics():
    """Crea las métricas simuladas de los modelos"""
    return {
        'PhonologicalPathway': {'params': '~2.1M', 'accuracy': '94.2%', 'delta': '2.1%'},
        'VisualPathway': {'params': '~850K', 'accuracy': '97.8%', 'delta': '1.5%'}
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
        dic_sel = master_config['diccionario_seleccionado']
        
        # Compatibilidad con formato antiguo (tipo/nombre)
        if 'tipo' in dic_sel:
            dic_tipo = dic_sel['tipo']
            if dic_tipo == 'predefinido':
                dic_nombre = dic_sel['nombre']
                # Si el nombre tiene formato "Nombre (LANG)", extraer solo el nombre base si es necesario
                # Pero get_diccionario_predefinido espera la clave (ej: 'animales')
                # Por ahora intentamos usar el nombre tal cual o buscar en metadatos
                # Mejor: si ya tiene 'palabras', usarlas directo
                if 'palabras' in dic_sel and dic_sel['palabras']:
                    return dic_sel['palabras']
                    
                dic_data = get_diccionario_predefinido(dic_nombre)
                if dic_data:
                    return dic_data['palabras']
            elif dic_tipo == 'personalizado':
                return dic_sel.get('palabras', [])
        
        # Nuevo formato (directamente el objeto diccionario)
        elif 'palabras' in dic_sel:
            return dic_sel['palabras']
            
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
    
    # Verificar dataset de audio (reutilizar master_config ya cargado)
    if master_config and 'generated_samples' in master_config:
        samples = master_config['generated_samples']
        
        # Detectar y aplanar si es necesario para obtener las palabras
        palabras_audio = set()
        
        is_nested = False
        if samples:
            first_val = next(iter(samples.values()))
            if isinstance(first_val, dict) and not isinstance(first_val, list):
                is_nested = True
        
        if is_nested:
            for lang_data in samples.values():
                if isinstance(lang_data, dict):
                    palabras_audio.update(lang_data.keys())
        else:
            palabras_audio = set(samples.keys())
        
        # Verificar si hay palabras generadas y si coinciden
        if not master_config['generated_samples']:
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
        metodo_dataset = master_config.get('configuracion_audio', {}).get('metodo_sintesis', 
                                        master_config.get('metodo_sintesis', 'desconocido'))
        if metodo_dataset != metodo_configurado and metodo_dataset != 'desconocido':
            resultado["audio"] = False
            resultado["mensaje"] = f"Método de síntesis desincronizado: {metodo_dataset} vs {metodo_configurado}"
    else:
        resultado["audio"] = False
        resultado["mensaje"] = "No se encontró dataset de audio generado"
    
    # Verificar dataset visual (reutilizar master_config ya cargado)
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
        if st.button("🔄 Sincronizar", type="primary", help="Actualiza el vocabulario activo"):
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
        # st.markdown("---")
        # st.info("⚙️ Para configuración detallada de parámetros, ve a las páginas de Audio Dataset y Visual Dataset")

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
    
    # Datasets con verificación de contenido real (cargar config una sola vez)
    master_config = load_dataset_config("master_dataset_config.json")
    
    # Audio dataset
    if master_config and 'generated_samples' in master_config:
        total_audio_samples = 0
        samples_data = master_config['generated_samples']
        
        # Detectar estructura (plana vs anidada)
        is_nested = False
        if samples_data:
            first_val = next(iter(samples_data.values()))
            if isinstance(first_val, dict) and not isinstance(first_val, list):
                is_nested = True
        
        if is_nested:
            # Estructura anidada: idioma -> palabra -> lista
            for lang_data in samples_data.values():
                if isinstance(lang_data, dict):
                    for word_variations in lang_data.values():
                        total_audio_samples += len(word_variations)
        else:
            # Estructura plana antigua: palabra -> lista
            for word_variations in samples_data.values():
                if isinstance(word_variations, list):
                    total_audio_samples += len(word_variations)
                    
        audio_status = f"✅ ({total_audio_samples})"
    else:
        audio_status = "⚙️"
    
    # Visual dataset
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
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
        <h4>🎵 Phonological Pathway</h4>
        <p><strong>Audio → Palabra</strong></p>
        <ul>
        <li>🤖 <strong>Feature Extractor:</strong> Custom CNN</li>
        <li>🧠 <strong>Encoder:</strong> Transformer (2 Layers)</li>
        <li>🎯 <strong>Output:</strong> Linear Classification Head</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="model-card">
        <h4>🖼️ Visual Pathway</h4>
        <p><strong>Imagen → Letra</strong></p>
        <ul>
        <li>🧠 <strong>Backbone:</strong> Custom CNN (V1→V2→V4→IT)</li>
        <li>🔄 <strong>Decoder:</strong> AvgPool → Flatten → Linear</li>
        <li>🎯 <strong>Output:</strong> Linear Classification Head</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)



def display_dataset_statistics():
    """Estadísticas detalladas y completas de los datasets generados"""
    st.markdown("### 📊 Estadísticas de los Datasets")
    
    # Cargar configuraciones
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
        # Usar función mejorada si hay imágenes generadas en master_config
        images_exist = (master_config and 
                       master_config.get('visual_dataset', {}).get('generated_images', {}))
        
        if images_exist:
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
    
    # Detectar estructura anidada (idioma -> palabra -> lista) y aplanar si es necesario
    is_nested = False
    if samples:
        first_val = next(iter(samples.values()))
        if isinstance(first_val, dict) and not isinstance(first_val, list):
            is_nested = True
            
    if is_nested:
        # Aplanar estructura: combinar todos los idiomas
        flat_samples = {}
        for lang, words_data in samples.items():
            if isinstance(words_data, dict):
                for word, vars_list in words_data.items():
                    flat_samples[word] = vars_list
        samples = flat_samples
    
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
        
    # Preparar samples de audio (aplanando si es necesario)
    audio_samples = audio_config.get('generated_samples', {}) if audio_config else {}
    if audio_samples:
        first_val = next(iter(audio_samples.values()))
        if isinstance(first_val, dict) and not isinstance(first_val, list):
            # Estructura anidada: idioma -> palabra -> lista
            flat_samples = {}
            for lang, words_data in audio_samples.items():
                if isinstance(words_data, dict):
                    for word, vars_list in words_data.items():
                        flat_samples[word] = vars_list
            audio_samples = flat_samples
    
    # Comparativa de completitud
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Estado de Completitud**")
        
        audio_completitud = 0
        if audio_samples:
            total = len(audio_samples)
            completas = sum(1 for v in audio_samples.values() if len(v) > 1)
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
    # Audio dataset
    if audio_samples:
        config_audio = master_config.get('configuracion_audio', {})
        
        summary_data.append({
            'Dataset': '🎵 Audio',
            'Estado': '✅ Generado',
            'Elementos': f"{len(audio_samples)} palabras",
            'Muestras': f"{sum(len(v) for v in audio_samples.values()):,}",
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
        models_data = ['TinyListener', 'TinyRecognizer']
        latencies = [45, 12]  # milisegundos
        
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
        modalities = ['Audio', 'Visión']
        accuracies = [94.2, 97.8]
        
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

# Función removida - mini galería del sidebar eliminada

# =============================================================================
# FUNCIONES MEJORADAS PARA DATASET VISUAL
# =============================================================================

def display_visual_preview_gallery(visual_config, master_config=None):
    """Galería de preview de imágenes del dataset visual"""
    # Buscar imágenes en el lugar correcto
    if master_config:
        images = master_config.get('visual_dataset', {}).get('generated_images', {})
    else:
        images = visual_config.get('generated_images', {}) if visual_config else {}
    
    if not images:
        return
    
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
                    img = None
                    
                    # Nuevo sistema: cargar desde archivo (PRIORITARIO)
                    if 'file_path' in img_data and img_data['file_path']:
                        from pathlib import Path
                        from PIL import Image as PILImage
                        
                        base_dir = Path(__file__).parent
                        image_path = base_dir / img_data['file_path']
                        if image_path.exists():
                            img = PILImage.open(image_path)
                    
                    # Sistema legacy: cargar desde base64 (SOLO SI NO HAY ARCHIVO)
                    elif 'image' in img_data and img_data['image'] and img_data['image'] != None:
                        img_bytes = base64.b64decode(img_data['image'])
                        img = Image.open(io.BytesIO(img_bytes))
                    
                    if img:
                        st.image(img, caption=f"{letra}", width=80)
                        
                        # Mostrar parámetros
                        if 'params' in img_data:
                            params = img_data['params']
                            st.caption(f"Font: {params.get('font_size', 'N/A')}px")
                            st.caption(f"Rot: {params.get('rotation', 0):.1f}°")
                    else:
                        st.caption(f"❌ {letra} (no disponible)")
                        
                except Exception as e:
                    st.caption(f"❌ Error: {letra}")

def display_enhanced_visual_statistics(visual_config, master_config):
    """Versión mejorada de estadísticas del dataset visual"""
    st.info("🖼️ **Dataset Visual**: Generación parametrizada con análisis dinámico")
    
    # Buscar imágenes en el lugar correcto del master_config
    images = master_config.get('visual_dataset', {}).get('generated_images', {}) if master_config else {}
    
    if not images:
        st.warning("📭 No hay dataset visual generado")
        st.info("💡 Ve a **🖼️ Visual Dataset** → **🖼️ Generación Inteligente** para comenzar")
        return
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
    display_visual_preview_gallery(visual_config, master_config)
    
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
    
    # Preparar (sin cargar pesos pesados por defecto) — las páginas cargarán cuando lo necesiten
    models = setup_models(autoload=False)
    
    # Sidebar modernizada
    display_modern_sidebar("dashboard")
    
    # Header principal
    st.markdown('<h1 class="main-header">🎤 TinySpeak Dashboard</h1>', unsafe_allow_html=True)
    
    # Renderizar configuración del experimento
    render_experiment_config()
    
    st.markdown("---")
    
    # Métricas del sistema en tiempo real
    display_system_metrics()
    
    # Dashboard de modelos
    st.markdown("---")
    st.markdown("### 🧠 Arquitectura del Sistema")
    display_model_cards()
    
    # Estadísticas completas de datasets
    st.markdown("---")
    display_dataset_statistics()
    
    # Gráficos de rendimiento (Desactivados)
    # st.markdown("---") 
    # st.markdown("### ⚡ Rendimiento del Sistema")
    # display_performance_charts()
    
    # Información técnica al final (Desactivada)
    # display_technical_info()


if __name__ == "__main__":
    main()