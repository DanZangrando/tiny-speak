"""
🎤 Audio Dataset Manager - Generación y Gestión de Datasets de Audio
Página para generar, modificar y gestionar datasets de audio usando Google Text-to-Speech
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import librosa
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize
import io
import base64
import random

# change_speed movido a utils.py

# Importar módulos
from utils import (
    encontrar_device, WAV2VEC_SR, get_default_words,
    change_speed, generar_audio_gtts, save_audio_file, generar_audio_espeak,
    generar_audio_segun_metodo, aplicar_variaciones_audio, generar_variaciones_completas
)

# Importar diccionarios predefinidos
from diccionarios import DICCIONARIOS_PREDEFINIDOS, get_diccionario_predefinido

# Importar sidebar moderna
import sys
sys.path.append(str(Path(__file__).parent.parent))
from components.modern_sidebar import display_modern_sidebar

# Importar funciones desde el módulo principal
sys.path.append('..')
try:
    from app import save_dataset_config
except ImportError:
    # Función alternativa si no se puede importar
    def save_dataset_config(config, config_path):
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"Error guardando configuración: {e}")
            return False

# Configurar página
st.set_page_config(
    page_title="Audio Dataset Manager",
    page_icon="👂",
    layout="wide"
)

def get_custom_css():
    """CSS moderno para la página de Audio Dataset"""
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .config-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .sync-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(240, 147, 251, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .status-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(79, 172, 254, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .generation-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(168, 237, 234, 0.3);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        color: #333;
    }
    </style>
    """

def load_master_config():
    """Cargar configuración maestra del proyecto"""
    try:
        config_path = Path('master_dataset_config.json')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error cargando configuración maestra: {e}")
        return {}

def load_audio_dataset_config():
    """Cargar configuración del dataset de audio"""
    try:
        config_path = Path('master_dataset_config.json')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error cargando configuración: {e}")
        return {}

def convert_numpy_types(obj):
    """Convierte tipos numpy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_audio_dataset_config(config):
    """Guardar configuración del dataset de audio"""
    try:
        # Convertir tipos numpy antes de guardar
        config_clean = convert_numpy_types(config)
        
        config_path = Path('master_dataset_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_clean, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error guardando configuración: {e}")
        return False

def limpiar_dataset_anterior():
    """Limpia el dataset anterior para evitar acumulación de datos"""
    try:
        config_path = Path('master_dataset_config.json')
        
        # Cargar configuración actual para preservar otros campos
        current_config = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                current_config = json.load(f)
        
        # Limpiar solo los samples generados
        current_config['generated_samples'] = {}
        current_config['last_update'] = datetime.now().isoformat()
        current_config['dataset_cleaned'] = True
        
        # Guardar configuración actualizada
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(current_config, f, indent=4, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"Error limpiando dataset anterior: {e}")
        return False

# Funciones de generación de audio movidas a utils.py

def configuracion_completa_audio():
    """Interfaz completa para configurar la generación de audio"""
    
    st.markdown("### ⚙️ Configuración de Generación de Audio")
    
    # Cargar configuración existente
    config = load_audio_dataset_config()
    master_config = load_master_config()
    
    # Usar configuración de master_config si existe, sino usar configuracion_audio del config local
    master_audio_config = master_config.get('configuracion_audio', {})
    local_audio_config = config.get('configuracion_audio', config.get('configuracion', {}))
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.markdown("#### 🎤 Parámetros de Síntesis")
        
        # Cargar configuración global del experimento
        exp_config = master_config.get('experiment_config', {})
        langs = exp_config.get('languages', ['es'])
        base_dict = exp_config.get('base_dictionary', 'animales')
        
        st.info(f"🌍 **Idiomas Configurados:** {', '.join([l.upper() for l in langs])}")
        st.info(f"📚 **Diccionario Base:** {base_dict}")
        
        # Número de variaciones (prioridad: master config > local config > default 3)
        # Asegurar que sea int
        try:
            default_num_variaciones = int(master_audio_config.get('num_variaciones', 
                                     local_audio_config.get('num_variaciones', 3)))
        except:
            default_num_variaciones = 3
            
        num_variaciones = st.slider(
            "Número de variaciones por palabra:",
            min_value=1, max_value=10, 
            value=default_num_variaciones,
            help="Cada palabra tendrá este número de variaciones además del audio original"
        )
        
        # Método de síntesis (de master config si existe)
        default_metodo = master_audio_config.get('metodo_sintesis',
                        local_audio_config.get('metodo_sintesis', 'gtts'))
        
        metodo_sintesis = st.selectbox(
            "Método de síntesis:",
            ['gtts', 'espeak'],
            index=0 if default_metodo == 'gtts' else 1,
            help="gTTS: Google Text-to-Speech (requiere internet)\nespeak: Síntesis local (requiere instalación)"
        )
        
        # Idioma ya no se selecciona aquí, viene de la configuración global
    
    with col_config2:
        st.markdown("#### 🎛️ Rangos de Variabilidad")
        
        # Usar rangos de master config si existe, sino usar defaults
        master_rangos = master_audio_config.get('rangos_variabilidad', {})
        rangos_actuales = local_audio_config.get('rangos', {})
        
        # Convertir rangos del master config al formato esperado
        if master_rangos:
            default_ranges = {
                'pitch': [master_rangos.get('pitch_bajo', [0.7, 0.9])[0], 
                         master_rangos.get('pitch_alto', [1.1, 1.3])[1]],
                'speed': [master_rangos.get('velocidad_lenta', [0.7, 0.9])[0], 
                         master_rangos.get('velocidad_rapida', [1.1, 1.4])[1]],
                'volume': [0.8, 1.2]  # Default para volumen
            }
        else:
            default_ranges = {
                'pitch': [0.8, 1.3],
                'speed': [0.7, 1.4],
                'volume': [0.8, 1.2]
            }
        
        # Usar rangos actuales si existen, sino usar defaults
        rangos_finales = {
            'pitch': rangos_actuales.get('pitch', default_ranges['pitch']),
            'speed': rangos_actuales.get('speed', default_ranges['speed']),
            'volume': rangos_actuales.get('volume', default_ranges['volume'])
        }
        
        # Controles de rango para pitch
        pitch_min, pitch_max = st.slider(
            "Rango de Pitch (Factor):",
            min_value=0.5, max_value=2.0, 
            value=(rangos_finales['pitch'][0], rangos_finales['pitch'][1]),
            step=0.1,
            help="Factor de modificación del tono de voz"
        )
        
        # Controles de rango para velocidad
        speed_min, speed_max = st.slider(
            "Rango de Velocidad (Factor):",
            min_value=0.3, max_value=3.0,
            value=(rangos_finales['speed'][0], rangos_finales['speed'][1]),
            step=0.1,
            help="Factor de modificación de la velocidad de habla"
        )
        
        # Controles de rango para volumen
        volume_min, volume_max = st.slider(
            "Rango de Volumen (Factor):",
            min_value=0.3, max_value=2.0,
            value=(rangos_finales['volume'][0], rangos_finales['volume'][1]),
            step=0.1,
            help="Factor de modificación del volumen"
        )
    
    # Crear objeto de rangos
    rangos = {
        'pitch': [pitch_min, pitch_max],
        'speed': [speed_min, speed_max],
        'volume': [volume_min, volume_max]
    }
    
    # Botones de acción
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("💾 Guardar Configuración", type="primary"):
            nueva_config = {
                'configuracion_audio': {
                    'num_variaciones': num_variaciones,
                    'metodo_sintesis': metodo_sintesis,
                    # 'idioma': idioma_audio, # Ya no se guarda un solo idioma aquí
                    'rangos_variabilidad': {
                        'pitch_bajo': [rangos_finales['pitch'][0], rangos_finales['pitch'][0]], # Hack para mantener estructura si es necesario o simplificar
                        'pitch_alto': [rangos_finales['pitch'][1], rangos_finales['pitch'][1]],
                        'velocidad_lenta': [rangos_finales['speed'][0], rangos_finales['speed'][0]],
                        'velocidad_rapida': [rangos_finales['speed'][1], rangos_finales['speed'][1]]
                    },
                    'rangos': rangos_finales
                }
            }
            
            # Actualizar master config preservando otros campos
            master_config.update(nueva_config)
            if save_audio_dataset_config(master_config):
                st.success("Configuración guardada exitosamente")
                # st.rerun() # Opcional, a veces mejor no recargar todo
            
    with col_btn2:
        if st.button("🔄 Restablecer Defaults"):
            config_default = {
                'configuracion': {
                    'num_variaciones': 3,
                    'metodo_sintesis': 'gtts',
                    'idioma': 'es',
                    'rangos': {
                        'pitch': [0.8, 1.3],
                        'speed': [0.7, 1.4],
                        'volume': [0.8, 1.2]
                    }
                }
            }
            save_audio_dataset_config(config_default)
            st.success("✅ Configuración restablecida")
            st.rerun()
    
    with col_btn3:
        if st.button("📋 Vista Previa"):
            st.info("🎵 Configuración actual:")
            st.json({
                'num_variaciones': num_variaciones,
                'metodo_sintesis': metodo_sintesis,
                'idiomas': langs, # Usar la lista de idiomas globales
                'rangos': rangos
            })
    
    # Mostrar configuración actual
    if config.get('configuracion'):
        st.markdown("---")
        st.markdown("#### 📊 Configuración Actual")
        
        config_actual = config['configuracion']
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown(f"""
            <div class="config-card">
                <h4>🎤 Síntesis</h4>
                <p><strong>Método:</strong> {config_actual.get('metodo_sintesis', 'N/A').upper()}</p>
                <p><strong>Idiomas:</strong> {', '.join([l.upper() for l in langs])}</p>
                <p><strong>Variaciones:</strong> {config_actual.get('num_variaciones', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info2:
            rangos_cfg = config_actual.get('rangos', {})
            st.markdown(f"""
            <div class="config-card">
                <h4>🎛️ Rangos</h4>
                <p><strong>Pitch:</strong> {rangos_cfg.get('pitch', [0, 0])[0]:.1f} - {rangos_cfg.get('pitch', [0, 0])[1]:.1f}</p>
                <p><strong>Speed:</strong> {rangos_cfg.get('speed', [0, 0])[0]:.1f} - {rangos_cfg.get('speed', [0, 0])[1]:.1f}</p>
                <p><strong>Volume:</strong> {rangos_cfg.get('volume', [0, 0])[0]:.1f} - {rangos_cfg.get('volume', [0, 0])[1]:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info3:
            # Estadísticas del dataset
            samples = config.get('generated_samples', {})
            total_palabras = len(samples)
            total_muestras = sum(len(variaciones) for variaciones in samples.values())
            
            st.markdown(f"""
            <div class="status-card">
                <h4>📊 Dataset Actual</h4>
                <p><strong>Palabras:</strong> {total_palabras}</p>
                <p><strong>Muestras:</strong> {total_muestras}</p>
                <p><strong>Estado:</strong> {'✅ Listo' if total_muestras > 0 else '⚠️ Vacío'}</p>
            </div>
            """, unsafe_allow_html=True)

def guardar_configuracion_audio(num_variaciones, metodo_sintesis, idioma_audio, rangos, master_config):
    """Guarda la configuración de audio"""
    try:
        config = load_audio_dataset_config()
        
        # Actualizar configuración en el formato esperado por app.py
        if 'configuracion_audio' not in config:
            config['configuracion_audio'] = {}
        
        config['configuracion_audio'].update({
            'num_variaciones': num_variaciones,
            'metodo_sintesis': metodo_sintesis,
            'idioma': idioma_audio,
            'rangos': rangos
        })
        
        # También mantener la estructura anterior para compatibilidad
        if 'configuracion' not in config:
            config['configuracion'] = {}
        
        config['configuracion'].update({
            'num_variaciones': num_variaciones,
            'metodo_sintesis': metodo_sintesis,
            'idioma': idioma_audio,
            'rangos': rangos
        })
        
        # Guardar
        return save_audio_dataset_config(config)
        
    except Exception as e:
        st.error(f"Error guardando configuración: {e}")
        return False

def generar_audio_base():
    """Interfaz para generar audios base del dataset (Multi-idioma)"""
    
    st.markdown("### 🎤 Generación de Audio Base")
    
    # Cargar configuraciones
    config = load_audio_dataset_config()
    master_config = load_master_config()
    exp_config = master_config.get('experiment_config', {})
    
    if not exp_config:
        st.warning("⚠️ No hay configuración de experimento. Configura primero en la Home.")
        return
        
    langs = exp_config.get('languages', [])
    base_dict = exp_config.get('base_dictionary', '')
    
    if not langs or not base_dict:
        st.error("❌ Configuración incompleta. Revisa la Home.")
        return

    # Mostrar estado por idioma
    st.markdown("#### 📊 Estado de Datasets por Idioma")
    
    status_cols = st.columns(len(langs))
    words_per_lang = {}
    
    for i, lang in enumerate(langs):
        with status_cols[i]:
            # Obtener diccionario para este idioma
            dic_data = get_diccionario_predefinido(base_dict, idioma=lang)
            if not dic_data:
                st.error(f"❌ {lang.upper()}: Falta diccionario")
                continue
                
            words = dic_data['palabras']
            words_per_lang[lang] = words
            
            # Verificar en disco
            missing = []
            for w in words:
                path = Path(f"data/audios/{lang}/{w}")
                if not path.exists() or not list(path.glob("*.wav")):
                    missing.append(w)
            
            if missing:
                st.warning(f"⚠️ {lang.upper()}: Faltan {len(missing)}/{len(words)}")
            else:
                st.success(f"✅ {lang.upper()}: Completo ({len(words)})")

    st.markdown("---")
    
    # Opciones de generación
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        limpiar_dataset = st.checkbox("🗑️ Limpiar dataset anterior antes de generar", value=True, 
                                    help="Si se marca, se borrarán las referencias a audios anteriores en el archivo de configuración.")
    
    # Botón de generación masiva
    if st.button("🚀 Generar Audios para Todos los Idiomas", type="primary"):
        # Leer parámetros de síntesis
        config_audio = config.get('configuracion_audio', {})
        num_vars = config_audio.get('num_variaciones', 3)
        metodo = config_audio.get('metodo_sintesis', 'gtts')
        rangos = config_audio.get('rangos', {})
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tasks = sum(len(words) for words in words_per_lang.values())
        current_task = 0
        
        # Inicializar/Limpiar estructura de samples
        if 'generated_samples' not in master_config or limpiar_dataset:
            master_config['generated_samples'] = {}

        for lang in langs:
            words = words_per_lang.get(lang, [])
            if not words: continue
            
            # Inicializar estructura para el idioma
            if lang not in master_config['generated_samples']:
                master_config['generated_samples'][lang] = {}
            
            status_text.write(f"⏳ Procesando {lang.upper()}...")
            
            for word in words:
                try:
                    # Generar variaciones y capturar metadatos
                    variaciones = generar_variaciones_completas(
                        texto=word,
                        idioma=lang,
                        num_variaciones=num_vars,
                        metodo_sintesis=metodo,
                        dataset_name=lang, # Usar código de idioma como nombre de dataset (carpeta)
                        rangos=rangos
                    )
                    
                    # Guardar metadatos en la estructura anidada
                    if variaciones:
                        master_config['generated_samples'][lang][word] = variaciones
                        
                except Exception as e:
                    st.error(f"Error generando {word} ({lang}): {e}")
                
                current_task += 1
                progress_bar.progress(min(current_task / total_tasks, 1.0))
                
        status_text.success("✅ Generación completada para todos los idiomas!")
        st.balloons()
        
        # Actualizar timestamp y guardar configuración completa
        master_config['last_update'] = datetime.now().isoformat()
        
        # Actualizar diccionario seleccionado para mantener sincronización
        # Esto asegura que la sidebar y otras partes sepan qué diccionario se usó
        master_config['diccionario_seleccionado'] = {
            "tipo": "predefinido",
            "nombre": base_dict,
            "idiomas": langs,
            "palabras_total": sum(len(w) for w in words_per_lang.values())
        }
        
        save_dataset_config(master_config, 'master_dataset_config.json')
        st.rerun()

    # Fin de la función de generación
    pass
        
    # Limpieza de código legado: La lógica de selección manual de palabras ha sido reemplazada
    # por la generación automática basada en la configuración global.
    
    st.markdown("---")
    st.info("ℹ️ Para modificar el vocabulario, ve a la página principal (Home) y selecciona un diccionario diferente.")

def generar_proceso_completo(palabras, config_audio, limpiar_anterior=False):
    """
    Genera el dataset completo para todas las palabras
    """
    palabras_procesadas = 0
    
    # Si se solicita limpiar el dataset anterior
    if limpiar_anterior:
        with st.status("🧹 Limpiando dataset anterior...", expanded=True) as status:
            if limpiar_dataset_anterior():
                st.write("✅ Dataset anterior limpiado correctamente")
                status.update(label="✅ Dataset limpiado", state="complete")
            else:
                st.error("❌ Error al limpiar dataset anterior")
                status.update(label="❌ Error en limpieza", state="error")
                return False
    
    config = load_audio_dataset_config()
    if not config:
        config = {'generated_samples': {}}
    
    # Asegurar que generated_samples existe
    if 'generated_samples' not in config:
        config['generated_samples'] = {}
    
    for palabra in palabras:
        with st.status(f"Procesando palabra: {palabra}", expanded=True) as status:
            try:
                st.write(f"🎵 Generando variaciones para: **{palabra}**")
                
                # Obtener nombre del dataset
                master_config = load_master_config()
                dataset_name = "custom_dataset"
                if master_config and 'diccionario_seleccionado' in master_config:
                    dataset_name = master_config['diccionario_seleccionado'].get('nombre', 'custom_dataset')
                
                # Generar variaciones para esta palabra
                variaciones = generar_variaciones_completas(
                    palabra,
                    config_audio['idioma'],
                    config_audio['num_variaciones'],
                    config_audio['metodo_sintesis'],
                    dataset_name
                )
                
                # Guardar en configuración
                config['generated_samples'][palabra] = variaciones
                
                st.write(f"✅ Completado: {len(variaciones)} variaciones generadas")
                status.update(label=f"✅ Completada: {palabra}", state="complete")
                palabras_procesadas += 1
                
            except Exception as e:
                st.error(f"❌ Error procesando {palabra}: {e}")
                status.update(label=f"❌ Error en: {palabra}", state="error")
        
        # Pequeña pausa para UI
        import time
        time.sleep(0.1)
    
    # Guardar configuración final
    if palabras_procesadas > 0:
        config['last_update'] = datetime.now().isoformat()
        
        # Obtener información del diccionario del master_config
        master_config = load_master_config()
        if master_config and 'diccionario_seleccionado' in master_config:
            config['diccionario_seleccionado'] = master_config['diccionario_seleccionado']
        else:
            # Si no hay info del diccionario en master_config, inferirla de las palabras generadas
            palabras_generadas = list(config.get('generated_samples', {}).keys())
            if palabras_generadas:
                # Buscar qué diccionario coincide con estas palabras
                from diccionarios import DICCIONARIOS_PREDEFINIDOS
                for dic_key, dic_data in DICCIONARIOS_PREDEFINIDOS.items():
                    if set(palabras_generadas) == set(dic_data['palabras'][:len(palabras_generadas)]):
                        config['diccionario_seleccionado'] = {
                            "tipo": "predefinido",
                            "nombre": dic_key,
                            "descripcion": dic_data['nombre'],
                            "palabras": palabras_generadas
                        }
                        break
        
        # Guardar la configuración actual de audio
        if 'configuracion_audio' not in config:
            config['configuracion_audio'] = {}
        config['configuracion_audio'].update(config_audio)
        
        save_audio_dataset_config(config)
        
        # Forzar sincronización con master_config
        sincronizar_con_master_config(config)
    
    return palabras_procesadas > 0

def sincronizar_con_master_config(config_local):
    """Sincroniza la configuración local con el master_config para mantener consistencia"""
    try:
        # El config_local ya contiene toda la información necesaria
        # Solo necesitamos asegurarnos de que esté en el formato correcto para master_config
        master_data = {
            "diccionario_seleccionado": config_local.get('diccionario_seleccionado', {}),
            "configuracion_audio": config_local.get('configuracion_audio', {}),
            "generated_samples": config_local.get('generated_samples', {}),
            "fecha_configuracion": config_local.get('last_update', pd.Timestamp.now().isoformat())
        }
        
        # Guardar como master config
        return save_dataset_config(master_data, "master_dataset_config.json")
    except Exception as e:
        st.error(f"Error en sincronización: {e}")
        return False

def sistema_verificacion():
    """Sistema para verificar y reproducir audios con interfaz mejorada"""
    st.markdown("### 🔊 Verificación y Reproducción")
    
    config = load_audio_dataset_config()
    if not config or not config.get('generated_samples'):
        st.warning("⚠️ No hay dataset generado para verificar.")
        st.info("💡 Ve a **🎤 Audio Base** para generar el dataset inicial")
        return
    
    raw_samples = config['generated_samples']
    
    # Estrategia de aplanado robusta (Maneja mezcla de estructuras plana y anidada)
    all_samples_flat = {}
    for key, value in raw_samples.items():
        if isinstance(value, list):
            # Estructura plana: key es la palabra
            all_samples_flat[key] = value
        elif isinstance(value, dict):
            # Estructura anidada: key es el idioma
            for word, variations in value.items():
                if isinstance(variations, list):
                    all_samples_flat[f"{key}/{word}"] = variations
    
    if not all_samples_flat:
        st.warning("⚠️ No se encontraron muestras válidas en el dataset.")
        return

    # Estadísticas del dataset
    st.markdown("#### 📊 Estadísticas del Dataset")
    
    total_palabras = len(all_samples_flat)
    total_muestras = sum(len(variaciones) for variaciones in all_samples_flat.values())
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("📝 Palabras", total_palabras)
    with col_stats2:
        st.metric("🎵 Total Muestras", total_muestras)
    with col_stats3:
        promedio = total_muestras / total_palabras if total_palabras > 0 else 0
        st.metric("📊 Promedio/Palabra", f"{promedio:.1f}")
    with col_stats4:
        # Calcular duración total aproximada
        duracion_total = 0
        for variaciones in all_samples_flat.values():
            for variacion in variaciones:
                if isinstance(variacion, dict):
                    duracion_total += variacion.get('duracion_ms', 0)
        st.metric("⏱️ Duración Total", f"{duracion_total/1000/60:.1f} min")
    
    st.markdown("---")
    
    # Interfaz de reproducción
    st.markdown("#### 🎵 Reproductor de Muestras")
    
    col_repro1, col_repro2 = st.columns(2)
    
    with col_repro1:
        # Ordenar claves para facilitar búsqueda
        sorted_keys = sorted(list(all_samples_flat.keys()))
        palabra_seleccionada = st.selectbox(
            "🔍 Seleccionar palabra (Formato: idioma/palabra o palabra):",
            sorted_keys,
            key="palabra_verificacion"
        )
    
    variacion_idx = None
    with col_repro2:
        if palabra_seleccionada:
            variaciones = all_samples_flat[palabra_seleccionada]
            variacion_names = [f"Variación {i+1}: {var.get('tipo', 'original').title()}" 
                             for i, var in enumerate(variaciones)]
            
            variacion_idx = st.selectbox(
                "🎛️ Seleccionar variación:",
                range(len(variacion_names)),
                format_func=lambda x: variacion_names[x],
                key="variacion_verificacion"
            )
    
    # Mostrar detalles de la muestra seleccionada
    if palabra_seleccionada and variacion_idx is not None:
        variacion_data = all_samples_flat[palabra_seleccionada][variacion_idx]
        
        st.markdown("##### 📊 Detalles de la Muestra")
        
        col_det1, col_det2, col_det3, col_det4 = st.columns(4)
        
        with col_det1:
            st.metric("🎵 Palabra", palabra_seleccionada)
        with col_det2:
            st.metric("🔄 Tipo", variacion_data.get('tipo', 'N/A').title())
        with col_det3:
            duracion = variacion_data.get('duracion_ms', 0)
            st.metric("⏱️ Duración", f"{duracion/1000:.2f}s" if duracion > 0 else "N/A")
        with col_det4:
            st.metric("🔊 Método", variacion_data.get('metodo_sintesis', 'N/A').upper())
        
        # Parámetros de la variación
        st.markdown("##### 🎛️ Parámetros de la Variación")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            pitch_factor = variacion_data.get('pitch_factor', 1.0)
            st.metric("🎵 Pitch Factor", f"{pitch_factor:.2f}x")
        with col_param2:
            speed_factor = variacion_data.get('speed_factor', 1.0)
            st.metric("⚡ Speed Factor", f"{speed_factor:.2f}x")
        with col_param3:
            volume_factor = variacion_data.get('volume_factor', 1.0)
            st.metric("🔊 Volume Factor", f"{volume_factor:.2f}x")
        
        # Reproductor de audio
        st.markdown("##### 🔊 Reproductor")
        
        if 'file_path' in variacion_data:
            try:
                # Construir ruta absoluta
                file_path = Path(__file__).parent.parent / variacion_data['file_path']
                
                if file_path.exists():
                    st.audio(str(file_path), format='audio/wav')
                    
                    # Botones adicionales
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        with open(file_path, 'rb') as f:
                            audio_bytes = f.read()
                            
                        st.download_button(
                            label="📥 Descargar WAV",
                            data=audio_bytes,
                            file_name=f"{palabra_seleccionada.replace('/', '_')}_{variacion_data.get('tipo', 'original')}.wav",
                            mime="audio/wav",
                            key=f"download_{palabra_seleccionada}_{variacion_idx}"
                        )
                    
                    with col_btn2:
                        if st.button("🗑️ Eliminar Muestra", key=f"delete_{palabra_seleccionada}_{variacion_idx}"):
                            # Eliminar archivo físico
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                st.warning(f"No se pudo eliminar archivo físico: {e}")
                                
                            # Eliminar la muestra del config
                            # Necesitamos saber si es anidado o plano para borrar del config original
                            parts = palabra_seleccionada.split('/')
                            if len(parts) > 1 and parts[0] in config['generated_samples'] and isinstance(config['generated_samples'][parts[0]], dict):
                                # Es anidado
                                lang, word = parts[0], parts[1]
                                del config['generated_samples'][lang][word][variacion_idx]
                                if not config['generated_samples'][lang][word]:
                                    del config['generated_samples'][lang][word]
                            else:
                                # Es plano
                                del config['generated_samples'][palabra_seleccionada][variacion_idx]
                                if not config['generated_samples'][palabra_seleccionada]:
                                    del config['generated_samples'][palabra_seleccionada]
                            
                            # Guardar cambios
                            save_audio_dataset_config(config)
                            st.success("✅ Muestra eliminada")
                            st.rerun()
                    
                    with col_btn3:
                        if st.button("🔄 Regenerar", key=f"regen_{palabra_seleccionada}_{variacion_idx}"):
                            # Regenerar esta muestra específica
                            config_audio = config.get('configuracion', {})
                            
                            # Determinar idioma y palabra
                            parts = palabra_seleccionada.split('/')
                            if len(parts) > 1:
                                lang = parts[0]
                                word = parts[1]
                            else:
                                lang = config_audio.get('idioma', 'es')
                                word = palabra_seleccionada
                            
                            # Obtener dataset name
                            dataset_name = lang
                                
                            nuevas_variaciones = generar_variaciones_completas(
                                word,
                                lang,
                                1,
                                config_audio.get('metodo_sintesis', 'gtts'),
                                dataset_name
                            )
                            
                            if nuevas_variaciones and len(nuevas_variaciones) > 1:
                                # Reemplazar la muestra
                                if len(parts) > 1 and parts[0] in config['generated_samples'] and isinstance(config['generated_samples'][parts[0]], dict):
                                    config['generated_samples'][parts[0]][parts[1]][variacion_idx] = nuevas_variaciones[1]
                                else:
                                    config['generated_samples'][palabra_seleccionada][variacion_idx] = nuevas_variaciones[1]
                                    
                                save_audio_dataset_config(config)
                                st.success("✅ Muestra regenerada")
                                st.rerun()
                else:
                    st.error(f"❌ Archivo no encontrado: {file_path}")
                    
            except Exception as e:
                st.error(f"Error reproduciendo audio: {e}")
        elif 'audio_base64' in variacion_data:
            # Soporte legacy para base64
            try:
                audio_bytes = base64.b64decode(variacion_data['audio_base64'])
                st.audio(audio_bytes, format='audio/wav')
                st.warning("⚠️ Esta muestra usa formato antiguo (Base64). Se recomienda regenerar el dataset.")
            except Exception as e:
                st.error(f"Error reproduciendo audio legacy: {e}")
        else:
            st.error("❌ No hay datos de audio para esta muestra")

def main():
    """Función principal de la página"""
    
    # Aplicar CSS moderno
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Sidebar modernizada persistente
    display_modern_sidebar("audio_dataset")
    
    # Header moderno
    st.markdown('<h1 class="main-header">🎤 Audio Dataset Manager</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <p style="font-size: 1.2em;">Generación y gestión avanzada de datasets de audio con variaciones aleatorias</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs principales organizadas
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Configuración", 
        "🎤 Audio Base", 
        "🎛️ Verificación",
        "📊 Estado del Dataset"
    ])
    
    with tab1:
        configuracion_completa_audio()
    
    with tab2:
        generar_audio_base()
    
    with tab3:
        sistema_verificacion()
    
    with tab4:
        # Estado y estadísticas del dataset
        st.markdown("### 📊 Estado General del Dataset")
        
        config = load_audio_dataset_config()
        if config and config.get('generated_samples'):
            raw_samples = config['generated_samples']
            
            # Aplanado robusto
            all_samples_flat = {}
            for key, value in raw_samples.items():
                if isinstance(value, list):
                    all_samples_flat[key] = value
                elif isinstance(value, dict):
                    for word, variations in value.items():
                        if isinstance(variations, list):
                            all_samples_flat[f"{key}/{word}"] = variations
            
            # Métricas generales
            total_palabras = len(all_samples_flat)
            total_muestras = sum(len(variaciones) for variaciones in all_samples_flat.values())
            
            col_1, col_2, col_3, col_4 = st.columns(4)
            
            with col_1:
                st.metric("📝 Total Palabras", total_palabras)
            with col_2:
                st.metric("🎵 Total Muestras", total_muestras)
            with col_3:
                promedio = total_muestras / total_palabras if total_palabras > 0 else 0
                st.metric("📊 Promedio por Palabra", f"{promedio:.1f}")
            with col_4:
                last_update = config.get('last_update', 'N/A')
                if last_update != 'N/A':
                    last_update = last_update[:19].replace('T', ' ')
                st.metric("🕐 Última Actualización", last_update)
            
            # Tabla resumen por palabra
            st.markdown("#### 📋 Resumen por Palabra")
            
            datos_resumen = []
            for key, variaciones in all_samples_flat.items():
                # Si es anidado, key es "lang/word", si no es "word"
                if '/' in key:
                    lang, palabra = key.split('/', 1)
                else:
                    palabra = key
                    lang = "N/A"
                
                tipos_count = {}
                duracion_total = 0
                
                for variacion in variaciones:
                    if isinstance(variacion, dict):
                        tipo = variacion.get('tipo', 'original')
                        tipos_count[tipo] = tipos_count.get(tipo, 0) + 1
                        duracion_total += variacion.get('duracion_ms', 0)
                
                row = {
                    'Palabra': palabra,
                    'Total Variaciones': len(variaciones),
                    'Original': tipos_count.get('original', 0),
                    'Pitch Alto': tipos_count.get('pitch_alto', 0),
                    'Pitch Bajo': tipos_count.get('pitch_bajo', 0),
                    'Rápido': tipos_count.get('rapido', 0),
                    'Lento': tipos_count.get('lento', 0),
                    'Fuerte': tipos_count.get('fuerte', 0),
                    'Suave': tipos_count.get('suave', 0),
                    'Duración Total (s)': f"{duracion_total/1000:.2f}"
                }
                if lang != "N/A":
                    row['Idioma'] = lang
                
                datos_resumen.append(row)
            
            df_resumen = pd.DataFrame(datos_resumen)
            
            # Reordenar columnas si hay idioma
            if 'Idioma' in df_resumen.columns:
                cols = ['Idioma', 'Palabra'] + [c for c in df_resumen.columns if c not in ['Idioma', 'Palabra']]
                df_resumen = df_resumen[cols]
                
            st.dataframe(df_resumen, width='stretch')
            
        else:
            st.info("📭 No hay dataset generado. Ve a **🎤 Audio Base** para comenzar.")

if __name__ == "__main__":
    main()