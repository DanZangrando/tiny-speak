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

def change_speed(audio_segment, speed=1.0):
    """Cambia la velocidad del audio usando pydub"""
    # Cambiar la velocidad cambiando la frecuencia de muestreo
    new_sample_rate = int(audio_segment.frame_rate * speed)
    return audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": new_sample_rate
    }).set_frame_rate(audio_segment.frame_rate)

# Importar módulos
from utils import encontrar_device, WAV2VEC_SR, get_default_words

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
        
        # Crear configuración limpia
        config_limpia = {
            'generated_samples': {},
            'last_update': datetime.now().isoformat(),
            'dataset_cleaned': True
        }
        
        # Guardar configuración limpia
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_limpia, f, indent=4, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"Error limpiando dataset anterior: {e}")
        return False

def generar_audio_gtts(texto, idioma='es', velocidad=1.0):
    """
    Genera audio usando Google Text-to-Speech (gTTS)
    
    Args:
        texto: Texto a convertir en audio
        idioma: Código de idioma (ej: 'es', 'en')
        velocidad: Factor de velocidad (no aplicable directamente en gTTS)
    
    Returns:
        bytes: Audio en formato WAV
    """
    try:
        # Crear objeto TTS
        tts = gTTS(text=texto, lang=idioma, slow=False)
        
        # Guardar en buffer temporal
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            tts.save(temp_file.name)
            
            # Cargar y convertir a WAV
            audio = AudioSegment.from_mp3(temp_file.name)
            
            # Aplicar velocidad si es diferente de 1.0
            if velocidad != 1.0:
                audio = change_speed(audio, velocidad)
            
            # Normalizar
            audio = normalize(audio)
            
            # Convertir a WAV bytes
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()
            
            # Limpiar archivo temporal
            os.unlink(temp_file.name)
            
            return wav_bytes
            
    except Exception as e:
        st.error(f"Error generando audio con gTTS: {e}")
        return None

def save_audio_file(audio_bytes, dataset_name, word, filename):
    """Guarda el archivo de audio en el sistema de archivos"""
    try:
        # Definir ruta base: data/audios/{dataset_name}/{word}/
        base_dir = Path(__file__).parent.parent / "data" / "audios" / dataset_name / word
        base_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = base_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
            
        # Retornar ruta relativa para portabilidad
        return str(file_path.relative_to(Path(__file__).parent.parent))
    except Exception as e:
        st.error(f"Error guardando archivo de audio {filename}: {e}")
        return None

def generar_audio_espeak(texto, idioma='es', rate=80, pitch=70, amplitude=120):
    """
    Genera audio usando espeak (requiere instalación del sistema)
    
    Args:
        texto: Texto a convertir
        idioma: Código de idioma
        rate: Velocidad de habla (palabras por minuto)
        pitch: Tono (0-99)
        amplitude: Amplitud (0-200)
    
    Returns:
        bytes: Audio en formato WAV o None si falla
    """
    try:
        import subprocess
        
        # Comando espeak
        cmd = [
            'espeak',
            '-v', f'{idioma}',
            '-s', str(rate),
            '-p', str(pitch),
            '-a', str(amplitude),
            '-w', '/dev/stdout',  # Output a stdout
            texto
        ]
        
        # Ejecutar comando
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        if result.returncode == 0:
            return result.stdout
        else:
            st.error(f"Error en espeak: {result.stderr}")
            return None
            
    except FileNotFoundError:
        st.warning("⚠️ espeak no encontrado. Instálalo con: sudo apt-get install espeak")
        return None
    except Exception as e:
        st.error(f"Error con espeak: {e}")
        return None

def generar_audio_segun_metodo(texto, metodo='gtts', idioma='es', **kwargs):
    """
    Genera audio según el método especificado
    
    Args:
        texto: Texto a sintetizar
        metodo: 'gtts' o 'espeak'
        idioma: Código de idioma
        **kwargs: Parámetros adicionales específicos del método
    
    Returns:
        bytes: Audio generado o None si falla
    """
    if metodo == 'gtts':
        return generar_audio_gtts(texto, idioma, kwargs.get('velocidad', 1.0))
    elif metodo == 'espeak':
        return generar_audio_espeak(texto, idioma, 
                                  kwargs.get('rate', 80),
                                  kwargs.get('pitch', 70),
                                  kwargs.get('amplitude', 120))
    else:
        st.error(f"Método de síntesis no soportado: {metodo}")
        return None

def aplicar_variaciones_audio(audio_bytes, variacion_tipo, config_rangos=None):
    """
    Aplica variaciones al audio base
    
    Args:
        audio_bytes: Audio original en bytes
        variacion_tipo: Tipo de variación ('pitch_alto', 'pitch_bajo', etc.)
        config_rangos: Diccionario con rangos de variación
    
    Returns:
        tuple: (audio_modificado_bytes, parametros_aplicados)
    """
    if not audio_bytes:
        return None, {}
    
    # Rangos por defecto si no se especifican
    if config_rangos is None:
        config_rangos = {
            'pitch': [0.8, 1.3],
            'speed': [0.7, 1.4], 
            'volume': [0.8, 1.2]
        }
    
    try:
        # Cargar audio original
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        
        # Parámetros aplicados
        params = {
            'pitch_factor': 1.0,
            'speed_factor': 1.0,
            'volume_factor': 1.0,
            'tipo': variacion_tipo
        }
        
        # Aplicar variaciones según el tipo
        if variacion_tipo == 'pitch_alto':
            params['pitch_factor'] = random.uniform(1.1, config_rangos['pitch'][1])
            # Simular cambio de pitch modificando velocidad y compensando duración
            new_rate = int(audio.frame_rate * params['pitch_factor'])
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
            audio = audio.set_frame_rate(22050)  # Normalizar sample rate
            
        elif variacion_tipo == 'pitch_bajo':
            params['pitch_factor'] = random.uniform(config_rangos['pitch'][0], 0.9)
            new_rate = int(audio.frame_rate * params['pitch_factor'])
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
            audio = audio.set_frame_rate(22050)
            
        elif variacion_tipo == 'rapido':
            params['speed_factor'] = random.uniform(1.1, config_rangos['speed'][1])
            audio = change_speed(audio, params['speed_factor'])
            
        elif variacion_tipo == 'lento':
            params['speed_factor'] = random.uniform(config_rangos['speed'][0], 0.9)
            audio = change_speed(audio, params['speed_factor'])
            
        elif variacion_tipo == 'fuerte':
            params['volume_factor'] = random.uniform(1.1, config_rangos['volume'][1])
            audio = audio + (20 * np.log10(params['volume_factor']))  # dB adjustment
            
        elif variacion_tipo == 'suave':
            params['volume_factor'] = random.uniform(config_rangos['volume'][0], 0.9)
            audio = audio + (20 * np.log10(params['volume_factor']))
            
        # Normalizar audio final
        audio = normalize(audio)
        
        # Convertir a bytes
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        audio_modificado = wav_buffer.getvalue()
        
        return audio_modificado, params
        
    except Exception as e:
        st.error(f"Error aplicando variación {variacion_tipo}: {e}")
        return None, {}

def generar_variaciones_completas(texto, idioma, num_variaciones, metodo_sintesis='gtts', dataset_name='custom_dataset'):
    """
    Genera el audio original y sus variaciones para una palabra
    
    Args:
        texto: Palabra a sintetizar
        idioma: Idioma de síntesis
        num_variaciones: Número de variaciones a generar
        metodo_sintesis: Método de síntesis ('gtts' o 'espeak')
        dataset_name: Nombre del dataset para organizar archivos
    
    Returns:
        list: Lista de diccionarios con audio y metadatos
    """
    resultados = []
    
    # Generar audio base
    st.write(f"🎵 Generando audio base para: **{texto}**")
    audio_base = generar_audio_segun_metodo(texto, metodo_sintesis, idioma)
    
    if audio_base is None:
        st.error(f"❌ No se pudo generar audio base para: {texto}")
        return []
    
    # Guardar audio original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_original = f"{texto}_original_{timestamp}.wav"
    file_path_original = save_audio_file(audio_base, dataset_name, texto, filename_original)
    
    duracion_ms = len(AudioSegment.from_wav(io.BytesIO(audio_base)))
    
    resultado_original = {
        'file_path': file_path_original,
        'duracion_ms': duracion_ms,
        'timestamp': datetime.now().isoformat(),
        'tipo': 'original',
        'metodo_sintesis': metodo_sintesis,
        'pitch_factor': 1.0,
        'speed_factor': 1.0,
        'volume_factor': 1.0
    }
    resultados.append(resultado_original)
    
    st.write(f"✅ Audio original: {duracion_ms/1000:.2f}s")
    
    # Generar variaciones
    tipos_variacion = ['pitch_alto', 'pitch_bajo', 'rapido', 'lento', 'fuerte', 'suave']
    
    config = load_audio_dataset_config()
    rangos = config.get('configuracion', {}).get('rangos', {
        'pitch': [0.8, 1.3],
        'speed': [0.7, 1.4],
        'volume': [0.8, 1.2]
    })
    
    for i in range(num_variaciones):
        tipo_var = random.choice(tipos_variacion)
        st.write(f"🎛️ Generando variación {i+1}: **{tipo_var}**")
        
        audio_variado, params = aplicar_variaciones_audio(audio_base, tipo_var, rangos)
        
        if audio_variado:
            # Guardar variación
            filename_var = f"{texto}_{tipo_var}_{i}_{timestamp}.wav"
            file_path_var = save_audio_file(audio_variado, dataset_name, texto, filename_var)
            
            duracion_var_ms = len(AudioSegment.from_wav(io.BytesIO(audio_variado)))
            
            resultado_variacion = {
                'file_path': file_path_var,
                'duracion_ms': duracion_var_ms,
                'timestamp': datetime.now().isoformat(),
                'tipo': tipo_var,
                'metodo_sintesis': metodo_sintesis,
                'pitch_factor': params['pitch_factor'],
                'speed_factor': params['speed_factor'],
                'volume_factor': params['volume_factor']
            }
            resultados.append(resultado_variacion)
            st.write(f"✅ Variación {tipo_var}: {duracion_var_ms/1000:.2f}s")
        else:
            st.warning(f"⚠️ No se pudo generar variación {tipo_var}")
    
    return resultados

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
        
        # Número de variaciones (de master config si existe)
        default_num_variaciones = master_audio_config.get('num_variaciones', 
                                 local_audio_config.get('num_variaciones', 3))
        
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
        
        # Idioma (de master config si existe)
        idiomas_disponibles = {
            'es': 'Español',
            'en': 'English', 
            'fr': 'Français',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português'
        }
        
        idioma_actual = master_audio_config.get('idioma', 
                       local_audio_config.get('idioma', 'es'))
        
        idioma_audio = st.selectbox(
            "Idioma de síntesis:",
            list(idiomas_disponibles.keys()),
            index=list(idiomas_disponibles.keys()).index(idioma_actual),
            format_func=lambda x: f"{idiomas_disponibles[x]} ({x})"
        )
    
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
            if guardar_configuracion_audio(num_variaciones, metodo_sintesis, idioma_audio, rangos, master_config):
                st.success("✅ Configuración guardada correctamente")
                st.rerun()
    
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
                'idioma': idioma_audio,
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
                <p><strong>Idioma:</strong> {idiomas_disponibles.get(config_actual.get('idioma', 'es'), 'N/A')}</p>
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
    """Interfaz para generar audios base del dataset"""
    
    st.markdown("### 🎤 Generación de Audio Base")
    
    # Verificar configuración
    config = load_audio_dataset_config()
    
    if not config.get('configuracion'):
        st.warning("⚠️ No hay configuración de audio. Configura primero en la pestaña de Configuración.")
        return
    
    config_audio = config['configuracion']
    
    # Mostrar configuración actual
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(f"""
        <div class="config-card">
            <h4>🎤 Configuración Activa</h4>
            <p><strong>Método:</strong> {config_audio.get('metodo_sintesis', 'N/A').upper()}</p>
            <p><strong>Idioma:</strong> {config_audio.get('idioma', 'N/A')}</p>
            <p><strong>Variaciones:</strong> {config_audio.get('num_variaciones', 0)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        # Dataset actual
        samples = config.get('generated_samples', {})
        total_palabras = len(samples)
        total_muestras = sum(len(variaciones) for variaciones in samples.values())
        
        st.markdown(f"""
        <div class="status-card">
            <h4>📊 Estado del Dataset</h4>
            <p><strong>Palabras:</strong> {total_palabras}</p>
            <p><strong>Muestras:</strong> {total_muestras}</p>
            <p><strong>Última actualización:</strong> {config.get('last_update', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Selección de palabras
    st.markdown("#### 📝 Selección de Palabras")
    
    tab_custom, tab_predefinidas = st.tabs(["📝 Palabras Personalizadas", "📚 Palabras Predefinidas"])
    
    with tab_custom:
        col_input1, col_input2 = st.columns([3, 1])
        
        with col_input1:
            palabras_input = st.text_area(
                "Escribe las palabras (una por línea):",
                height=150,
                placeholder="hola\nadiós\ngracias\npor favor\nayuda"
            )
        
        with col_input2:
            st.markdown("**💡 Tips:**")
            st.markdown("- Una palabra por línea")
            st.markdown("- Evita caracteres especiales")
            st.markdown("- Palabras cortas funcionan mejor")
            st.markdown("- Máximo 50 palabras por lote")
        
        palabras_custom = []
        if palabras_input:
            palabras_custom = [p.strip() for p in palabras_input.split('\n') if p.strip()]
            if len(palabras_custom) > 50:
                st.warning("⚠️ Demasiadas palabras. Se procesarán solo las primeras 50.")
                palabras_custom = palabras_custom[:50]
            
            st.info(f"📊 {len(palabras_custom)} palabras listas para procesar")
    
    with tab_predefinidas:
        # Verificar diccionario configurado en master config
        master_config = load_master_config()
        
        if master_config and 'diccionario_seleccionado' in master_config:
            dic_info = master_config['diccionario_seleccionado']
            st.markdown(f"""
            <div class="config-card">
                <h4>✅ Vocabulario Configurado</h4>
                <p><strong>Diccionario:</strong> {dic_info.get('nombre', 'N/A')}</p>
                <p><strong>Descripción:</strong> {dic_info.get('descripcion', 'N/A')}</p>
                <p><strong>Palabras:</strong> {len(dic_info.get('palabras', []))}</p>
            </div>
            """, unsafe_allow_html=True)
            
            palabras_seleccionadas = dic_info.get('palabras', [])
            
            if st.checkbox("🔍 Usar vocabulario configurado", value=True):
                with st.expander("👁️ Vista previa del vocabulario", expanded=False):
                    # Mostrar palabras en columnas
                    num_cols = 4
                    cols = st.columns(num_cols)
                    for i, palabra in enumerate(palabras_seleccionadas):
                        with cols[i % num_cols]:
                            st.write(f"• {palabra}")
        else:
            # Mostrar diccionarios predefinidos disponibles
            st.warning("⚠️ No hay vocabulario configurado en la configuración maestra")
            st.markdown("**Selecciona un diccionario predefinido:**")
            
            diccionario_nombres = {
                key: f"{dic['nombre']} ({len(dic['palabras'])} palabras)"
                for key, dic in DICCIONARIOS_PREDEFINIDOS.items()
            }
            
            diccionario_elegido = st.selectbox(
                "Diccionario:",
                list(diccionario_nombres.keys()),
                format_func=lambda x: diccionario_nombres[x]
            )
            
            if diccionario_elegido:
                dic_data = DICCIONARIOS_PREDEFINIDOS[diccionario_elegido]
                palabras_seleccionadas = dic_data['palabras']
                
                # Actualizar el master_config con el diccionario seleccionado
                master_config = load_master_config()
                if not master_config:
                    master_config = {}
                
                diccionario_info = {
                    "tipo": "predefinido",
                    "nombre": diccionario_elegido,
                    "descripcion": dic_data['nombre'],
                    "palabras": dic_data['palabras']
                }
                
                # Solo actualizar si es diferente al actual
                if master_config.get('diccionario_seleccionado') != diccionario_info:
                    master_config['diccionario_seleccionado'] = diccionario_info
                    master_config['fecha_configuracion'] = pd.Timestamp.now().isoformat()
                    save_dataset_config(master_config, "master_dataset_config.json")
                
                st.info(f"📚 **{dic_data['nombre']}**\n\n{dic_data['descripcion']}")
                
                with st.expander("👁️ Vista previa del vocabulario", expanded=False):
                    # Mostrar palabras en columnas
                    num_cols = 4
                    cols = st.columns(num_cols)
                    for i, palabra in enumerate(palabras_seleccionadas):
                        with cols[i % num_cols]:
                            st.write(f"• {palabra}")
            else:
                palabras_seleccionadas = []
        
        if palabras_seleccionadas:
            st.success(f"✅ {len(palabras_seleccionadas)} palabras disponibles para generar")
    
    # Determinar palabras finales
    palabras_finales = []
    if tab_custom and palabras_custom:
        palabras_finales = palabras_custom
    elif palabras_seleccionadas:
        palabras_finales = palabras_seleccionadas
    
    # Generar dataset
    st.markdown("---")
    
    if palabras_finales:
        # Opciones de generación
        st.markdown("#### 🎛️ Opciones de Generación")
        
        # Checkbox para limpiar dataset anterior
        limpiar_anterior = st.checkbox(
            "🧹 Limpiar dataset anterior antes de generar", 
            value=False,
            help="Elimina todos los audios generados anteriormente para evitar acumulación de datos y problemas de compatibilidad"
        )
        
        if limpiar_anterior:
            st.warning("⚠️ **Atención**: Esta acción eliminará todos los audios generados anteriormente. Esta acción es recomendada para evitar problemas de compatibilidad.")
        
        st.markdown("---")
        
        col_gen1, col_gen2, col_gen3 = st.columns(3)
        
        with col_gen1:
            if st.button("🎵 Generar Dataset Completo", type="primary"):
                if generar_proceso_completo(palabras_finales, config_audio, limpiar_anterior=limpiar_anterior):
                    st.success("✅ Dataset generado exitosamente!")
                    st.balloons()
        
        with col_gen2:
            if st.button("🔄 Actualizar Palabras Existentes"):
                # Solo actualizar palabras que ya existen
                palabras_existentes = [p for p in palabras_finales if p in samples]
                if palabras_existentes:
                    if generar_proceso_completo(palabras_existentes, config_audio, limpiar_anterior=limpiar_anterior):
                        st.success(f"✅ {len(palabras_existentes)} palabras actualizadas!")
                else:
                    st.warning("⚠️ No hay palabras existentes para actualizar")
        
        with col_gen3:
            if st.button("➕ Añadir Nuevas Palabras"):
                # Solo añadir palabras nuevas
                palabras_nuevas = [p for p in palabras_finales if p not in samples]
                if palabras_nuevas:
                    if generar_proceso_completo(palabras_nuevas, config_audio, limpiar_anterior=False):
                        st.success(f"✅ {len(palabras_nuevas)} palabras nuevas añadidas!")
                else:
                    st.warning("⚠️ No hay palabras nuevas para añadir")
        
        # Preview de palabras
        st.markdown("#### 📋 Preview de Palabras a Procesar")
        df_preview = pd.DataFrame({
            'Palabra': palabras_finales,
            'Estado': ['✅ Existe' if p in samples else '🆕 Nueva' for p in palabras_finales],
            'Muestras Actuales': [len(samples.get(p, [])) for p in palabras_finales]
        })
        st.dataframe(df_preview, width='stretch')
        
        # Acciones adicionales
        st.markdown("---")
        st.markdown("#### 🗑️ Acciones de Limpieza")
        
        col_clean1, col_clean2 = st.columns(2)
        
        with col_clean1:
            if st.button("🧹 Limpiar Dataset Completo", type="secondary"):
                with st.spinner("Limpiando dataset..."):
                    if limpiar_dataset_anterior():
                        st.success("✅ Dataset limpiado completamente")
                        st.rerun()
                    else:
                        st.error("❌ Error al limpiar dataset")
        
        with col_clean2:
            # Mostrar información del dataset actual
            if samples:
                total_palabras = len(samples)
                total_muestras = sum(len(variaciones) for variaciones in samples.values())
                st.info(f"📊 Dataset actual: {total_palabras} palabras, {total_muestras} muestras totales")
            else:
                st.info("📊 No hay dataset actual")
        
    else:
        st.info("📝 Selecciona o escribe palabras para comenzar la generación")

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
    
    samples = config['generated_samples']
    
    # Estadísticas del dataset
    st.markdown("#### 📊 Estadísticas del Dataset")
    
    total_palabras = len(samples)
    total_muestras = sum(len(variaciones) for variaciones in samples.values())
    
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
        for variaciones in samples.values():
            for variacion in variaciones:
                duracion_total += variacion.get('duracion_ms', 0)
        st.metric("⏱️ Duración Total", f"{duracion_total/1000/60:.1f} min")
    
    st.markdown("---")
    
    # Interfaz de reproducción
    st.markdown("#### 🎵 Reproductor de Muestras")
    
    col_repro1, col_repro2 = st.columns(2)
    
    with col_repro1:
        palabra_seleccionada = st.selectbox(
            "🔍 Seleccionar palabra:",
            list(samples.keys()),
            key="palabra_verificacion"
        )
    
    with col_repro2:
        if palabra_seleccionada:
            variaciones = samples[palabra_seleccionada]
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
        variacion_data = samples[palabra_seleccionada][variacion_idx]
        
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
                            file_name=f"{palabra_seleccionada}_{variacion_data.get('tipo', 'original')}.wav",
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
                            del config['generated_samples'][palabra_seleccionada][variacion_idx]
                            
                            # Si no quedan muestras, eliminar la palabra
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
                            
                            # Obtener dataset name
                            master_config = load_master_config()
                            dataset_name = "custom_dataset"
                            if master_config and 'diccionario_seleccionado' in master_config:
                                dataset_name = master_config['diccionario_seleccionado'].get('nombre', 'custom_dataset')
                                
                            nuevas_variaciones = generar_variaciones_completas(
                                palabra_seleccionada,
                                config_audio.get('idioma', 'es'),
                                1,
                                config_audio.get('metodo_sintesis', 'gtts'),
                                dataset_name
                            )
                            
                            if nuevas_variaciones and len(nuevas_variaciones) > 1:
                                # Reemplazar la muestra (tomar la primera variación, no el original)
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
            samples = config['generated_samples']
            
            # Métricas generales
            total_palabras = len(samples)
            total_muestras = sum(len(variaciones) for variaciones in samples.values())
            
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
            for palabra, variaciones in samples.items():
                tipos_count = {}
                duracion_total = 0
                
                for variacion in variaciones:
                    tipo = variacion.get('tipo', 'original')
                    tipos_count[tipo] = tipos_count.get(tipo, 0) + 1
                    duracion_total += variacion.get('duracion_ms', 0)
                
                datos_resumen.append({
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
                })
            
            df_resumen = pd.DataFrame(datos_resumen)
            st.dataframe(df_resumen, width='stretch')
            
        else:
            st.info("📭 No hay dataset generado. Ve a **🎤 Audio Base** para comenzar.")

if __name__ == "__main__":
    main()