"""
🎵 Audio Dataset Manager - Gestión de Datasets de Audio
Página para generar, modificar y analizar datasets de audio
"""

import streamlit as st
import torch
import numpy as np
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import librosa
import soundfile as sf

# Importar módulos
from utils import (
    encontrar_device, synthesize_word, save_waveform_to_audio_file, 
    WAV2VEC_SR, get_default_words
)

# Configurar página
st.set_page_config(
    page_title="Audio Dataset Manager",
    page_icon="🎵",
    layout="wide"
)

def main():
    """Función principal de la página"""
    st.title("🎵 Audio Dataset Manager")
    st.markdown("**Gestiona, genera y analiza datasets de audio para entrenar TinySpeak**")
    
    # Inicializar session state
    if 'dataset_config' not in st.session_state:
        st.session_state.dataset_config = load_dataset_config()
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Configuración", 
        "🎵 Generación de Audio", 
        "📊 Análisis del Dataset",
        "💾 Exportar/Importar"
    ])
    
    with tab1:
        dataset_configuration()
    
    with tab2:
        audio_generation()
    
    with tab3:
        dataset_analysis()
    
    with tab4:
        export_import_dataset()

def load_dataset_config():
    """Carga la configuración del dataset"""
    config_path = Path("dataset_config.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "name": "TinySpeak_Audio_Dataset",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "words": get_default_words()[:20],  # Empezar con 20 palabras
            "audio_params": {
                "sample_rate": WAV2VEC_SR,
                "rate_range": [60, 120],
                "pitch_range": [30, 70],
                "amplitude_range": [80, 150]
            },
            "variations_per_word": 5,
            "total_samples": 0,
            "generated_samples": {}
        }

def save_dataset_config(config):
    """Guarda la configuración del dataset de manera segura"""
    config_path = Path("dataset_config.json")
    config_temp_path = Path("dataset_config_temp.json")
    
    try:
        config["modified"] = datetime.now().isoformat()
        
        # Guardar en archivo temporal primero
        with open(config_temp_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Si la escritura fue exitosa, mover el archivo temporal al final
        import shutil
        shutil.move(str(config_temp_path), str(config_path))
        
        st.session_state.dataset_config = config
        
    except Exception as e:
        # Limpiar archivo temporal si hay error
        if config_temp_path.exists():
            config_temp_path.unlink()
        st.error(f"Error guardando configuración: {e}")
        raise e

def dataset_configuration():
    """Configuración del dataset"""
    st.header("📁 Configuración del Dataset")
    
    config = st.session_state.dataset_config.copy()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 Parámetros Generales")
        
        config["name"] = st.text_input(
            "Nombre del Dataset", 
            value=config["name"]
        )
        
        config["version"] = st.text_input(
            "Versión", 
            value=config["version"]
        )
        
        config["variations_per_word"] = st.slider(
            "Variaciones por palabra",
            min_value=1,
            max_value=20,
            value=config["variations_per_word"],
            help="Número de variaciones de audio a generar por cada palabra"
        )
        
        # Gestión de palabras
        st.subheader("📝 Gestión de Vocabulario")
        
        # Palabras actuales
        current_words = config["words"]
        
        # Selector de palabras predefinidas
        all_default_words = get_default_words()
        available_words = [w for w in all_default_words if w not in current_words]
        
        if available_words:
            selected_words = st.multiselect(
                "Agregar palabras del vocabulario predefinido",
                options=available_words,
                help="Selecciona palabras del vocabulario predefinido para agregar"
            )
            
            if st.button("➕ Agregar palabras seleccionadas"):
                config["words"].extend(selected_words)
                st.rerun()
        
        # Agregar palabra personalizada
        col_word, col_add = st.columns([3, 1])
        with col_word:
            new_word = st.text_input("Agregar palabra personalizada")
        with col_add:
            st.write("")  # Espaciado
            if st.button("➕ Agregar") and new_word and new_word not in current_words:
                config["words"].append(new_word.lower())
                st.rerun()
        
        # Mostrar palabras actuales
        if current_words:
            st.write(f"**Palabras en el dataset ({len(current_words)}):**")
            
            # Crear una grid de palabras editables
            cols_per_row = 4
            for i in range(0, len(current_words), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(current_words):
                        word = current_words[idx]
                        col.write(f"• {word}")
                        if col.button(f"🗑️", key=f"delete_{idx}"):
                            config["words"].remove(word)
                            st.rerun()
    
    with col2:
        st.subheader("🎛️ Parámetros de Audio")
        
        # Configuración de síntesis
        st.write("**Rangos de Variación:**")
        
        rate_min, rate_max = st.slider(
            "Rango de Velocidad",
            min_value=50,
            max_value=200,
            value=config["audio_params"]["rate_range"],
            help="Rango de velocidades para generar variaciones"
        )
        config["audio_params"]["rate_range"] = [rate_min, rate_max]
        
        pitch_min, pitch_max = st.slider(
            "Rango de Tono",
            min_value=0,
            max_value=100,
            value=config["audio_params"]["pitch_range"],
            help="Rango de tonos para generar variaciones"
        )
        config["audio_params"]["pitch_range"] = [pitch_min, pitch_max]
        
        amp_min, amp_max = st.slider(
            "Rango de Volumen",
            min_value=50,
            max_value=200,
            value=config["audio_params"]["amplitude_range"],
            help="Rango de amplitudes para generar variaciones"
        )
        config["audio_params"]["amplitude_range"] = [amp_min, amp_max]
        
        # Estadísticas
        st.subheader("📊 Estadísticas")
        total_planned = len(config["words"]) * config["variations_per_word"]
        st.metric("Total de muestras planificadas", total_planned)
        st.metric("Palabras en vocabulario", len(config["words"]))
        st.metric("Muestras generadas", config["total_samples"])
    
    # Botón para guardar
    if st.button("💾 Guardar Configuración", type="primary"):
        save_dataset_config(config)
        st.success("✅ Configuración guardada exitosamente!")

def audio_generation():
    """Generación de audio"""
    st.header("🎵 Generación de Audio")
    
    config = st.session_state.dataset_config
    
    if not config["words"]:
        st.warning("⚠️ No hay palabras configuradas. Ve a la pestaña de Configuración para agregar palabras.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Controles de Generación")
        
        # Selector de palabras
        selected_words = st.multiselect(
            "Palabras a generar",
            options=config["words"],
            default=config["words"][:5] if len(config["words"]) > 5 else config["words"],
            help="Selecciona las palabras para las que quieres generar audio"
        )
        
        # Número de variaciones
        num_variations = st.slider(
            "Variaciones a generar",
            min_value=1,
            max_value=config["variations_per_word"],
            value=min(3, config["variations_per_word"])
        )
        
        # Vista previa de parámetros
        st.write("**Vista previa de parámetros:**")
        st.json({
            "Velocidad": f"{config['audio_params']['rate_range'][0]} - {config['audio_params']['rate_range'][1]}",
            "Tono": f"{config['audio_params']['pitch_range'][0]} - {config['audio_params']['pitch_range'][1]}",
            "Volumen": f"{config['audio_params']['amplitude_range'][0]} - {config['audio_params']['amplitude_range'][1]}"
        })
        
        if st.button("🎵 Generar Audio", type="primary"):
            generate_audio_samples(selected_words, num_variations, config)
    
    with col2:
        st.subheader("🎧 Muestras Generadas")
        
        # Mostrar muestras generadas recientes
        if 'generated_samples' in config and config['generated_samples']:
            st.write("**Últimas muestras generadas:**")
            
            for word, samples in list(config['generated_samples'].items())[-5:]:
                with st.expander(f"🔊 {word} ({len(samples)} variaciones)"):
                    for i, sample in enumerate(samples[-3:]):  # Mostrar últimas 3
                        col_info, col_audio = st.columns([1, 1])
                        
                        with col_info:
                            st.write(f"**Variación {i+1}**")
                            st.write(f"Velocidad: {sample['params']['rate']}")
                            st.write(f"Tono: {sample['params']['pitch']}")
                            st.write(f"Volumen: {sample['params']['amplitude']}")
                        
                        with col_audio:
                            if 'waveform' in sample:
                                # Crear archivo temporal para reproducción
                                try:
                                    # Convertir de lista a numpy/tensor si es necesario
                                    waveform = sample['waveform']
                                    if isinstance(waveform, list):
                                        import torch
                                        waveform = torch.tensor(waveform)
                                    
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                        if save_waveform_to_audio_file(waveform, tmp_file.name, WAV2VEC_SR):
                                            with open(tmp_file.name, 'rb') as audio_file:
                                                st.audio(audio_file.read(), format='audio/wav')
                                        os.unlink(tmp_file.name)
                                except Exception as e:
                                    st.error(f"Error reproduciendo audio: {e}")
        else:
            st.info("No hay muestras generadas aún. Usa los controles de la izquierda para generar audio.")

def generate_audio_samples(words, num_variations, config):
    """Genera muestras de audio"""
    
    if 'generated_samples' not in config:
        config['generated_samples'] = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_samples = len(words) * num_variations
    current_sample = 0
    
    for word in words:
        if word not in config['generated_samples']:
            config['generated_samples'][word] = []
        
        status_text.text(f"Generando variaciones para: {word}")
        
        for i in range(num_variations):
            # Generar parámetros aleatorios dentro de los rangos
            rate = np.random.randint(*config["audio_params"]["rate_range"])
            pitch = np.random.randint(*config["audio_params"]["pitch_range"])
            amplitude = np.random.randint(*config["audio_params"]["amplitude_range"])
            
            # Sintetizar audio
            try:
                waveform = synthesize_word(word, rate=rate, pitch=pitch, amplitude=amplitude)
                
                if waveform is not None:
                    sample = {
                        'id': f"{word}_{len(config['generated_samples'][word])}",
                        'word': word,
                        'params': {
                            'rate': rate,
                            'pitch': pitch,
                            'amplitude': amplitude
                        },
                        'waveform': waveform.cpu().numpy().tolist() if hasattr(waveform, 'cpu') else waveform.tolist(),
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    config['generated_samples'][word].append(sample)
                    config['total_samples'] += 1
            
            except Exception as e:
                st.error(f"Error generando audio para '{word}': {e}")
            
            current_sample += 1
            progress_bar.progress(current_sample / total_samples)
    
    progress_bar.empty()
    status_text.empty()
    
    # Guardar configuración actualizada
    save_dataset_config(config)
    st.success(f"✅ Generadas {current_sample} muestras de audio exitosamente!")

def dataset_analysis():
    """Análisis del dataset"""
    st.header("📊 Análisis del Dataset")
    
    config = st.session_state.dataset_config
    
    if not config.get('generated_samples'):
        st.info("No hay muestras generadas para analizar. Ve a la pestaña de Generación para crear muestras.")
        return
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    
    total_words = len(config['generated_samples'])
    total_samples = sum(len(samples) for samples in config['generated_samples'].values())
    avg_per_word = total_samples / total_words if total_words > 0 else 0
    
    col1.metric("Total Palabras", total_words)
    col2.metric("Total Muestras", total_samples)
    col3.metric("Promedio por Palabra", f"{avg_per_word:.1f}")
    col4.metric("Tamaño Dataset", f"{total_samples * 0.5:.1f} MB")  # Estimación
    
    # Gráficos de distribución
    st.subheader("📈 Distribución de Muestras")
    
    # Preparar datos para visualización
    words = []
    counts = []
    rates = []
    pitches = []
    amplitudes = []
    
    for word, samples in config['generated_samples'].items():
        words.append(word)
        counts.append(len(samples))
        
        for sample in samples:
            rates.append(sample['params']['rate'])
            pitches.append(sample['params']['pitch'])
            amplitudes.append(sample['params']['amplitude'])
    
    # Gráfico de barras - Muestras por palabra
    fig_counts = px.bar(
        x=words, 
        y=counts,
        title="Número de Muestras por Palabra",
        labels={'x': 'Palabra', 'y': 'Número de Muestras'},
        color=counts,
        color_continuous_scale='Viridis'
    )
    fig_counts.update_layout(height=400)
    st.plotly_chart(fig_counts, use_container_width=True)
    
    # Histogramas de parámetros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_rate = px.histogram(
            x=rates,
            title="Distribución de Velocidades",
            labels={'x': 'Velocidad', 'y': 'Frecuencia'},
            nbins=20
        )
        fig_rate.update_layout(height=300)
        st.plotly_chart(fig_rate, use_container_width=True)
    
    with col2:
        fig_pitch = px.histogram(
            x=pitches,
            title="Distribución de Tonos",
            labels={'x': 'Tono', 'y': 'Frecuencia'},
            nbins=20
        )
        fig_pitch.update_layout(height=300)
        st.plotly_chart(fig_pitch, use_container_width=True)
    
    with col3:
        fig_amp = px.histogram(
            x=amplitudes,
            title="Distribución de Volúmenes",
            labels={'x': 'Volumen', 'y': 'Frecuencia'},
            nbins=20
        )
        fig_amp.update_layout(height=300)
        st.plotly_chart(fig_amp, use_container_width=True)
    
    # Análisis de duración de audio
    st.subheader("⏱️ Análisis de Duración")
    
    durations = []
    for samples in config['generated_samples'].values():
        for sample in samples:
            if 'waveform' in sample:
                duration = len(sample['waveform']) / WAV2VEC_SR
                durations.append(duration)
    
    if durations:
        fig_duration = px.histogram(
            x=durations,
            title="Distribución de Duraciones de Audio",
            labels={'x': 'Duración (segundos)', 'y': 'Frecuencia'},
            nbins=30
        )
        fig_duration.update_layout(height=400)
        st.plotly_chart(fig_duration, use_container_width=True)
        
        # Estadísticas de duración
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duración Promedio", f"{np.mean(durations):.2f}s")
        col2.metric("Duración Mínima", f"{np.min(durations):.2f}s")
        col3.metric("Duración Máxima", f"{np.max(durations):.2f}s")
        col4.metric("Duración Total", f"{np.sum(durations):.1f}s")

def export_import_dataset():
    """Exportar e importar dataset"""
    st.header("💾 Exportar/Importar Dataset")
    
    config = st.session_state.dataset_config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Exportar Dataset")
        
        if config.get('generated_samples'):
            # Información del export
            total_samples = sum(len(samples) for samples in config['generated_samples'].values())
            st.info(f"Dataset listo para exportar: {len(config['generated_samples'])} palabras, {total_samples} muestras")
            
            # Opciones de export
            export_format = st.selectbox(
                "Formato de exportación",
                ["JSON + WAV", "JSON solamente", "Configuración solamente"]
            )
            
            if st.button("📤 Exportar Dataset", type="primary"):
                export_dataset(config, export_format)
        else:
            st.warning("No hay muestras generadas para exportar.")
    
    with col2:
        st.subheader("📥 Importar Dataset")
        
        uploaded_file = st.file_uploader(
            "Seleccionar archivo de configuración",
            type=['json'],
            help="Sube un archivo JSON con la configuración del dataset"
        )
        
        if uploaded_file is not None:
            try:
                imported_config = json.load(uploaded_file)
                
                # Validar estructura básica
                if 'name' in imported_config and 'words' in imported_config:
                    st.success("✅ Archivo válido detectado")
                    
                    # Mostrar preview
                    st.write("**Preview del dataset:**")
                    st.json({
                        "Nombre": imported_config.get('name', 'Sin nombre'),
                        "Palabras": len(imported_config.get('words', [])),
                        "Muestras": imported_config.get('total_samples', 0)
                    })
                    
                    if st.button("📥 Importar Configuración"):
                        st.session_state.dataset_config = imported_config
                        save_dataset_config(imported_config)
                        st.success("✅ Dataset importado exitosamente!")
                        st.rerun()
                else:
                    st.error("❌ Archivo no válido. Falta estructura requerida.")
            
            except Exception as e:
                st.error(f"❌ Error leyendo archivo: {e}")

def export_dataset(config, export_format):
    """Exporta el dataset en el formato especificado"""
    
    if export_format == "Configuración solamente":
        # Solo exportar configuración
        config_export = config.copy()
        if 'generated_samples' in config_export:
            # Remover waveforms para hacer el archivo más pequeño
            for word in config_export['generated_samples']:
                for sample in config_export['generated_samples'][word]:
                    if 'waveform' in sample:
                        del sample['waveform']
        
        config_json = json.dumps(config_export, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Descargar Configuración",
            data=config_json,
            file_name=f"{config['name']}_config.json",
            mime="application/json"
        )
    
    elif export_format == "JSON solamente":
        # Exportar configuración completa con metadatos pero sin waveforms
        config_export = config.copy()
        config_json = json.dumps(config_export, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="💾 Descargar Dataset (JSON)",
            data=config_json,
            file_name=f"{config['name']}_dataset.json",
            mime="application/json"
        )
    
    else:  # JSON + WAV
        st.info("Para exportar con archivos WAV, usa la funcionalidad de descarga individual de cada muestra.")

if __name__ == "__main__":
    main()