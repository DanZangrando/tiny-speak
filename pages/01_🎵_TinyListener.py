"""
🎵 TinyListener - Reconocimiento de Audio
Página dedicada para testing del modelo de audio a palabras
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path

# Importar módulos
from models import TinySpeak, TinyListener
from utils import (
    encontrar_device, load_wav2vec_model, load_waveform, plot_waveform_native, 
    plot_logits_native, get_default_words, synthesize_word, save_waveform_to_audio_file,
    WAV2VEC_SR, WAV2VEC_DIM
)

# Configurar página
st.set_page_config(
    page_title="TinyListener - Audio Recognition",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_audio_models():
    """Cargar solo los modelos necesarios para audio"""
    device = encontrar_device()
    wav2vec_model = load_wav2vec_model(device=device)
    words = get_default_words()
    
    tiny_speak = TinySpeak(words=words, hidden_dim=64, num_layers=2, wav2vec_dim=WAV2VEC_DIM)
    tiny_listener = TinyListener(tiny_speak=tiny_speak, wav2vec_model=wav2vec_model)
    
    tiny_speak = tiny_speak.to(device)
    tiny_listener = tiny_listener.to(device)
    
    return {
        'device': device,
        'wav2vec_model': wav2vec_model,
        'tiny_speak': tiny_speak,
        'tiny_listener': tiny_listener,
        'words': words
    }

def main():
    st.title("🎵 TinyListener - Reconocimiento de Audio")
    
    # Información del modelo
    with st.expander("🔍 Arquitectura del Modelo", expanded=True):
        st.markdown("""
        ### 🧠 **TinyListener Architecture**
        
        ```
        Audio Input (16kHz WAV) 
               ↓
        🤖 Wav2Vec2-Base-ES (Facebook)
        - Modelo preentrenado en español
        - Extrae embeddings de 768 dimensiones
        - Procesa audio a ~49Hz de características
               ↓
        📊 Feature Processing
        - Máscara de activaciones (capa 5)
        - Downsampling por factor 7
        - Padding de secuencias variables
               ↓
        🔄 LSTM Network
        - Input: 768 dim
        - Hidden: 64 dim  
        - Layers: 2
        - Batch-first processing
               ↓
        🎯 Linear Classifier
        - Input: 64 dim (último estado LSTM)
        - Output: 200 clases (palabras españolas)
        ```
        """)
    
    # Cargar modelos
    if 'audio_models' not in st.session_state:
        with st.spinner("🤖 Cargando modelos de audio..."):
            st.session_state.audio_models = load_audio_models()
    
    models = st.session_state.audio_models
    
    # Métricas del modelo
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Vocabulario", f"{len(models['words'])} palabras")
    with col2:
        st.metric("🧠 Parámetros LSTM", f"{sum(p.numel() for p in models['tiny_speak'].parameters()):,}")
    with col3:
        st.metric("📊 Wav2Vec2 Dim", f"{WAV2VEC_DIM}")
    with col4:
        st.metric("🔄 Sample Rate", f"{WAV2VEC_SR} Hz")
    
    # Pestañas para diferentes tipos de testing
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Cargar Audio", "🎤 Grabar Audio", "🔊 Síntesis + Test", "📊 Análisis Interno"])
    
    with tab1:
        test_file_upload(models)
    
    with tab2:
        test_audio_recording(models)
        
    with tab3:
        test_synthesis_and_recognition(models)
        
    with tab4:
        test_internal_analysis(models)

def test_file_upload(models):
    """Testing con archivos subidos"""
    st.subheader("📁 Test con Archivo de Audio")
    
    audio_file = st.file_uploader(
        "Sube un archivo de audio:", 
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Formatos soportados: WAV, MP3, FLAC, M4A"
    )
    
    if audio_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.audio(audio_file)
            
        with col2:
            if st.button("🔍 Analizar Audio", type="primary"):
                analyze_audio_file(audio_file, models)

def test_audio_recording(models):
    """Testing con grabación en tiempo real"""
    st.subheader("🎤 Test con Grabación en Tiempo Real")
    
    recorded_audio = st.audio_input("Graba tu voz")
    
    if recorded_audio is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.audio(recorded_audio)
            
        with col2:
            if st.button("🔍 Analizar Grabación", type="primary"):
                analyze_audio_file(recorded_audio, models)

def test_synthesis_and_recognition(models):
    """Testing con síntesis y reconocimiento"""
    st.subheader("🔊 Test: Síntesis → Reconocimiento")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ⚙️ Configuración de Síntesis")
        
        # Selector de palabra del vocabulario
        word_options = ["Palabra personalizada"] + models['words'][:50]  # Primeras 50 para el selector
        selected_option = st.selectbox("Selecciona una palabra:", word_options)
        
        if selected_option == "Palabra personalizada":
            text_input = st.text_input("Escribe una palabra:", value="hola")
        else:
            text_input = selected_option
        
        col_rate, col_pitch = st.columns(2)
        with col_rate:
            rate = st.slider("Velocidad", 50, 200, 80)
        with col_pitch:
            pitch = st.slider("Tono", 0, 100, 50)
        
        amplitude = st.slider("Volumen", 50, 200, 120)
        
        if st.button("🎵 Sintetizar y Analizar", type="primary", key="synthesize_btn"):
            with st.spinner("Sintetizando y analizando..."):
                synthesize_and_analyze(text_input, rate, pitch, amplitude, models)
                st.rerun()
    
    with col2:
        st.markdown("#### 📊 Resultados del Test")
        if 'synthesis_results' in st.session_state:
            display_synthesis_results(st.session_state.synthesis_results)

def test_internal_analysis(models):
    """Análisis interno del modelo"""
    st.subheader("📊 Análisis Interno del Modelo")
    
    # Mostrar vocabulario
    with st.expander("📝 Vocabulario Completo"):
        st.write(f"**Total de palabras:** {len(models['words'])}")
        
        # Mostrar en columnas
        cols = st.columns(4)
        for i, word in enumerate(models['words']):
            with cols[i % 4]:
                st.write(f"{i+1}. {word}")
    
    # Análisis de arquitectura
    with st.expander("🏗️ Detalles de Arquitectura"):
        st.code(f"""
# TinySpeak Architecture
TinySpeak(
  (lstm): LSTM(768, 64, num_layers=2, batch_first=True)
  (classifier): Linear(in_features=64, out_features={len(models['words'])})
)

# Parámetros por capa:
LSTM: {sum(p.numel() for p in models['tiny_speak'].lstm.parameters()):,} parámetros
Classifier: {sum(p.numel() for p in models['tiny_speak'].classifier.parameters()):,} parámetros
Total: {sum(p.numel() for p in models['tiny_speak'].parameters()):,} parámetros

# Wav2Vec2 (preentrenado, congelado)
Modelo: facebook/wav2vec2-base-es-voxpopuli-v2
Parámetros: ~95M (no entrenables en TinySpeak)
        """)
    
    # Test de palabra específica
    st.markdown("#### 🎯 Test de Palabra Específica")
    selected_word = st.selectbox("Selecciona palabra para análisis:", models['words'][:20])
    
    if st.button("🔬 Analizar Palabra"):
        analyze_specific_word(selected_word, models)

def analyze_audio_file(audio_file, models):
    """Analiza un archivo de audio"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        waveform = load_waveform(tmp_path, target_sr=WAV2VEC_SR)
        os.unlink(tmp_path)
        
        if waveform is not None:
            # Mostrar waveform
            fig = plot_waveform_native(waveform, "Audio Cargado")
            st.plotly_chart(fig, use_container_width=True)
            
            # Hacer predicción
            device = models['device']
            waveform = waveform.unsqueeze(0).to(device)
            
            models['tiny_listener'].eval()
            with torch.no_grad():
                logits, hidden_states = models['tiny_listener']([waveform.squeeze(0)])
            
            # Resultados
            display_prediction_results(logits, models, hidden_states)
        
        else:
            st.error("❌ Error al cargar el archivo de audio")
    
    except Exception as e:
        st.error(f"❌ Error procesando audio: {str(e)}")

def synthesize_and_analyze(text, rate, pitch, amplitude, models):
    """Sintetiza audio y lo analiza"""
    try:
        from utils import synthesize_word
        
        waveform = synthesize_word(text, rate=rate, pitch=pitch, amplitude=amplitude)
        
        if waveform is not None:
            # Guardar en session state para mostrar en la otra columna
            results = {
                'text': text,
                'waveform': waveform,
                'rate': rate,
                'pitch': pitch,
                'amplitude': amplitude
            }
            
            # Análisis con TinyListener
            device = models['device']
            waveform_device = waveform.to(device)
            
            models['tiny_listener'].eval()
            with torch.no_grad():
                logits, hidden_states = models['tiny_listener']([waveform_device])
            
            results['logits'] = logits
            results['prediction'] = models['words'][logits.argmax(dim=1).item()]
            results['confidence'] = torch.softmax(logits, dim=1).max().item()
            
            st.session_state.synthesis_results = results
            
        else:
            st.error("❌ Error generando audio")
    
    except Exception as e:
        st.error(f"❌ Error en síntesis: {str(e)}")

def display_synthesis_results(results):
    """Muestra resultados de síntesis"""
    # Audio sintetizado
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            if save_waveform_to_audio_file(results['waveform'], tmp_file.name, WAV2VEC_SR):
                with open(tmp_file.name, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/wav')
            else:
                st.warning("⚠️ No se pudo guardar el archivo de audio")
            
            os.unlink(tmp_file.name)
    
    except Exception as e:
        st.warning(f"⚠️ No se puede reproducir el audio: {str(e)}")
    
    # Waveform
    fig = plot_waveform_native(results['waveform'], f"Síntesis: '{results['text']}'")
    st.plotly_chart(fig, use_container_width=True)
    
    # Predicción
    if results['prediction'].lower() == results['text'].lower():
        st.success(f"✅ Reconocido correctamente: **{results['prediction']}** ({results['confidence']:.2%})")
    else:
        st.warning(f"⚠️ Reconocido como: **{results['prediction']}** ({results['confidence']:.2%})")

def display_prediction_results(logits, models, hidden_states=None):
    """Muestra resultados de predicción"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Predicción principal
        predicted_idx = logits.argmax(dim=1).item()
        predicted_word = models['words'][predicted_idx]
        confidence = torch.softmax(logits, dim=1).max().item()
        
        st.metric("🎯 Predicción", predicted_word, help=f"Confianza: {confidence:.2%}")
        
        # Top 5
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        st.markdown("**🏆 Top 5:**")
        for i, idx in enumerate(top_indices):
            word = models['words'][idx]
            prob = probabilities[idx]
            st.write(f"{i+1}. **{word}** ({prob:.2%})")
    
    with col2:
        # Gráfico de logits
        fig = plot_logits_native(logits, models['words'], "Distribución de Predicciones")
        st.plotly_chart(fig, use_container_width=True)

def analyze_specific_word(word, models):
    """Análiza una palabra específica del vocabulario"""
    try:
        # Sintetizar la palabra
        waveform = synthesize_word(word)
        
        if waveform is not None:
            st.success(f"✅ Palabra sintetizada: **{word}**")
            
            # Audio
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    if save_waveform_to_audio_file(waveform, tmp_file.name, WAV2VEC_SR):
                        with open(tmp_file.name, 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/wav')
                    else:
                        st.warning("⚠️ No se pudo guardar el archivo de audio")
                    
                    os.unlink(tmp_file.name)
            except Exception as e:
                st.warning(f"⚠️ No se puede reproducir el audio: {str(e)}")
            
            # Análisis
            device = models['device']
            waveform_device = waveform.to(device)
            
            models['tiny_listener'].eval()
            with torch.no_grad():
                logits, hidden_states = models['tiny_listener']([waveform_device])
            
            display_prediction_results(logits, models, hidden_states)
            
        else:
            st.error("❌ Error sintetizando la palabra")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()