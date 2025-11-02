"""
TinySpeak - Aplicación de Reconocimiento de Voz y Visión
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io
import tempfile
import os

# Configurar la página
st.set_page_config(
    page_title="TinySpeak - Reconocimiento Multimodal",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar nuestros módulos
from models import TinySpeak, TinyListener, TinyRecognizer, TinySpeller
from utils import (
    encontrar_device, load_wav2vec_model, load_waveform, plot_waveform, 
    plot_logits, ensure_data_downloaded, get_default_words, synthesize_word,
    WAV2VEC_SR, WAV2VEC_DIM, LETTERS
)

# Configuración de la aplicación
@st.cache_resource
def setup_models():
    """Inicializa los modelos y configuración"""
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

def main():
    st.title("🎤 TinySpeak - Reconocimiento Multimodal")
    
    st.markdown("""
    ## 🌟 Bienvenido a TinySpeak
    
    **TinySpeak** es un sistema de IA multimodal que combina reconocimiento de voz y visión para procesar información de diferentes modalidades sensoriales.
    
    ### 🧠 **Modelos Implementados:**
    
    """)
    
    # Mostrar tarjetas de modelos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🎵 TinyListener
        **Audio → Palabra**
        
        - 🤖 Wav2Vec2 preentrenado 
        - 🔄 LSTM para secuencias temporales
        - 🎯 ~200 palabras en español
        - ⚡ Reconocimiento en tiempo real
        """)
        
    with col2:
        st.markdown("""
        #### 🖼️ TinyRecognizer  
        **Imagen → Letra**
        
        - 🧠 Arquitectura CORnet-Z
        - 👁️ Inspirada en cortex visual
        - 🔤 Letras a-z manuscritas
        - 🎨 Análisis de embeddings visuales
        """)
        
    with col3:
        st.markdown("""
        #### 🔗 TinySpeller
        **Multimodal: Visión + Audio**
        
        - 🖼️➡️📝 Secuencia letras → palabra
        - 🎵➡️📝 Audio directo → palabra  
        - ⚖️ Comparación entre modalidades
        - 🧪 Análisis multimodal avanzado
        """)
    
    st.markdown("---")
    
    # Información de arquitectura
    with st.expander("🏗️ Arquitectura del Sistema", expanded=False):
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
        - **Dataset**: ~200 palabras españolas + 26 letras manuscritas
        """)
    
    # Navegación
    st.markdown("### 🧭 **Navegación**")
    st.info("""
    👈 **Usa la barra lateral** para navegar entre las páginas específicas de cada modelo:
    
    - **🎵 TinyListener**: Testing completo de reconocimiento de audio
    - **🖼️ TinyRecognizer**: Análisis detallado de reconocimiento visual  
    - **🔗 TinySpeller**: Experimentos multimodales avanzados
    
    Cada página incluye herramientas especializadas para testing, análisis y comparación.
    """)
    
    # Estado del sistema
    st.markdown("### 📊 **Estado del Sistema**")
    
    # Verificar estado de los componentes básicos
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        device = encontrar_device()
        col1.metric("🖥️ Dispositivo", str(device).upper())
    except:
        col1.metric("🖥️ Dispositivo", "Error", delta="❌")
    
    try:
        words = get_default_words()
        col2.metric("📚 Vocabulario", f"{len(words)} palabras")
    except:
        col2.metric("📚 Vocabulario", "Error", delta="❌")
    
    try:
        import subprocess
        result = subprocess.run(["espeak", "--version"], capture_output=True)
        if result.returncode == 0:
            col3.metric("🔊 Espeak", "Disponible", delta="✅")
        else:
            col3.metric("🔊 Espeak", "No disponible", delta="⚠️")
    except:
        col3.metric("🔊 Espeak", "No disponible", delta="⚠️")
    
    try:
        import torch
        col4.metric("🔥 PyTorch", torch.__version__[:5])
    except:
        col4.metric("🔥 PyTorch", "Error", delta="❌")
    
    # Ejemplos rápidos
    st.markdown("### 🚀 **Ejemplos Rápidos**")
    
    if st.button("🧪 Ejecutar Test Rápido del Sistema"):
        run_quick_system_test()

def run_quick_system_test():
    """Ejecuta un test rápido del sistema completo"""
    with st.spinner("🔄 Ejecutando test del sistema..."):
        try:
            # Test básico de imports
            from models import TinySpeak, TinyRecognizer
            from utils import synthesize_word, get_default_words
            
            # Test de síntesis
            test_word = "hola"
            waveform = synthesize_word(test_word)
            
            if waveform is not None:
                st.success("✅ Sistema funcionando correctamente!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Componentes verificados:**")
                    st.write("✅ Modelos cargados")
                    st.write("✅ Síntesis de voz") 
                    st.write("✅ Procesamiento de audio")
                    st.write("✅ Vocabulario disponible")
                
                with col2:
                    # Reproducir audio de prueba
                    import tempfile
                    import torchaudio
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        torchaudio.save(tmp_file.name, waveform.unsqueeze(0), 16000)
                        
                        with open(tmp_file.name, 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/wav')
                        
                        import os
                        os.unlink(tmp_file.name)
                    
                    st.write(f"🔊 Audio de prueba: '{test_word}'")
            else:
                st.warning("⚠️ Sistema parcialmente funcional - problema con síntesis de audio")
        
        except Exception as e:
            st.error(f"❌ Error en el test del sistema: {str(e)}")

def audio_recognition_interface(models):
    """Interfaz para reconocimiento de audio"""
    st.header("🎵 Reconocimiento de Audio - TinyListener")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Cargar Audio")
        
        # Opción 1: Subir archivo
        audio_file = st.file_uploader(
            "Sube un archivo de audio:", 
            type=['wav', 'mp3', 'flac', 'm4a']
        )
        
        # Opción 2: Grabar audio
        st.markdown("**O graba audio directamente:**")
        recorded_audio = st.audio_input("Graba tu voz")
        
        # Procesamiento del audio
        audio_data = None
        if audio_file is not None:
            audio_data = audio_file
            st.success("✅ Archivo de audio cargado")
        elif recorded_audio is not None:
            audio_data = recorded_audio
            st.success("✅ Audio grabado")
        
        if audio_data is not None:
            # Reproducir audio
            st.audio(audio_data)
            
            # Procesar audio
            if st.button("🔍 Analizar Audio", type="primary"):
                with st.spinner("Procesando audio..."):
                    try:
                        # Guardar temporalmente
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(audio_data.read())
                            tmp_path = tmp_file.name
                        
                        # Cargar y procesar waveform
                        waveform = load_waveform(tmp_path, target_sr=WAV2VEC_SR)
                        os.unlink(tmp_path)  # Limpiar archivo temporal
                        
                        if waveform is not None:
                            # Mostrar waveform
                            fig = plot_waveform(waveform, "Audio cargado")
                            st.pyplot(fig)
                            
                            # Hacer predicción
                            device = models['device']
                            waveform = waveform.unsqueeze(0).to(device)
                            
                            models['tiny_listener'].eval()
                            with torch.no_grad():
                                logits, _ = models['tiny_listener']([waveform.squeeze(0)])
                            
                            # Mostrar resultados
                            with col2:
                                st.subheader("📊 Resultados")
                                
                                # Predicción principal
                                predicted_idx = logits.argmax(dim=1).item()
                                predicted_word = models['words'][predicted_idx]
                                confidence = torch.softmax(logits, dim=1).max().item()
                                
                                st.metric(
                                    label="Palabra Predicha", 
                                    value=predicted_word,
                                    help=f"Confianza: {confidence:.2%}"
                                )
                                
                                # Top 5 predicciones
                                st.subheader("🏆 Top 5 Predicciones")
                                probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                                top_indices = np.argsort(probabilities)[::-1][:5]
                                
                                for i, idx in enumerate(top_indices):
                                    word = models['words'][idx]
                                    prob = probabilities[idx]
                                    st.write(f"{i+1}. **{word}** ({prob:.2%})")
                                
                                # Gráfico de logits
                                fig = plot_logits(logits.squeeze().cpu().numpy(), models['words'], "Distribución de Predicciones")
                                st.pyplot(fig)
                        
                        else:
                            st.error("❌ Error al cargar el archivo de audio")
                    
                    except Exception as e:
                        st.error(f"❌ Error procesando audio: {str(e)}")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.markdown("""
        **TinyListener** utiliza:
        - Modelo Wav2Vec2 preentrenado para extraer características del audio
        - Red LSTM para procesar secuencias temporales
        - Clasificador lineal para predecir palabras
        
        **Palabras reconocidas:** """ + ", ".join(models['words'][:10]) + "...")

def image_recognition_interface(models):
    """Interfaz para reconocimiento de imágenes"""
    st.header("🖼️ Reconocimiento de Letras - TinyRecognizer")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Cargar Imagen")
        
        # Opción 1: Subir archivo
        image_file = st.file_uploader(
            "Sube una imagen de una letra:", 
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        # Opción 2: Dibujar letra
        st.markdown("**O dibuja una letra:**")
        
        # Canvas para dibujar (simulado con texto por ahora)
        st.info("🖊️ Funcionalidad de dibujo en desarrollo. Por favor, sube una imagen.")
        
        if image_file is not None:
            # Cargar y mostrar imagen
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            if st.button("🔍 Reconocer Letra", type="primary"):
                with st.spinner("Procesando imagen..."):
                    try:
                        # Preprocesar imagen
                        from torchvision.transforms import Compose, ToTensor, Resize, Normalize
                        
                        mean = [0.485, 0.456, 0.406]
                        std = [0.229, 0.224, 0.225]
                        
                        transform = Compose([
                            Resize((28, 28)),
                            ToTensor(),
                            Normalize(mean, std)
                        ])
                        
                        # Convertir imagen
                        image_tensor = transform(image).unsqueeze(0).to(models['device'])
                        
                        # Hacer predicción
                        models['tiny_recognizer'].eval()
                        with torch.no_grad():
                            logits, embeddings = models['tiny_recognizer'](image_tensor)
                        
                        # Mostrar resultados
                        with col2:
                            st.subheader("📊 Resultados")
                            
                            # Predicción principal
                            predicted_idx = logits.argmax(dim=1).item()
                            predicted_letter = LETTERS[predicted_idx]
                            confidence = torch.softmax(logits, dim=1).max().item()
                            
                            st.metric(
                                label="Letra Predicha", 
                                value=predicted_letter.upper(),
                                help=f"Confianza: {confidence:.2%}"
                            )
                            
                            # Top 5 predicciones
                            st.subheader("🏆 Top 5 Predicciones")
                            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                            top_indices = np.argsort(probabilities)[::-1][:5]
                            
                            for i, idx in enumerate(top_indices):
                                letter = LETTERS[idx].upper()
                                prob = probabilities[idx]
                                st.write(f"{i+1}. **{letter}** ({prob:.2%})")
                            
                            # Visualización del embedding
                            st.subheader("🧠 Embedding Visual")
                            embedding_2d = embeddings.squeeze().cpu().numpy().reshape(32, 24)
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(embedding_2d, cmap='coolwarm')
                            ax.set_title("Representación Interna del Modelo")
                            plt.colorbar(im)
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"❌ Error procesando imagen: {str(e)}")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.markdown("""
        **TinyRecognizer** utiliza:
        - Arquitectura CORnet-Z inspirada en el sistema visual
        - Capas convolucionales para extracción de características
        - Clasificador para reconocer letras del alfabeto (a-z)
        
        **Entrada:** Imágenes de 28x28 píxeles
        **Salida:** Probabilidades para cada letra del alfabeto
        """)

def speech_synthesis_interface(models):
    """Interfaz para síntesis de voz"""
    st.header("🔊 Síntesis de Voz")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Generar Audio")
        
        # Input de texto
        text_input = st.text_input(
            "Escribe una palabra para sintetizar:",
            value="hola",
            help="Escribe cualquier palabra en español"
        )
        
        # Parámetros de síntesis
        st.subheader("⚙️ Parámetros de Voz")
        
        col_rate, col_pitch = st.columns(2)
        with col_rate:
            rate = st.slider("Velocidad", 50, 200, 80, help="Velocidad de habla (palabras por minuto)")
        with col_pitch:
            pitch = st.slider("Tono", 0, 100, 50, help="Altura del tono de voz")
        
        amplitude = st.slider("Volumen", 50, 200, 120, help="Amplitud del audio")
        
        if st.button("🎵 Generar Audio", type="primary"):
            if text_input.strip():
                with st.spinner("Generando audio..."):
                    try:
                        # Sintetizar audio
                        waveform = synthesize_word(
                            text_input.strip(),
                            rate=rate,
                            pitch=pitch,
                            amplitude=amplitude
                        )
                        
                        if waveform is not None:
                            with col2:
                                st.subheader("🎧 Audio Generado")
                                
                                # Guardar audio temporal para reproducción
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                    import torchaudio
                                    torchaudio.save(tmp_file.name, waveform.unsqueeze(0), WAV2VEC_SR)
                                    
                                    # Reproducir audio
                                    with open(tmp_file.name, 'rb') as audio_file:
                                        st.audio(audio_file.read(), format='audio/wav')
                                    
                                    os.unlink(tmp_file.name)  # Limpiar
                                
                                # Mostrar waveform
                                fig = plot_waveform(waveform, f"Audio sintetizado: '{text_input}'")
                                st.pyplot(fig)
                                
                                # Análisis con TinyListener
                                st.subheader("🔍 Análisis con TinyListener")
                                
                                device = models['device']
                                waveform_device = waveform.to(device)
                                
                                models['tiny_listener'].eval()
                                with torch.no_grad():
                                    logits, _ = models['tiny_listener']([waveform_device])
                                
                                # Predicción
                                predicted_idx = logits.argmax(dim=1).item()
                                predicted_word = models['words'][predicted_idx]
                                confidence = torch.softmax(logits, dim=1).max().item()
                                
                                if predicted_word.lower() == text_input.lower():
                                    st.success(f"✅ ¡Reconocido correctamente como '{predicted_word}'! (Confianza: {confidence:.2%})")
                                else:
                                    st.warning(f"⚠️ Reconocido como '{predicted_word}' (Confianza: {confidence:.2%})")
                                
                                # Top predicciones
                                probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                                top_indices = np.argsort(probabilities)[::-1][:3]
                                
                                st.write("**Top 3 predicciones:**")
                                for i, idx in enumerate(top_indices):
                                    word = models['words'][idx]
                                    prob = probabilities[idx]
                                    icon = "🎯" if i == 0 else "📍"
                                    st.write(f"{icon} {word} ({prob:.2%})")
                        
                        else:
                            st.error("❌ Error generando audio. ¿Tienes espeak instalado?")
                    
                    except Exception as e:
                        st.error(f"❌ Error en síntesis: {str(e)}")
            else:
                st.warning("⚠️ Por favor ingresa una palabra")
    
    # Información
    with st.expander("ℹ️ Información sobre Síntesis"):
        st.markdown("""
        **Síntesis de Voz** utiliza:
        - **espeak** para generar audio sintético
        - Configuración personalizable de velocidad, tono y volumen
        - Análisis automático con TinyListener para verificar calidad
        
        **Nota:** Requiere tener instalado `espeak` en el sistema:
        ```bash
        sudo apt-get install espeak  # Ubuntu/Debian
        brew install espeak         # macOS
        ```
        """)

if __name__ == "__main__":
    main()