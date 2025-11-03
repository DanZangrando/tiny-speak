"""
🔗 TinySpeller - Modelo Multimodal
Página para testing del modelo que combina visión y audio
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tempfile
import os
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

# Importar módulos
from models import TinySpeak, TinyListener, TinyRecognizer, TinySpeller
from utils import (
    encontrar_device, load_wav2vec_model, get_default_words, 
    synthesize_word, save_waveform_to_audio_file, WAV2VEC_SR, WAV2VEC_DIM, LETTERS
)

# Importar sidebar moderna
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from components.modern_sidebar import display_modern_sidebar

# Configurar página
st.set_page_config(
    page_title="TinySpeller - Multimodal AI",
    page_icon="🔗",
    layout="wide"
)

@st.cache_resource
def load_multimodal_models():
    """Cargar todos los modelos para funcionalidad multimodal"""
    device = encontrar_device()
    
    # Wav2Vec2 para audio
    wav2vec_model = load_wav2vec_model(device=device)
    words = get_default_words()
    
    # Modelos principales
    tiny_speak = TinySpeak(words=words, hidden_dim=64, num_layers=2, wav2vec_dim=WAV2VEC_DIM)
    tiny_listener = TinyListener(tiny_speak=tiny_speak, wav2vec_model=wav2vec_model)
    tiny_recognizer = TinyRecognizer(wav2vec_dim=WAV2VEC_DIM)
    tiny_speller = TinySpeller(tiny_recognizer=tiny_recognizer, tiny_speak=tiny_speak)
    
    # Mover al dispositivo
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
    # Sidebar modernizada persistente
    display_modern_sidebar()
    
    st.title("🔗 TinySpeller - Reconocimiento Multimodal")
    
    # Información del modelo
    with st.expander("🔍 Arquitectura Multimodal", expanded=True):
        st.markdown("""
        ### 🧠 **TinySpeller: Visión + Audio → Palabra**
        
        ```
        📷 Secuencia de Imágenes (Letras)     🎵 Audio Directo
                    ↓                              ↓
        🖼️ TinyRecognizer (CORnet-Z)         🎤 TinyListener (Wav2Vec2+LSTM)
        - Reconoce letra individual              - Audio → Embedding 768D
        - Output: Embedding 768D                 - Clasificador → Palabra directa
                    ↓                              
        🔄 Secuencia de Embeddings                
        - [embed_L1, embed_L2, ..., embed_Ln]    
                    ↓                              
        📝 TinySpeak (LSTM + Clasificador)        
        - Procesa secuencia de embeddings         
        - Output: Palabra predicha                
        ```
        
        ### 🎯 **Casos de Uso:**
        1. **Imagen → Palabra**: Secuencia de letras manuscritas → palabra completa
        2. **Audio → Palabra**: Audio hablado → reconocimiento directo  
        3. **Comparación Multimodal**: Comparar resultados entre modalidades
        4. **Síntesis + Reconocimiento**: Generar audio → reconocer → validar
        """)
    
    # Cargar modelos
    if 'multimodal_models' not in st.session_state:
        with st.spinner("🤖 Cargando modelos multimodales..."):
            st.session_state.multimodal_models = load_multimodal_models()
    
    models = st.session_state.multimodal_models
    
    # Métricas del sistema
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Vocabulario", f"{len(models['words'])} palabras")
    with col2:
        st.metric("🔤 Letras", f"{len(LETTERS)} clases")
    with col3:
        st.metric("🧠 Parámetros Total", f"{sum(p.numel() for p in models['tiny_speller'].parameters()):,}")
    with col4:
        st.metric("🔗 Modalidades", "Visión + Audio")
    
    # Pestañas para diferentes tipos de testing
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️➡️📝 Imagen→Palabra", "🎵➡️📝 Audio→Palabra", "⚖️ Comparación", "🔬 Análisis Avanzado"])
    
    with tab1:
        test_image_to_word(models)
    
    with tab2:
        test_audio_to_word(models)
        
    with tab3:
        test_multimodal_comparison(models)
        
    with tab4:
        test_advanced_analysis(models)

def test_image_to_word(models):
    """Test de secuencia de imágenes a palabra"""
    st.subheader("🖼️ Test: Secuencia de Letras → Palabra")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ⚙️ Configuración")
        
        # Selector de palabra para generar
        word_source = st.radio(
            "Fuente de la palabra:",
            ["Del vocabulario", "Palabra personalizada"]
        )
        
        if word_source == "Del vocabulario":
            target_word = st.selectbox(
                "Selecciona palabra del vocabulario:",
                models['words'][:50]  # Primeras 50 para el selector
            )
        else:
            target_word = st.text_input(
                "Escribe una palabra:",
                value="hola",
                max_chars=10,
                help="Solo letras a-z, máximo 10 caracteres"
            ).lower()
        
        # Validar que solo contenga letras válidas
        if target_word and all(c in LETTERS for c in target_word):
            st.success(f"✅ Palabra válida: **{target_word}** ({len(target_word)} letras)")
            
            if st.button("🎨 Generar Secuencia de Letras", type="primary"):
                generate_letter_sequence(target_word, models)
        
        elif target_word:
            st.error(f"❌ La palabra contiene caracteres inválidos. Solo usar letras a-z.")
    
    with col2:
        if 'letter_sequence_results' in st.session_state:
            display_sequence_results(st.session_state.letter_sequence_results)

def test_audio_to_word(models):
    """Test de audio directo a palabra"""
    st.subheader("🎵 Test: Audio → Palabra Directa")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎤 Input de Audio")
        
        # Opción 1: Audio grabado
        st.markdown("**Grabación directa:**")
        recorded_audio = st.audio_input("Graba una palabra")
        
        # Opción 2: Síntesis
        st.markdown("**O síntesis de palabra:**")
        synth_word = st.text_input("Palabra para sintetizar:", value="casa")
        
        col_params1, col_params2 = st.columns(2)
        with col_params1:
            rate = st.slider("Velocidad", 50, 200, 80)
        with col_params2:
            pitch = st.slider("Tono", 0, 100, 50)
        
        if st.button("🔊 Sintetizar y Analizar"):
            synthesize_and_analyze_audio(synth_word, rate, pitch, models)
        
        # Análisis de audio grabado
        if recorded_audio and st.button("🔍 Analizar Grabación"):
            analyze_recorded_audio(recorded_audio, models)
    
    with col2:
        if 'audio_results' in st.session_state:
            display_audio_results(st.session_state.audio_results)

def test_multimodal_comparison(models):
    """Comparación entre modalidades"""
    st.subheader("⚖️ Comparación Multimodal")
    
    st.markdown("""
    **Objetivo:** Comparar cómo cada modalidad reconoce la misma palabra
    """)
    
    # Configuración
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ Configuración del Test")
        
        test_word = st.selectbox(
            "Palabra para test comparativo:",
            models['words'][:30]
        )
        
        modalities_to_test = st.multiselect(
            "Modalidades a comparar:",
            ["🖼️ Secuencia de Letras", "🎵 Audio Sintetizado", "🎤 Audio Directo"],
            default=["🖼️ Secuencia de Letras", "🎵 Audio Sintetizado"]
        )
        
        if st.button("🚀 Ejecutar Comparación Multimodal"):
            run_multimodal_comparison(test_word, modalities_to_test, models)
    
    with col2:
        if 'comparison_results' in st.session_state:
            display_comparison_results(st.session_state.comparison_results)

def test_advanced_analysis(models):
    """Análisis avanzado del sistema multimodal"""
    st.subheader("🔬 Análisis Avanzado del Sistema")
    
    # Análisis de arquitectura
    with st.expander("🏗️ Análisis de Arquitectura Completa"):
        display_architecture_analysis(models)
    
    # Análisis de embeddings
    with st.expander("🧠 Análisis de Espacios de Embeddings"):
        analyze_embedding_spaces(models)
    
    # Benchmark del sistema
    with st.expander("⚡ Benchmark de Rendimiento"):
        run_performance_benchmark(models)

def generate_letter_sequence(word, models):
    """Genera secuencia de imágenes para una palabra"""
    try:
        st.info(f"🎨 Generando secuencia para: **{word}**")
        
        # Generar imagen para cada letra
        letter_images = []
        letter_tensors = []
        
        # Configuración de imagen
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize(mean, std)
        ])
        
        for letter in word:
            # Crear imagen de la letra
            img = Image.new('RGB', (28, 28), 'white')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Centrar texto
            bbox = draw.textbbox((0, 0), letter.upper(), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (28 - text_width) // 2
            y = (28 - text_height) // 2
            
            draw.text((x, y), letter.upper(), fill='black', font=font)
            
            letter_images.append(img)
            letter_tensors.append(transform(img))
        
        # Crear tensor de secuencia
        sequence_tensor = torch.stack(letter_tensors).unsqueeze(0).to(models['device'])
        
        # Procesar con TinySpeller
        models['tiny_speller'].eval()
        with torch.no_grad():
            logits, hidden_states = models['tiny_speller'](sequence_tensor)
        
        # Guardar resultados
        predicted_idx = logits.argmax(dim=1).item()
        predicted_word = models['words'][predicted_idx]
        confidence = torch.softmax(logits, dim=1).max().item()
        
        results = {
            'target_word': word,
            'predicted_word': predicted_word,
            'confidence': confidence,
            'letter_images': letter_images,
            'logits': logits.cpu(),
            'correct': word == predicted_word
        }
        
        st.session_state.letter_sequence_results = results
        
    except Exception as e:
        st.error(f"❌ Error generando secuencia: {str(e)}")

def display_sequence_results(results):
    """Muestra resultados de secuencia de letras"""
    st.markdown("#### 📊 Resultados")
    
    # Mostrar secuencia de letras generada
    st.markdown("**🔤 Secuencia Generada:**")
    cols = st.columns(len(results['letter_images']))
    for i, img in enumerate(results['letter_images']):
        with cols[i]:
            st.image(img, caption=f"Letra {i+1}", width=60)
    
    # Resultado de predicción
    if results['correct']:
        st.success(f"✅ **Correcto!** Predicha: {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
    else:
        st.error(f"❌ **Incorrecto.** Esperada: {results['target_word']}, Predicha: {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
    
    # Top 5 predicciones
    probabilities = torch.softmax(results['logits'], dim=1).squeeze().numpy()
    top_indices = np.argsort(probabilities)[::-1][:5]
    
    st.markdown("**🏆 Top 5 Predicciones:**")
    for i, idx in enumerate(top_indices):
        word = st.session_state.multimodal_models['words'][idx]
        prob = probabilities[idx]
        emoji = "🎯" if i == 0 else "📍"
        st.write(f"{emoji} {word} ({prob:.2%})")

def synthesize_and_analyze_audio(word, rate, pitch, models):
    """Sintetiza y analiza audio"""
    try:
        # Sintetizar
        waveform = synthesize_word(word, rate=rate, pitch=pitch)
        
        if waveform is not None:
            # Analizar con TinyListener
            device = models['device']
            waveform_device = waveform.to(device)
            
            models['tiny_listener'].eval()
            with torch.no_grad():
                logits, hidden_states = models['tiny_listener']([waveform_device])
            
            # Guardar resultados
            predicted_idx = logits.argmax(dim=1).item()
            predicted_word = models['words'][predicted_idx]
            confidence = torch.softmax(logits, dim=1).max().item()
            
            results = {
                'target_word': word,
                'predicted_word': predicted_word,
                'confidence': confidence,
                'waveform': waveform,
                'logits': logits.cpu(),
                'correct': word == predicted_word,
                'source': 'synthesis'
            }
            
            st.session_state.audio_results = results
        
        else:
            st.error("❌ Error en síntesis de audio")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def analyze_recorded_audio(audio_file, models):
    """Analiza audio grabado"""
    try:
        from utils import load_waveform
        
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        waveform = load_waveform(tmp_path, target_sr=WAV2VEC_SR)
        os.unlink(tmp_path)
        
        if waveform is not None:
            # Analizar
            device = models['device']
            waveform_device = waveform.to(device)
            
            models['tiny_listener'].eval()
            with torch.no_grad():
                logits, hidden_states = models['tiny_listener']([waveform_device])
            
            predicted_idx = logits.argmax(dim=1).item()
            predicted_word = models['words'][predicted_idx]
            confidence = torch.softmax(logits, dim=1).max().item()
            
            results = {
                'target_word': 'unknown',
                'predicted_word': predicted_word,
                'confidence': confidence,
                'waveform': waveform,
                'logits': logits.cpu(),
                'correct': None,
                'source': 'recording'
            }
            
            st.session_state.audio_results = results
        
        else:
            st.error("❌ Error cargando audio")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def display_audio_results(results):
    """Muestra resultados de análisis de audio"""
    st.markdown("#### 🎧 Resultados de Audio")
    
    # Reproducir audio
    if results['source'] == 'synthesis':
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
    
    # Resultados
    if results['correct'] is not None:
        if results['correct']:
            st.success(f"✅ **Correcto!** {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
        else:
            st.warning(f"⚠️ Esperada: {results['target_word']}, Predicha: {results['predicted_word']} ({results['confidence']:.2%})")
    else:
        st.info(f"🎯 **Predicción:** {results['predicted_word']} (Confianza: {results['confidence']:.2%})")
    
    # Waveform
    from utils import plot_waveform
    fig = plot_waveform(results['waveform'], "Audio Analizado")
    st.pyplot(fig)

def run_multimodal_comparison(word, modalities, models):
    """Ejecuta comparación entre modalidades"""
    st.info(f"🔄 Ejecutando comparación para: **{word}**")
    
    results = {'word': word, 'comparisons': {}}
    
    # Test de secuencia de letras
    if "🖼️ Secuencia de Letras" in modalities:
        try:
            generate_letter_sequence(word, models)
            if 'letter_sequence_results' in st.session_state:
                seq_results = st.session_state.letter_sequence_results
                results['comparisons']['vision'] = {
                    'predicted': seq_results['predicted_word'],
                    'confidence': seq_results['confidence'],
                    'correct': seq_results['correct']
                }
        except Exception as e:
            results['comparisons']['vision'] = {'error': str(e)}
    
    # Test de audio sintetizado
    if "🎵 Audio Sintetizado" in modalities:
        try:
            synthesize_and_analyze_audio(word, 80, 50, models)
            if 'audio_results' in st.session_state:
                audio_results = st.session_state.audio_results
                results['comparisons']['audio_synth'] = {
                    'predicted': audio_results['predicted_word'],
                    'confidence': audio_results['confidence'],
                    'correct': audio_results['correct']
                }
        except Exception as e:
            results['comparisons']['audio_synth'] = {'error': str(e)}
    
    st.session_state.comparison_results = results

def display_comparison_results(results):
    """Muestra resultados de comparación multimodal"""
    st.markdown("#### 📊 Resultados de Comparación")
    
    word = results['word']
    comparisons = results['comparisons']
    
    # Tabla comparativa
    st.markdown(f"**Palabra objetivo:** {word}")
    
    for modality, result in comparisons.items():
        if 'error' in result:
            st.error(f"❌ {modality}: Error - {result['error']}")
        else:
            status = "✅" if result['correct'] else "❌"
            st.write(f"{status} **{modality}**: {result['predicted']} (conf: {result['confidence']:.2%})")
    
    # Análisis conjunto
    if len(comparisons) > 1:
        predictions = [r['predicted'] for r in comparisons.values() if 'predicted' in r]
        if len(set(predictions)) == 1:
            st.success("🎉 **Consenso:** Todas las modalidades coinciden!")
        else:
            st.warning("⚠️ **Discrepancia:** Las modalidades difieren en la predicción")

def display_architecture_analysis(models):
    """Muestra análisis detallado de arquitectura"""
    st.markdown("### 🏗️ Análisis Completo del Sistema")
    
    # Componentes del sistema
    components = {
        'TinyRecognizer (CORnet-Z)': models['tiny_recognizer'],
        'TinySpeak (LSTM)': models['tiny_speak'],  
        'TinyListener (Wav2Vec2+LSTM)': models['tiny_listener'],
        'TinySpeller (Vision+Audio)': models['tiny_speller']
    }
    
    total_params = 0
    
    for name, model in components.items():
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params += params
        
        st.markdown(f"**{name}:**")
        st.write(f"- Parámetros totales: {params:,}")
        st.write(f"- Parámetros entrenables: {trainable:,}")
        st.write(f"- Parámetros congelados: {params - trainable:,}")
        st.write("")
    
    st.metric("🧠 Sistema Completo", f"{total_params:,} parámetros")

def analyze_embedding_spaces(models):
    """Analiza espacios de embeddings"""
    st.markdown("### 🧠 Análisis de Espacios de Embeddings")
    
    # Por ahora información teórica
    st.markdown("""
    **Espacios de representación en TinySpeak:**
    
    1. **Wav2Vec2 Features (768D)**: Características acústicas del audio
    2. **CORnet-Z Features (768D)**: Características visuales de letras  
    3. **LSTM Hidden States (64D)**: Estados internos de secuencia
    4. **Word Embeddings**: Espacio de palabras del vocabulario
    
    **Hipótesis:** Los embeddings de audio y visión deberían ser similares 
    para la misma letra/palabra, permitiendo transferencia entre modalidades.
    """)
    
    if st.button("🔬 Analizar Embeddings de Ejemplo"):
        # Crear ejemplo simple
        letter = 'a'
        
        # Embedding visual
        img = Image.new('RGB', (28, 28), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((8, 5), letter.upper(), fill='black')
        
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(img).unsqueeze(0).to(models['device'])
        
        models['tiny_recognizer'].eval()
        with torch.no_grad():
            _, visual_embed = models['tiny_recognizer'](image_tensor)
        
        st.success(f"✅ Embedding visual para '{letter}': {visual_embed.shape}")
        
        # Mostrar estadísticas básicas
        embed_np = visual_embed.squeeze().cpu().numpy()
        st.write(f"- Media: {embed_np.mean():.4f}")
        st.write(f"- Std: {embed_np.std():.4f}")
        st.write(f"- Min: {embed_np.min():.4f}")
        st.write(f"- Max: {embed_np.max():.4f}")

def run_performance_benchmark(models):
    """Ejecuta benchmark de rendimiento"""
    st.markdown("### ⚡ Benchmark de Rendimiento")
    
    if st.button("🚀 Ejecutar Benchmark"):
        import time
        
        # Benchmark de TinyRecognizer
        img = Image.new('RGB', (28, 28), 'white')
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(img).unsqueeze(0).to(models['device'])
        
        # Tiempo de inferencia visual
        models['tiny_recognizer'].eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _, _ = models['tiny_recognizer'](image_tensor)
        vision_time = (time.time() - start_time) / 100
        
        # Síntesis de audio para benchmark
        waveform = synthesize_word("test")
        if waveform is not None:
            waveform_device = waveform.to(models['device'])
            
            # Tiempo de inferencia audio
            models['tiny_listener'].eval()
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):  # Menos iteraciones por ser más lento
                    _, _ = models['tiny_listener']([waveform_device])
            audio_time = (time.time() - start_time) / 10
        else:
            audio_time = None
        
        # Mostrar resultados
        st.success("✅ Benchmark completado!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🖼️ Visión (ms)", f"{vision_time*1000:.2f}")
        with col2:
            if audio_time:
                st.metric("🎵 Audio (ms)", f"{audio_time*1000:.2f}")
            else:
                st.metric("🎵 Audio", "Error")

if __name__ == "__main__":
    main()