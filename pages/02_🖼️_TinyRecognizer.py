"""
🖼️ TinyRecognizer - Reconocimiento de Imágenes
Página dedicada para testing del modelo de visión para letras
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import string
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

# Importar módulos
from models import TinyRecognizer, CORnet_Z
from utils import encontrar_device, WAV2VEC_DIM, LETTERS

# Importar sidebar moderna
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from components.modern_sidebar import display_modern_sidebar

# Configurar página
st.set_page_config(
    page_title="TinyRecognizer - Image Recognition",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_resource
def load_vision_models():
    """Cargar solo los modelos necesarios para visión"""
    device = encontrar_device()
    
    tiny_recognizer = TinyRecognizer(wav2vec_dim=WAV2VEC_DIM)
    tiny_recognizer = tiny_recognizer.to(device)
    
    return {
        'device': device,
        'tiny_recognizer': tiny_recognizer,
        'letters': LETTERS
    }

def main():
    # Sidebar modernizada persistente
    display_modern_sidebar()
    
    st.title("🖼️ TinyRecognizer - Reconocimiento de Letras")
    
    # Información del modelo
    with st.expander("🔍 Arquitectura del Modelo", expanded=True):
        st.markdown("""
        ### 🧠 **TinyRecognizer Architecture (CORnet-Z)**
        
        ```
        Image Input (28x28x3 RGB)
               ↓
        🎯 V1 Block (Visual Cortex Area 1)
        - Conv2d(3→64, kernel=7, stride=2) 
        - ReLU + MaxPool2d(3x3, stride=2)
        - Output: 64 channels
               ↓
        👁️ V2 Block (Visual Cortex Area 2)  
        - Conv2d(64→128, kernel=3)
        - ReLU + MaxPool2d(3x3, stride=2)
        - Output: 128 channels
               ↓
        🔍 V4 Block (Visual Cortex Area 4)
        - Conv2d(128→256, kernel=3) 
        - ReLU + MaxPool2d(3x3, stride=2)
        - Output: 256 channels
               ↓
        🧠 IT Block (Inferotemporal Cortex)
        - Conv2d(256→512, kernel=3)
        - ReLU + MaxPool2d(3x3, stride=2) 
        - Output: 512 channels
               ↓
        🎨 Decoder
        - AdaptiveAvgPool2d(1x1)
        - Flatten → Linear(512→1024) → ReLU
        - Linear(1024→768) [Embedding space]
               ↓
        🎯 Classifier
        - Linear(768→26) [Letter classes a-z]
        ```
        
        **Inspiración:** Arquitectura basada en el sistema visual cortical humano
        """)
    
    # Cargar modelos
    if 'vision_models' not in st.session_state:
        with st.spinner("🤖 Cargando modelos de visión..."):
            st.session_state.vision_models = load_vision_models()
    
    models = st.session_state.vision_models
    
    # Métricas del modelo
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔤 Clases", f"{len(LETTERS)} letras")
    with col2:
        st.metric("🧠 Parámetros", f"{sum(p.numel() for p in models['tiny_recognizer'].parameters()):,}")
    with col3:
        st.metric("📊 Embedding Dim", f"{WAV2VEC_DIM}")
    with col4:
        st.metric("🖼️ Input Size", "28×28×3")
    
    # Pestañas para diferentes tipos de testing
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Cargar Imagen", "✏️ Dibujar Letra", "🔬 Test Sistemático", "📊 Análisis Interno"])
    
    with tab1:
        test_image_upload(models)
    
    with tab2:
        test_drawing_interface(models)
        
    with tab3:
        test_systematic_evaluation(models)
        
    with tab4:
        test_internal_analysis(models)

def test_image_upload(models):
    """Testing con imágenes subidas"""
    st.subheader("📁 Test con Imagen Cargada")
    
    image_file = st.file_uploader(
        "Sube una imagen de una letra:", 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Imágenes de letras manuscritas preferiblemente 28x28 píxeles"
    )
    
    if image_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Mostrar imagen original
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption="Imagen Original", use_column_width=True)
            
            # Información de la imagen
            st.write(f"**Dimensiones:** {image.size}")
            st.write(f"**Modo:** {image.mode}")
            
        with col2:
            if st.button("🔍 Reconocer Letra", type="primary"):
                analyze_image(image, models, "Imagen Cargada")

def test_drawing_interface(models):
    """Interfaz para dibujar letras"""
    st.subheader("✏️ Dibujar una Letra")
    
    # Por ahora una implementación simple - en el futuro se puede usar streamlit-drawable-canvas
    st.info("🚧 **Próximamente:** Interfaz de dibujo interactiva con canvas")
    st.markdown("""
    **Mientras tanto, puedes:**
    1. Usar cualquier app de dibujo para crear una letra
    2. Guardarla como imagen (PNG/JPG)  
    3. Subirla en la pestaña "Cargar Imagen"
    
    **Recomendaciones para mejores resultados:**
    - Letra clara sobre fondo blanco
    - Tamaño aproximado 28x28 píxeles
    - Trazo negro o oscuro
    """)
    
    # Generar letras de muestra
    if st.button("🎲 Generar Letra de Muestra"):
        generate_sample_letter(models)

def test_systematic_evaluation(models):
    """Evaluación sistemática del alfabeto"""
    st.subheader("🔬 Test Sistemático del Alfabeto")
    
    # Selector de letras para probar
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎯 Configuración del Test")
        
        # Selector de letra específica
        test_mode = st.radio(
            "Modo de test:",
            ["Letra específica", "Rango de letras", "Alfabeto completo"]
        )
        
        if test_mode == "Letra específica":
            selected_letter = st.selectbox("Selecciona letra:", LETTERS)
            letters_to_test = [selected_letter]
        elif test_mode == "Rango de letras":
            start_letter = st.selectbox("Letra inicial:", LETTERS, index=0)
            end_letter = st.selectbox("Letra final:", LETTERS, index=4)
            start_idx = LETTERS.index(start_letter)
            end_idx = LETTERS.index(end_letter)
            letters_to_test = LETTERS[start_idx:end_idx+1]
        else:
            letters_to_test = LETTERS
        
        st.write(f"**Letras a probar:** {len(letters_to_test)}")
        st.write(f"**Lista:** {', '.join(letters_to_test[:10])}{'...' if len(letters_to_test) > 10 else ''}")
        
        if st.button("🚀 Ejecutar Test Sistemático"):
            run_systematic_test(letters_to_test, models)
    
    with col2:
        if 'systematic_results' in st.session_state:
            display_systematic_results(st.session_state.systematic_results)

def test_internal_analysis(models):
    """Análisis interno del modelo"""
    st.subheader("📊 Análisis Interno del Modelo")
    
    # Detalles de arquitectura
    with st.expander("🏗️ Detalles de Arquitectura CORnet-Z"):
        model = models['tiny_recognizer']
        
        st.code(f"""
# TinyRecognizer Architecture
TinyRecognizer(
  (cornet): CORnet_Z(
    (V1): CORblock_Z(Conv2d(3, 64, 7), ReLU, MaxPool2d(3))
    (V2): CORblock_Z(Conv2d(64, 128, 3), ReLU, MaxPool2d(3))  
    (V4): CORblock_Z(Conv2d(128, 256, 3), ReLU, MaxPool2d(3))
    (IT): CORblock_Z(Conv2d(256, 512, 3), ReLU, MaxPool2d(3))
    (decoder): Sequential(
      (avgpool): AdaptiveAvgPool2d(1)
      (flatten): Flatten()
      (linear_input): Linear(512, 1024)
      (relu): ReLU()
      (linear_output): Linear(1024, 768)
    )
  )
  (classifier): Linear(768, 26)
)

# Distribución de parámetros:
CORnet backbone: {sum(p.numel() for p in model.cornet.parameters()):,} parámetros
Classifier: {sum(p.numel() for p in model.classifier.parameters()):,} parámetros
Total: {sum(p.numel() for p in model.parameters()):,} parámetros
        """)
    
    # Visualización de activaciones
    with st.expander("🧠 Análisis de Activaciones"):
        st.markdown("#### 📊 Test con Imagen de Ejemplo")
        
        # Crear imagen de prueba simple
        if st.button("🔬 Analizar Activaciones Internas"):
            analyze_internal_activations(models)

def analyze_image(image, models, title="Imagen"):
    """Analiza una imagen con el modelo"""
    try:
        # Preprocesar imagen
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize(mean, std)
        ])
        
        # Convertir y procesar imagen
        image_tensor = transform(image).unsqueeze(0).to(models['device'])
        
        # Mostrar imagen preprocesada
        st.markdown("#### 🔄 Imagen Preprocesada")
        processed_img = image_tensor.squeeze().cpu()
        # Denormalizar para visualización
        for i in range(3):
            processed_img[i] = processed_img[i] * std[i] + mean[i]
        processed_img = torch.clamp(processed_img, 0, 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(image)
        ax1.set_title("Original")
        ax1.axis('off')
        
        ax2.imshow(processed_img.permute(1, 2, 0))
        ax2.set_title("Preprocesada (28x28)")
        ax2.axis('off')
        
        st.pyplot(fig)
        
        # Hacer predicción
        models['tiny_recognizer'].eval()
        with torch.no_grad():
            logits, embeddings = models['tiny_recognizer'](image_tensor)
        
        # Mostrar resultados
        display_image_results(logits, embeddings, models, title)
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {str(e)}")

def display_image_results(logits, embeddings, models, title):
    """Muestra resultados de reconocimiento de imagen"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎯 Resultados de Predicción")
        
        # Predicción principal
        predicted_idx = logits.argmax(dim=1).item()
        predicted_letter = LETTERS[predicted_idx].upper()
        confidence = torch.softmax(logits, dim=1).max().item()
        
        st.metric("🎯 Letra Predicha", predicted_letter, help=f"Confianza: {confidence:.2%}")
        
        # Top 5
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        st.markdown("**🏆 Top 5:**")
        for i, idx in enumerate(top_indices):
            letter = LETTERS[idx].upper()
            prob = probabilities[idx]
            emoji = "🎯" if i == 0 else "📍"
            st.write(f"{emoji} **{letter}** ({prob:.2%})")
    
    with col2:
        st.markdown("#### 🧠 Embedding Visual")
        
        # Visualizar embedding como imagen
        embedding_2d = embeddings.squeeze().cpu().numpy()
        
        # Reshape a una forma cuadrada aproximada
        dim = int(np.sqrt(len(embedding_2d)))
        if dim * dim < len(embedding_2d):
            # Pad con ceros si es necesario
            pad_size = (dim + 1) * (dim + 1) - len(embedding_2d)
            embedding_2d = np.pad(embedding_2d, (0, pad_size))
            dim = dim + 1
        
        embedding_img = embedding_2d[:dim*dim].reshape(dim, dim)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(embedding_img, cmap='coolwarm')
        ax.set_title(f"Representación Interna ({len(embedding_2d)} dim)")
        plt.colorbar(im)
        st.pyplot(fig)

def generate_sample_letter(models):
    """Genera una letra de muestra simple"""
    st.info("🎲 Generando letra de muestra...")
    
    # Por ahora crear una imagen simple con PIL
    from PIL import Image, ImageDraw, ImageFont
    
    # Crear imagen simple
    img = Image.new('RGB', (28, 28), 'white')
    draw = ImageDraw.Draw(img)
    
    # Dibujar una letra simple (por ejemplo 'A')
    letter = np.random.choice(LETTERS).upper()
    
    try:
        # Intentar usar fuente del sistema
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
    except:
        # Fallback a fuente por defecto
        font = ImageFont.load_default()
    
    # Centrar el texto
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (28 - text_width) // 2
    y = (28 - text_height) // 2
    
    draw.text((x, y), letter, fill='black', font=font)
    
    st.success(f"✅ Letra generada: **{letter}**")
    analyze_image(img, models, f"Letra Generada: {letter}")

def run_systematic_test(letters_to_test, models):
    """Ejecuta test sistemático en múltiples letras"""
    st.info(f"🔄 Ejecutando test en {len(letters_to_test)} letras...")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, letter in enumerate(letters_to_test):
        # Generar imagen de la letra
        img = Image.new('RGB', (28, 28), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Dibujar letra
        bbox = draw.textbbox((0, 0), letter.upper(), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (28 - text_width) // 2
        y = (28 - text_height) // 2
        
        draw.text((x, y), letter.upper(), fill='black', font=font)
        
        # Analizar con el modelo
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize(mean, std)
        ])
        
        image_tensor = transform(img).unsqueeze(0).to(models['device'])
        
        models['tiny_recognizer'].eval()
        with torch.no_grad():
            logits, _ = models['tiny_recognizer'](image_tensor)
        
        predicted_idx = logits.argmax(dim=1).item()
        predicted_letter = LETTERS[predicted_idx]
        confidence = torch.softmax(logits, dim=1).max().item()
        
        results.append({
            'true_letter': letter,
            'predicted_letter': predicted_letter,
            'confidence': confidence,
            'correct': letter == predicted_letter
        })
        
        progress_bar.progress((i + 1) / len(letters_to_test))
    
    # Guardar resultados
    st.session_state.systematic_results = results
    st.success("✅ Test sistemático completado!")

def display_systematic_results(results):
    """Muestra resultados del test sistemático"""
    st.markdown("#### 📊 Resultados del Test Sistemático")
    
    # Métricas generales
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Precisión", f"{accuracy:.2%}")
    with col2:
        st.metric("✅ Correctas", f"{correct}/{total}")
    with col3:
        st.metric("📊 Confianza Promedio", f"{np.mean([r['confidence'] for r in results]):.2%}")
    
    # Tabla de resultados
    st.markdown("#### 📋 Resultados Detallados")
    
    import pandas as pd
    df = pd.DataFrame(results)
    df['status'] = df['correct'].apply(lambda x: '✅' if x else '❌')
    df['confidence_pct'] = df['confidence'].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(
        df[['true_letter', 'predicted_letter', 'confidence_pct', 'status']].rename(columns={
            'true_letter': 'Letra Real',
            'predicted_letter': 'Predicción',
            'confidence_pct': 'Confianza',
            'status': 'Estado'
        }),
        width='stretch'
    )
    
    # Matriz de confusión simple
    if len(results) > 1:
        st.markdown("#### 🔍 Análisis de Errores")
        errors = [r for r in results if not r['correct']]
        if errors:
            st.write(f"**Errores encontrados:** {len(errors)}")
            for error in errors[:5]:  # Mostrar primeros 5 errores
                st.write(f"• {error['true_letter']} → {error['predicted_letter']} (conf: {error['confidence']:.2%})")
        else:
            st.success("🎉 ¡Sin errores detectados!")

def analyze_internal_activations(models):
    """Analiza las activaciones internas del modelo"""
    # Crear una imagen de prueba
    img = Image.new('RGB', (28, 28), 'white')
    draw = ImageDraw.Draw(img)
    
    # Dibujar una 'A' simple
    draw.text((8, 5), 'A', fill='black')
    
    # Procesar imagen
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = Compose([
        Resize((28, 28)),
        ToTensor(),
        Normalize(mean, std)
    ])
    
    image_tensor = transform(img).unsqueeze(0).to(models['device'])
    
    # Extraer activaciones de cada capa
    model = models['tiny_recognizer']
    
    # Hook para capturar activaciones
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Registrar hooks
    model.cornet.V1.register_forward_hook(get_activation('V1'))
    model.cornet.V2.register_forward_hook(get_activation('V2'))
    model.cornet.V4.register_forward_hook(get_activation('V4'))
    model.cornet.IT.register_forward_hook(get_activation('IT'))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(image_tensor)
    
    # Visualizar activaciones
    st.success("✅ Activaciones capturadas!")
    
    for layer_name, activation in activations.items():
        st.markdown(f"#### 🧠 Activaciones {layer_name}")
        
        # Tomar los primeros canales para visualización
        act = activation.squeeze().cpu().numpy()
        if len(act.shape) == 3:  # [channels, height, width]
            n_channels_to_show = min(8, act.shape[0])
            
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()
            
            for i in range(n_channels_to_show):
                axes[i].imshow(act[i], cmap='viridis')
                axes[i].set_title(f'Canal {i}')
                axes[i].axis('off')
            
            # Ocultar ejes no usados
            for i in range(n_channels_to_show, 8):
                axes[i].axis('off')
            
            st.pyplot(fig)
            st.write(f"Shape: {act.shape} | Channels mostrados: {n_channels_to_show}")

if __name__ == "__main__":
    main()