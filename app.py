"""
TinySpeak - Dashboard Principal del Sistema Multimodal
"""
import streamlit as st
from PIL import Image
from components.modern_sidebar import display_modern_sidebar
from training.config import load_master_dataset_config

# Configurar la página
st.set_page_config(
    page_title="TinySpeak Home",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar moderna
    display_modern_sidebar("home")
    
    st.title("TinySpeak: El Cerebro Artificial que Lee")
    st.markdown("### 🧠 Explorando las Bases Biológicas de la Lectura")
    
    tab1, tab2 = st.tabs(["📖 Fundamentos Teóricos", "🤖 Arquitectura de Modelos"])
    
    with tab1:
        st.markdown("""
        TinySpeak es un experimento de **Neurociencia Computacional** que busca simular cómo el cerebro humano aprende a leer a través de la integración de las rutas **Auditiva** y **Visual**. El sistema modela el proceso de adquisición de la lectura basándose en la neurobiología del lenguaje.
        
        #### 1. Conciencia Fonológica y Síntesis Auditiva
        Nuestra hipótesis central sostiene que el aprendizaje de la lectura depende críticamente de la **conciencia fonológica**: la capacidad de identificar, manipular y concatenar sonidos individuales (fonemas). 
        *   **El Proceso:** Al leer, el cerebro concatena los fonemas decodificados visualmente para formar una **imagen auditiva** de la palabra.
        *   **El Reconocimiento:** Esta imagen auditiva se contrasta con el léxico mental preexistente. Si la concatenación coincide con una palabra conocida, ocurre el reconocimiento léxico.
        
        #### 2. Hipótesis de Transparencia Ortográfica
        El experimento evalúa esta capacidad en tres idiomas con distintos niveles de complejidad en su relación grafema-fonema:
        *   **🇪🇸 Español:** Ortografía **transparente**. Existe una relación casi biunívoca entre letras y sonidos, facilitando la ruta fonológica inicial.
        *   **🇫🇷 Francés:** Ortografía **intermedia/opaca**. Posee reglas posicionales complejas combinadas con múltiples grafemas asociados a diferentes sonidos o letras silenciosas.
        *   **🇺🇸 Inglés:** Ortografía **opaca** o altamente irregular. Un mismo grafema puede representar múltiples sonidos (o ninguno) dependiendo del contexto, exigiendo un modelo de predicción más robusto y mayor dependencia de la ruta léxica.
        
        #### 3. Codificación Predictiva y Plausibilidad
        El modelo implementa el principio de **Codificación Predictiva**: el sistema aprende minimizando el error entre la "imaginación" del sonido (generada a partir de la visión) y la percepción auditiva real.
        """)
        
        st.info("💡 Configura el diccionario y los parámetros en **00 Configuracion Experimento** para iniciar la regeneración de datos.")

    with tab2:
        st.markdown("### 🛠️ Eslabones de la Arquitectura Cognitiva")
        
        with st.expander("👂 TinyEars (Corteza Auditiva & Procesamiento Perceptual)", expanded=False):
            st.markdown("""
            **Función:** Simula el procesamiento desde la cóclea hasta la corteza auditiva primaria.
            
            **Detalles Técnicos:**
            *   **Espectrograma de Mel:** El modelo no procesa audio en bruto, sino representaciones en la escala Mel, que imita la resolución no lineal del oído humano (más sensible a frecuencias bajas).
            *   **Tono-topía:** La arquitectura de convoluciones 1D y 2D sobre el espectrograma emula la organización tono-tópica de la corteza auditiva, donde neuronas específicas responden a rangos de frecuencia determinados.
            *   **Rol:** Provee el "espacio latente de referencia" al cual el resto del sistema debe aspirar a llegar.
            """)
            
        with st.expander("👁️ TinyEyes (VWFA - Visual Word Form Area)", expanded=False):
            st.markdown("""
            **Función:** Simula la vía ventral del procesamiento visual (desde la corteza temprana V1 hasta el área de la forma visual de las palabras, VWFA).
            
            **Detalles Técnicos:**
            *   **CNNs Jerárquicas:** Utiliza Redes Neuronales Convolucionales para extraer características visuales de los grafemas, imitando el reconocimiento progresivo de bordes y formas.
            *   **Invariancia Perceptual:** Gracias al uso de Data Augmentation y pooling, emula la capacidad humana de reconocer grafemas sin importar ligeras rotaciones, ruido visual o la fuente tipográfica.
            *   **Rol:** Extraer y representar internamente la "identidad visual" de los grafemas.
            """)
            
        with st.expander("🗣️ TinySpeller (Grapheme-to-Phoneme / Sublexical Route)", expanded=False):
            st.markdown("""
            **Función:** Simula la vía subléxica y el proceso transmodal que va de la visión al lenguaje fonológico (Giro Angular y Wernicke temprano).
            
            **Detalles Técnicos:**
            *   **Mapeo Cross-Modal:** Recibe la representación visual de un grafema (extraída por TinyEyes) y genera la serie de embeddings que representan ese fonema en el espacio latente auditivo.
            *   **Generación de Secuencias:** Utiliza redes recurrentes para generar la evolución temporal del sonido de un fonema a partir de una única entrada visual (por ejemplo, el grafema complejo 'ch').
            *   **Rol:** Actúa como un "generador de imaginación sonora" a nivel de fonema, cuyas salidas sirven de base para que el modelo final aprenda a ensamblar la palabra completa.
            """)
            
        with st.expander("🧠 TinyReader (Phoneme-to-Word / Integration)", expanded=False):
            st.markdown("""
            **Función:** Simula la integración fonológica superior para ensamblar los componentes sonoros en unidades léxicas estructuradas.
            
            **Detalles Técnicos:**
            *   **Pooling Temporal:** Resume la secuencia continua de estados de TinySpeller generados en el tiempo.
            *   **Perceptual Loss y Soft-DTW:** El modelo se alinea por minimización del error de predicción auditivo. Ajusta y comprime las series de tiempo mediante Dynamic Time Warping permitiendo que las velocidades de fluidez en la síntesis simulada sean variables.
            *   **Rol:** Une los fonemas reconstruidos para comparar su forma conjunta frente al registro auditivo de palabras reales escuchadas previamente.
            """)

    st.markdown("---")
    st.caption("TinySpeak Project - Investigando la Transparencia Ortográfica y la Cognición Multimodal")

if __name__ == "__main__":
    main()