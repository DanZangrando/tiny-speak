# 🧠 TinySpeak: Sistema Cognitivo Multimodal

**TinySpeak** es una plataforma de investigación y educación en Inteligencia Artificial que simula los procesos cognitivos humanos de **audición**, **visión** e **imaginación**. Construido con PyTorch Lightning y Streamlit, ofrece una interfaz interactiva para entrenar, evaluar y experimentar con modelos de Deep Learning de vanguardia.

## ✨ Características Principales

### 🤖 Arquitectura Cognitiva Modular
El sistema se divide en tres "vías" o agentes especializados:

1.  **👂 PhonologicalPathway (El Oído)**
    *   **Modelo**: MelSpectrogram + Transformer Encoder.
    *   **Función**: Reconocimiento Automático del Habla (ASR).
    *   **Capacidad**: Entiende palabras habladas y las mapea a conceptos.
    *   **Innovación**: Arquitectura ligera entrenada desde cero para eficiencia.

2.  **👁️ VisualPathway (La Vista)**
    *   **Modelo**: CNN + Linear Decoder.
    *   **Función**: Reconocimiento Óptico de Caracteres (OCR).
    *   **Capacidad**: Lee letras manuscritas y tipografías variadas.
    *   **Innovación**: Simula la vía ventral del procesamiento visual humano.

3.  **🧠 TinyReader (La Voz Interior)**
    *   **Modelo**: Transformer Decoder (Spelling-to-Audio).
    *   **Función**: Imaginación Auditiva.
    *   **Capacidad**: "Lee" una secuencia de letras y genera una alucinación auditiva (embedding) de cómo debería sonar.
    *   **Innovación**: Entrenamiento con **Pérdida Perceptual**, usando al *PhonologicalPathway* como juez para validar sus imaginaciones.

### 📊 Analítica Avanzada e Interactiva
Cada modelo cuenta con un panel de control profesional:
*   **Curvas de Aprendizaje**: Gráficos interactivos de pérdida y precisión en tiempo real (Plotly).
*   **Predicciones en Vivo**: Visualización animada de lo que el modelo "piensa" mientras entrena.
*   **Matrices de Confusión**: Mapas de calor para visualizar errores de clasificación.
*   **Espacio Latente 3D**: Proyección PCA interactiva para explorar cómo la IA organiza los conceptos.

### 🔬 Experimento de Transparencia
Un módulo dedicado para validar la hipótesis científica del proyecto:
*   **Entrenamiento Multi-idioma**: Ejecución automatizada de experimentos en Español, Inglés y Francés.
*   **Evaluación Cruzada**: Comparación de rendimiento entre idiomas y modelos.
*   **Laboratorio Comparativo**: Prueba interactiva donde escribes una palabra y ves cómo cada "cerebro" (ES/EN/FR) la imagina y pronuncia.

## 🏗️ Estructura del Proyecto

```
TinySpeak/
├── 🎯 app.py                    # Punto de entrada de la aplicación
├── 🧠 models.py                 # Arquitecturas de redes neuronales (PyTorch)
├── 🔧 utils.py                  # Utilidades compartidas (audio, visualización)
├── 
├── 📄 pages/                    # Interfaz de Usuario (Streamlit)
│   ├── 01_👂_Audio_Dataset.py   # Gestión de datos de audio
│   ├── 02_👁️_Visual_Dataset.py  # Gestión de datos visuales
│   ├── 03_👂_Audio_Analytics.py # Exploración de datos
│   ├── 05_👂_TinyListener.py    # Entrenamiento y Lab: Listener
│   ├── 06_👁️_VisualPathway.py   # Entrenamiento y Lab: Recognizer
│   ├── 07_👁️👂_TinyReader.py    # Entrenamiento y Lab: Reader
│   ├── 08_🔬_Transparency_Experiment.py # Experimento Científico Automatizado
│   └── README.md                # Documentación detallada de páginas
│
├── 📁 components/               # Componentes UI reutilizables
│   ├── analytics.py             # Motores de visualización y métricas
│   ├── diagrams.py              # Generadores de diagramas de arquitectura
│   └── README.md                # Documentación de componentes
│
├── 🏋️ training/                 # Lógica de Entrenamiento (Lightning)
│   ├── audio_module.py          # LightningModule: Listener
│   ├── visual_module.py         # LightningModule: Recognizer
│   ├── reader_module.py         # LightningModule: Reader
│   └── README.md                # Documentación de entrenamiento
│
├── 📁 models/                   # Checkpoints y metadatos guardados
├── 📁 data/                     # Datasets crudos y procesados
└── 📁 metrics/                  # Logs de entrenamiento (JSON)
```

## 🚀 Instalación y Uso

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/tu-usuario/tiny-speak.git
    cd tiny_speak
    ```

2.  **Crear entorno virtual** (Recomendado):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    # .venv\Scripts\activate   # Windows
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación**:
    ```bash
    streamlit run app.py
    ```

## 🔬 Fundamentos Científicos
Este proyecto explora conceptos avanzados de IA:
*   **Self-Supervised Learning**: Uso de Wav2Vec 2.0.
*   **Transfer Learning**: Adaptación de modelos pre-entrenados a tareas específicas.
*   **Multimodal Learning**: Integración de visión y audio en un espacio latente común.
*   **Generative AI**: Creación de representaciones sintéticas a partir de conceptos abstractos.

---
*Desarrollado con ❤️ para la investigación en IA Cognitiva.*