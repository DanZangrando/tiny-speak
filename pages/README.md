# 📄 Páginas de la Aplicación

Este directorio contiene los scripts de Streamlit que definen cada una de las páginas de la aplicación **TinySpeak**.

## 🗂️ Estructura y Funcionalidad

### 🛠️ Gestión de Datasets
*   **`01_👂_Audio_Dataset.py`**:
    *   Gestión del dataset de audio (TinyKalulu, TinyPhones).
    *   Descarga de datasets base.
    *   Síntesis de audio TTS (Text-to-Speech) para generar nuevas muestras.
    *   Validación y preprocesamiento de audio.

*   **`02_👁️_Visual_Dataset.py`**:
    *   Gestión del dataset visual (letras y grafemas).
    *   Generación sintética de imágenes de letras con diferentes fuentes y transformaciones.
    *   Visualización de muestras generadas.

### 📊 Analíticas de Datos
*   **`03_👂_Audio_Analytics.py`**:
    *   Exploración profunda del dataset de audio.
    *   Visualización de formas de onda y espectrogramas.
    *   Estadísticas de distribución de clases y duración.

*   **`04_👁️_Visual_Analytics.py`**:
    *   Exploración del dataset visual.
    *   Galería de imágenes generadas.
    *   Estadísticas de distribución de clases visuales.

### 🧠 Modelos de Inteligencia Artificial
Cada página de modelo sigue una estructura estandarizada de 4 pestañas: **Arquitectura**, **Entrenamiento**, **Modelos Guardados** y **Laboratorio**.

*   **`05_👂_TinyListener.py` (El Oído)**:
    *   **Modelo**: `TinyListener` (Wav2Vec 2.0 + LSTM).
    *   **Función**: Reconocimiento de palabras habladas (ASR).
    *   **Características**: Entrenamiento con PyTorch Lightning, visualización de métricas en tiempo real, evaluación con mapas de calor de probabilidad.

*   **`06_👁️_TinyRecognizer.py` (La Vista)**:
    *   **Modelo**: `TinyRecognizer` (CORnet-Z).
    *   **Función**: Reconocimiento de caracteres visuales (OCR simplificado).
    *   **Características**: Aprende a identificar letras a partir de imágenes, curvas de aprendizaje interactivas.

*   **`07_👁️👂_TinyReader.py` (La Voz Interior)**:
    *   **Modelo**: `TinyReader` (Generativo Top-Down).
    *   **Función**: "Imaginación" auditiva. Convierte conceptos visuales (letras) en representaciones latentes de audio.
    *   **Características**:
        *   **Evaluación Perceptual**: Usa al *TinyListener* como juez para validar si lo que "imagina" se entiende.
        *   **Visualización Latente**: Proyección PCA 3D del espacio vectorial imaginado.