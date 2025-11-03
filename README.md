# TinySpeak - Sistema Multimodal de Reconocimiento de Voz y Visión

TinySpeak es una aplicación completa de Streamlit que combina reconocimiento de voz y visión con capacidades de generación de datasets personalizados.

## 🚀 Características Principales

### 🧠 Modelos de IA
- **TinyListener**: Reconocimiento de palabras usando Wav2Vec2 + LSTM
- **TinyRecognizer**: Reconocimiento de letras manuscritas con arquitectura CORnet-Z
- **TinySpeller**: Sistema multimodal que combina audio y visión

### � Gestión de Vocabularios
- **Diccionarios Predefinidos**: Incluye vocabularios originales (Kalulu, Phones) y temáticos
- **Diccionarios Personalizados**: Crea vocabularios propios palabra por palabra
- **Sincronización Automática**: Configuración centralizada para todos los datasets

### 🎵 Generación de Audio (gTTS)
- **Síntesis con Google Text-to-Speech**: Calidad superior a espeak
- **Variaciones Automáticas**: 6 tipos (original, velocidad, volumen, normalizado)
- **Conversión a WAV**: Procesamiento automático para compatibilidad
- **Sistema de Verificación**: Reproduce y valida cada muestra generada

### 🖼️ Generación de Imágenes
- **Letras Sintéticas**: Genera imágenes de letras con múltiples fuentes
- **Variaciones de Estilo**: Diferentes tipos de letra, tamaños y efectos
- **Dataset Visual Completo**: Para entrenar reconocimiento de caracteres

### 📊 Dashboard Inteligente
- **Métricas Dinámicas**: Estado real de datasets (no estático)
- **Validación de Consistencia**: Verifica sincronización entre configuración y datasets
- **Interfaz Moderna**: Tema oscuro con componentes glassmorfismo

## 📋 Requisitos del Sistema

- Python 3.8+
- Entorno virtual configurado
- Conexión a internet (para gTTS)

### Dependencias Principales
- **Streamlit**: Framework web para la aplicación
- **PyTorch**: Modelos de deep learning
- **Transformers**: Modelo Wav2Vec2 de Hugging Face
- **gTTS**: Google Text-to-Speech para síntesis de audio
- **Plotly**: Gráficos interactivos nativos
- **Pillow**: Procesamiento de imágenes
- **librosa/torchaudio**: Procesamiento de audio

## 🛠️ Instalación

1. Clona o descarga este proyecto
2. Activa tu entorno virtual:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # o
   .venv\Scripts\activate     # Windows
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
tiny_speak/
├── app.py              # Aplicación principal de Streamlit
├── models.py           # Definiciones de los modelos
├── utils.py            # Funciones utilitarias
├── tiny_speak.ipynb    # Notebook original
├── requirements.txt    # Dependencias de Python
├── .streamlit/
│   └── config.toml     # Configuración de Streamlit
└── data/               # Datos descargados automáticamente
```

## 🎯 Funcionalidades

### TinyListener (Audio → Palabra)
- Carga archivos de audio (WAV, MP3, FLAC, M4A)
- Grabación de audio en tiempo real
- Análisis de waveform
- Predicción de palabras con confianza
- Visualización de logits

### TinyRecognizer (Imagen → Letra)
- Carga imágenes de letras manuscritas
- Reconocimiento de letras a-z
- Visualización de embeddings internos
- Métricas de confianza

### Síntesis de Voz
- Generación de audio con espeak
- Parámetros configurables (velocidad, tono, volumen)
- Análisis automático con TinyListener
- Verificación de calidad de síntesis

## 🔧 Configuración

La aplicación descarga automáticamente los datasets necesarios:
- tiny-kalulu-200: Palabras en español
- tiny-phones-200: Fonemas concatenados
- tiny-emnist-26: Letras manuscritas

Los modelos se inicializan automáticamente y detectan el mejor dispositivo disponible (CPU, CUDA, MPS).

## 🐛 Solución de Problemas

### Error "espeak not found"
Instala espeak siguiendo las instrucciones de tu sistema operativo.

### Problemas con CUDA/GPU
La aplicación funciona en CPU. Si tienes GPU, asegúrate de tener las drivers correctas instaladas.

### Datasets no se descargan
Verifica tu conexión a internet. Los datasets se descargan desde Google Drive.

## 🤝 Contribuciones

Este proyecto está basado en el notebook de investigación `tiny_speak.ipynb`. 
Las contribuciones son bienvenidas para mejorar la interfaz y añadir nuevas funcionalidades.

## 📄 Licencia

Este proyecto es de uso educativo y de investigación.