# TinySpeak - Aplicación de Reconocimiento Multimodal

Esta aplicación de Streamlit implementa los modelos desarrollados en el notebook `tiny_speak.ipynb` para reconocimiento de voz y visión.

## 🚀 Características

- **TinyListener**: Reconocimiento de palabras a partir de audio usando Wav2Vec2
- **TinyRecognizer**: Reconocimiento de letras escritas a mano
- **Síntesis de Voz**: Generación de audio con espeak y análisis automático

## 📋 Requisitos

- Python 3.8+
- Entorno virtual configurado
- espeak (para síntesis de voz)

### Instalación de espeak

```bash
# Ubuntu/Debian
sudo apt-get install espeak

# macOS
brew install espeak

# Windows
# Descargar desde: http://espeak.sourceforge.net/download.html
```

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