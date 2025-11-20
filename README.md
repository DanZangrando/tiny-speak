# 🎤 TinySpeak - Sistema Multimodal de IA

**TinySpeak** es un sistema completo de inteligencia artificial que combina reconocimiento de voz, visión computacional y síntesis de audio en una aplicación web moderna construida con Streamlit.

## ✨ Características Principales

### 🧠 **Modelos de IA Integrados**
- **🎵 TinyListener**: Reconocimiento de palabras usando Wav2Vec2 + LSTM
- **👁️ TinyRecognizer**: Reconocimiento de letras manuscritas con CORnet-Z 

### 📚 **Gestión Inteligente de Vocabularios**
- **Diccionarios Predefinidos**: Kalulu (español), Phones (fonemas), temáticos
- **Diccionarios Personalizados**: Creación de vocabularios específicos
- **Sincronización Automática**: Configuración centralizada y consistente

### � **Generación Avanzada de Audio**
- **Síntesis gTTS**: Google Text-to-Speech de alta calidad
- **Variaciones Automáticas**: 6 tipos (velocidad, tono, volumen, normalizado)
- **Conversión WAV**: Procesamiento automático para compatibilidad
- **Verificación Inteligente**: Validación automática de cada muestra

### 🖼️ **Generación de Datasets Visuales**
- **Letras Sintéticas**: Múltiples fuentes y estilos tipográficos
- **Variaciones Personalizables**: Tamaños, efectos y transformaciones
- **Dataset Visual Completo**: Para entrenamiento de reconocimiento OCR

### ⚡ **Entrenamiento con PyTorch Lightning**
- **TinyListener Training**: Entrenamiento completo de reconocimiento de audio
- **TinyRecognizer Training**: Entrenamiento de reconocimiento visual
- **Callbacks Avanzados**: Early stopping, checkpoints y métricas en tiempo real

## � Instalación Rápida

### **Prerrequisitos**
- Python 3.8+ 
- Entorno virtual recomendado
- Conexión a internet (para gTTS)

### **Configuración**
```bash
# 1. Clonar el proyecto
git clone [repository-url]
cd tiny_speak

# 2. Activar entorno virtual
source .venv/bin/activate  # Linux/macOS
# o .venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

## 🏗️ Arquitectura del Sistema

```
TinySpeak/
├── 🎯 app.py                    # Aplicación principal Streamlit
├── 🧠 models.py                 # Definiciones de modelos IA
├── 🔧 utils.py                  # Utilidades y helpers
├── 📋 requirements.txt          # Dependencias Python
├── 📊 master_dataset_config.json # Configuración centralizada
├── 
├── 📁 components/               # Componentes UI reutilizables
│   └── modern_sidebar.py        # Sidebar con glassmorphism
├── 
├── 📄 pages/                    # Páginas de la aplicación
│   ├── 01_🎵_Audio_Dataset.py   # Gestión dataset audio
│   ├── 02_🖼️_Visual_Dataset.py  # Gestión dataset visual
│   ├── 03_🎵_Audio_Analytics.py # Analíticas audio
│   ├── 04_🖼️_Visual_Analytics.py # Analíticas visual
│   ├── 05_🎵_TinyListener.py    # Modelo TinyListener
│   └── 06_🖼️_TinyRecognizer.py  # Modelo TinyRecognizer
├── 
├── 🏋️ training/                 # Módulos de entrenamiento
│   ├── audio_module.py          # TinyListener Lightning
│   ├── visual_module.py         # TinyRecognizer Lightning
│   ├── audio_dataset.py         # Datasets de audio
│   ├── visual_dataset.py        # Datasets visuales
│   └── config.py               # Configuración datasets
├── 
├── 📁 data/                     # Datasets descargados
├── 📁 checkpoints/              # Modelos entrenados
├── 📁 visual_dataset/           # Imágenes generadas
└── 📁 .streamlit/               # Configuración UI
```

## 💡 Funcionalidades Detalladas

### 🎵 **TinyListener - Reconocimiento de Audio**
- **Entrada Múltiple**: WAV, MP3, FLAC, M4A, grabación en vivo
- **Análisis Completo**: Waveform, espectrograma, predicciones
- **Métricas Avanzadas**: Confianza, logits, embeddings internos
- **Entrenamiento**: PyTorch Lightning con callbacks personalizados

### 👁️ **TinyRecognizer - Reconocimiento Visual**  
- **Carga Flexible**: Imágenes manuscritas, sintéticas, fotografías
- **Análisis Visual**: Embeddings, mapas de atención, confianza
- **Entrenamiento Avanzado**: CORnet-Z backbone, augmentations automáticas
- **Evaluación**: Métricas por clase, matriz de confusión

### 🎤 **Audio Dataset Manager**
- **Síntesis gTTS**: Múltiples idiomas y voces
- **Variaciones Automáticas**: Speed (0.8x-1.2x), pitch, volumen
- **Postprocesamiento**: Normalización, padding, conversión de formatos
- **Validación**: Reproducción automática y verificación de calidad

### �️ **Visual Dataset Manager**
- **Generación Tipográfica**: 15+ fuentes, múltiples tamaños
- **Augmentations**: Rotación, ruido, blur, transformaciones afines
- **Balanceo Automático**: Distribución equitativa por clase
- **Exportación**: PNG optimizado, metadatos JSON

### 📊 **Dashboard Analytics**
- **Métricas en Tiempo Real**: Estado de datasets, progreso de entrenamiento
- **Visualizaciones**: Plotly interactivo, métricas dinámicas
- **Consistencia**: Verificación automática de sincronización
- **Performance**: Métricas de modelos, comparaciones A/B

## 🔧 Configuración Avanzada

### **Variables de Entorno**
```bash
# Configuración CUDA (opcional)
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Configuración gTTS
export GTTS_LANG=es  # Idioma por defecto
```

### **Configuración Personalizada**
Edita `master_dataset_config.json` para personalizar:
- Vocabularios por defecto
- Rutas de datasets 
- Parámetros de generación
- Configuraciones de entrenamiento

## 🐛 Solución de Problemas

### **Problemas Comunes**

| Error | Causa | Solución |
|-------|-------|----------|
| `ModuleNotFoundError: pytorch_lightning` | Dependencia faltante | `pip install pytorch-lightning` |
| `CUDA deterministic warning` | Configuración PyTorch | Ya corregido en v2.0+ |
| `gTTS network error` | Conexión internet | Verificar conectividad |
| `Tensor size mismatch` | Audio padding | Collate function implementado |

### **Debugging**
```bash
# Verificar instalación
python -c "import torch, streamlit, transformers; print('OK')"

# Test modelo básico
python -c "from models import TinyListener; print('Models OK')"

# Verificar GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📈 Roadmap

### **v2.1 - Próximas Mejoras**
- [ ] **Exportación de Modelos**: ONNX, TorchScript, Hugging Face Hub
- [ ] **API REST**: Endpoints para inferencia programática
- [ ] **Métricas Avanzadas**: WandB integration, experiment tracking
- [ ] **Deployment**: Docker, cloud deployment automático

### **v3.0 - Funcionalidades Avanzadas**
- [ ] **Modelos Transformer**: Arquitecturas state-of-the-art
- [ ] **Multi-idioma**: Soporte completo para múltiples idiomas
- [ ] **Real-time**: Inferencia en tiempo real optimizada
- [ ] **Federado**: Entrenamiento federado para privacidad

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. **Fork** el repositorio
2. **Crea** una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** los cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crea** un Pull Request

### **Áreas de Contribución Prioritarias**
- 🧠 Nuevos modelos y arquitecturas
- 🎨 Mejoras de UI/UX
- 📊 Nuevas métricas y visualizaciones
- 🔧 Optimizaciones de performance
- 📚 Documentación y tutoriales

## 📄 Licencia

Este proyecto está licenciado bajo **MIT License** - ver `LICENSE` para detalles.

## 🙏 Reconocimientos

- **Hugging Face** por los modelos pre-entrenados
- **PyTorch Lightning** por el framework de entrenamiento
- **Streamlit** por la plataforma web
- **Google** por gTTS y servicios de síntesis

---

**Desarrollado con ❤️ para la comunidad de IA multimodal**