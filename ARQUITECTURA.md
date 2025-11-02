# 🧠 Arquitectura y Organización de TinySpeak

## 📋 **Resumen Ejecutivo**

TinySpeak es un sistema de IA multimodal que implementa tres arquitecturas principales para reconocimiento de patrones en audio y visión. La aplicación está organizada en páginas especializadas que permiten testing detallado de cada componente.

## 🏗️ **Arquitecturas de Modelos**

### 1. 🎵 **TinyListener (Audio → Palabra)**

```python
# Arquitectura completa
Audio (16kHz WAV) 
    ↓
Wav2Vec2-Base-ES (Facebook)
- Modelo: facebook/wav2vec2-base-es-voxpopuli-v2  
- Parámetros: ~95M (congelados)
- Output: 768D embeddings a ~49Hz
    ↓
Feature Processing
- Extracción capa 5 de Wav2Vec2
- Downsampling factor 7
- Padding de secuencias variables
    ↓  
LSTM Network
- Input: 768 dim
- Hidden: 64 dim
- Layers: 2  
- Batch-first: True
    ↓
Linear Classifier  
- Input: 64 dim (último estado LSTM)
- Output: 200 clases (palabras español)
```

**Parámetros entrenables:** ~592K
**Vocabulario:** 200 palabras en español
**Input:** Audio WAV a 16kHz
**Output:** Probabilidades por palabra

---

### 2. 🖼️ **TinyRecognizer (Imagen → Letra)**

```python
# Arquitectura CORnet-Z (inspirada en cortex visual)
Image (28×28×3 RGB)
    ↓
V1 Block (Visual Area 1)
- Conv2d(3→64, kernel=7, stride=2)
- ReLU + MaxPool2d(3×3, stride=2)  
    ↓
V2 Block (Visual Area 2)
- Conv2d(64→128, kernel=3)
- ReLU + MaxPool2d(3×3, stride=2)
    ↓  
V4 Block (Visual Area 4)
- Conv2d(128→256, kernel=3)
- ReLU + MaxPool2d(3×3, stride=2)
    ↓
IT Block (Inferotemporal Cortex) 
- Conv2d(256→512, kernel=3)
- ReLU + MaxPool2d(3×3, stride=2)
    ↓
Decoder
- AdaptiveAvgPool2d(1×1) 
- Flatten → Linear(512→1024) → ReLU
- Linear(1024→768) [Embedding space]
    ↓
Classifier
- Linear(768→26) [Letter classes a-z]
```

**Parámetros entrenables:** ~2.1M
**Input:** Imágenes RGB 28×28
**Output:** Probabilidades para 26 letras (a-z)
**Embedding:** Espacio de 768 dimensiones

---

### 3. 🔗 **TinySpeller (Multimodal: Visión + Audio)**

```python
# Arquitectura combinada
Secuencia Imágenes [L1, L2, ..., Ln]
    ↓
TinyRecognizer (para cada letra)
- Output: Embedding 768D por letra
    ↓  
Secuencia Embeddings [E1, E2, ..., En]
    ↓
TinySpeak (LSTM compartido)
- Input: Secuencia de embeddings 768D
- LSTM(768, 64, num_layers=2)  
- Output: Clasificador(64→200 palabras)
```

**Componentes:**
- TinyRecognizer (congelado): Extrae embeddings visuales
- TinySpeak (entrenado): Procesa secuencias para palabras

**Casos de uso:**
1. Secuencia letras manuscritas → palabra completa
2. Comparación con reconocimiento de audio directo

## 📁 **Organización de la Aplicación**

### **Página Principal** (`app.py`)
- 🏠 **Dashboard general** del sistema
- 📊 **Estado de componentes** (dispositivo, vocabulario, espeak)
- 🧪 **Test rápido** del sistema completo
- 🧭 **Navegación** a páginas especializadas

### **Páginas Especializadas** (`pages/`)

#### 1. 🎵 **TinyListener** (`01_🎵_TinyListener.py`)
**Funcionalidades:**
- 📁 **Carga de archivos** (WAV, MP3, FLAC, M4A)
- 🎤 **Grabación en tiempo real** 
- 🔊 **Síntesis + reconocimiento** (test loop cerrado)
- 📊 **Análisis interno** (arquitectura, vocabulario, embeddings)

**Testing incluido:**
- Upload y análisis de archivos de audio
- Grabación directa desde micrófono
- Síntesis controlada con parámetros (velocidad, tono, volumen)
- Análisis de palabras específicas del vocabulario
- Visualización de waveforms y distribuciones de logits

#### 2. 🖼️ **TinyRecognizer** (`02_🖼️_TinyRecognizer.py`)
**Funcionalidades:**
- 📁 **Carga de imágenes** de letras manuscritas
- ✏️ **Generación de letras** sintéticas (próximamente: canvas)
- 🔬 **Test sistemático** del alfabeto completo
- 📊 **Análisis interno** (activaciones por capa, embeddings)

**Testing incluido:**
- Reconocimiento de letras individuales
- Test sistemático A-Z con métricas de precisión
- Visualización de activaciones internas (V1, V2, V4, IT)
- Análisis de embeddings de 768 dimensiones
- Matriz de confusión para errores

#### 3. 🔗 **TinySpeller** (`03_🔗_TinySpeller.py`)
**Funcionalidades:**
- 🖼️➡️📝 **Secuencia letras → palabra** 
- 🎵➡️📝 **Audio directo → palabra**
- ⚖️ **Comparación multimodal** (visión vs audio)
- 🔬 **Análisis avanzado** (arquitectura, embeddings, benchmarks)

**Testing incluido:**
- Generación automática de secuencias de letras
- Análisis comparativo entre modalidades
- Benchmark de rendimiento (latencia por modalidad)
- Análisis de consenso/discrepancia entre modelos
- Exploración de espacios de embeddings

## 🛠️ **Archivos de Soporte**

### **Modelos** (`models.py`)
```python
# Clases principales
- TinySpeak: LSTM + Classifier base
- TinyListener: Wav2Vec2 + TinySpeak  
- TinyRecognizer: CORnet-Z + Classifier
- TinySpeller: TinyRecognizer + TinySpeak
- CORnet_Z: Arquitectura visual cortical
```

### **Utilidades** (`utils.py`)
```python
# Funciones clave
- encontrar_device(): Detección GPU/CPU/MPS
- load_wav2vec_model(): Carga Wav2Vec2 con config
- load_waveform(): Carga audio con fallback librosa
- synthesize_word(): Generación con espeak
- plot_waveform(): Visualización de audio
- plot_logits(): Gráficos de predicciones
- ensure_data_downloaded(): Descarga automática datasets
```

### **Testing** (`test_setup.py`)
```python
# Verificaciones del sistema
- test_imports(): PyTorch, Transformers, Streamlit
- test_device(): Detección y configuración dispositivo  
- test_espeak(): Síntesis de voz funcional
- test_models(): Carga e inicialización modelos
```

## 📊 **Datasets y Datos**

### **Datasets Descargados Automáticamente:**
1. **tiny-kalulu-200**: 200 palabras español (train/val)
2. **tiny-phones-200**: Fonemas concatenados 
3. **tiny-emnist-26**: Letras manuscritas A-Z

### **Estructura de Datos:**
```
data/
├── tiny-kalulu-200/
│   ├── train/[palabra]/[archivos.wav]
│   └── val/[palabra]/[archivos.wav]  
├── tiny-phones-200/
│   └── val/[fonema]/[archivos.wav]
└── tiny-emnist-26/
    ├── train/[letra]/[imágenes.JPEG] 
    └── val/[letra]/[imágenes.JPEG]
```

## 🎯 **Flujos de Testing Recomendados**

### **1. Test Individual de Modelos:**
```
🎵 TinyListener → Cargar audio → Verificar predicción
🖼️ TinyRecognizer → Cargar letra → Verificar reconocimiento  
🔗 TinySpeller → Secuencia letras → Verificar palabra
```

### **2. Test de Consistencia:**
```
Palabra "casa" → Síntesis → TinyListener → ¿Reconoce "casa"?
Letras C-A-S-A → TinySpeller → ¿Predice "casa"?
```

### **3. Test Comparativo:**
```
Misma palabra por múltiples modalidades → ¿Consenso?
Audio vs Visión → ¿Misma predicción?  
```

### **4. Test de Robustez:**
```
Parámetros síntesis variados → ¿Estabilidad?
Letras con diferentes estilos → ¿Generalización?
```

## 🚀 **Extensiones Futuras**

### **Mejoras de UI:**
- Canvas interactivo para dibujo de letras
- Grabación de audio con control de calidad
- Visualización 3D de embeddings
- Dashboard de métricas en tiempo real

### **Mejoras de Modelos:**
- Fine-tuning de Wav2Vec2 en dominio específico
- Aumento de datos para TinyRecognizer  
- Arquitectura attention para TinySpeller
- Modelos más grandes con mejor precisión

### **Nuevas Funcionalidades:**
- Reconocimiento de palabras fuera del vocabulario
- Detección de idioma automática
- Síntesis de voz con múltiples voces
- API REST para integración externa

## 📈 **Métricas de Rendimiento Actuales**

### **TinyListener:**
- Vocabulario: 200 palabras español
- Latencia: ~100-200ms por audio
- Precisión: Depende de calidad audio

### **TinyRecognizer:** 
- Clases: 26 letras (a-z)
- Latencia: ~10-20ms por imagen
- Precisión: Alta en letras claras

### **TinySpeller:**
- Modalidades: 2 (visión + audio)
- Comparación: Análisis de consenso
- Versatilidad: Palabras de longitud variable

---

*Esta arquitectura permite testing exhaustivo y comprensión profunda de cada componente del sistema TinySpeak, facilitando tanto la investigación como la demostración de capacidades multimodales.*