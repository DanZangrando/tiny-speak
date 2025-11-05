# 📄 Documentación de Páginas - TinySpeak

Esta documentación describe cada página de la aplicación TinySpeak y su funcionalidad específica.

## 🎵 01_TinyListener.py - Reconocimiento de Audio

### **Propósito**
Página principal para el reconocimiento de palabras a partir de audio usando el modelo TinyListener (Wav2Vec2 + LSTM).

### **Funcionalidades Principales**

#### **🔍 Inferencia de Audio**
- **Carga de Archivos**: Soporta WAV, MP3, FLAC, M4A
- **Grabación en Vivo**: Captura audio directamente desde el micrófono
- **Análisis Visual**: Waveform y espectrograma interactivos
- **Predicciones**: Top-5 palabras con scores de confianza
- **Visualización de Logits**: Gráficos de activaciones internas

#### **🏋️ Entrenamiento de TinyListener**
- **Configuración Completa**: Batch size, learning rate, épocas, device
- **Dataset Management**: Carga automática de splits train/val/test
- **Training con PyTorch Lightning**: Callbacks, early stopping, checkpoints
- **Métricas en Tiempo Real**: Loss, accuracy, learning rate scheduling
- **Visualización de Progreso**: Gráficos interactivos de entrenamiento

#### **🎤 Síntesis de Voz**
- **Text-to-Speech**: Generación de audio desde texto
- **Parámetros Configurables**: Velocidad, tono, volumen, idioma
- **Verificación Automática**: Análisis inmediato con TinyListener
- **Exportación**: Descarga de archivos WAV generados

### **Componentes UI**
- **Tab de Inferencia**: Upload, grabación, análisis y predicción
- **Tab de Entrenamiento**: Configuración completa de hyperparámetros
- **Tab de Síntesis**: Generación y validación de audio sintético
- **Sidebar**: Navegación y métricas del modelo

---

## 👁️ 02_TinyRecognizer.py - Reconocimiento Visual

### **Propósito**
Reconocimiento de letras manuscritas (a-z) usando arquitectura CORnet-Z con análisis de embeddings internos.

### **Funcionalidades Principales**

#### **🖼️ Inferencia Visual**
- **Carga de Imágenes**: Drag & drop, upload múltiple
- **Preprocesamiento Automático**: Redimensionado a 64x64, normalización
- **Predicción de Letras**: Clasificación a-z con confianza
- **Visualización de Embeddings**: Mapas de características internas
- **Análisis de Atención**: Regiones relevantes para la predicción

#### **🏋️ Entrenamiento de TinyRecognizer**
- **Dataset Visual**: Integración con Visual Dataset Manager
- **Configuración Avanzada**: Augmentations, regularización, optimizers
- **CORnet-Z Backbone**: Arquitectura inspirada en neurociencia
- **Métricas Detalladas**: Accuracy por clase, matriz de confusión
- **Checkpointing**: Guardado automático de mejores modelos

#### **📊 Análisis y Evaluación**
- **Embeddings Visualization**: t-SNE, PCA de representaciones
- **Confusion Matrix**: Análisis detallado de errores por clase
- **Learning Curves**: Evolución de métricas durante entrenamiento
- **Feature Maps**: Visualización de filtros convolucionales

### **Componentes UI**
- **Tab de Inferencia**: Análisis individual de imágenes
- **Tab de Entrenamiento**: Configuración completa de training
- **Tab de Evaluación**: Métricas y análisis post-entrenamiento
- **Galería Visual**: Ejemplos del dataset y predicciones

---

## 🔗 03_TinySpeller.py - Sistema Multimodal

### **Propósito**
Sistema híbrido que combina audio (TinyListener) y secuencias visuales (TinyRecognizer) para reconocimiento de palabras completas.

### **Funcionalidades Principales**

#### **🔄 Inferencia Multimodal**
- **Input Dual**: Audio de palabra + secuencia de imágenes de letras
- **Stack Multimodal**: Carga automática de TinyListener + TinyRecognizer + TinySpeller
- **Predicción Combinada**: Fusión de modalidades para decisión final
- **Comparación de Modalidades**: Audio vs Visual vs Multimodal
- **Análisis de Coherencia**: Validación entre predicciones de modalidades

#### **🏋️ Entrenamiento Multimodal**
- **Dataset Sincronizado**: Combinación automática de audio y visual datasets
- **Arquitectura Híbrida**: LSTM bidireccional + attention cross-modal
- **Collate Function Personalizada**: Padding inteligente para diferentes longitudes
- **PyTorch Lightning**: Entrenamiento distribuido y optimizado
- **Métricas Multimodales**: Word accuracy, top-k accuracy, ablation studies

#### **🔧 Configuración Avanzada**
- **Hyperparámetros**: Learning rates diferenciados por modalidad
- **Regularización**: Dropout, weight decay, label smoothing
- **Data Augmentation**: Augmentations específicas por modalidad
- **Early Stopping**: Basado en métricas multimodales combinadas

### **Componentes UI**
- **Tab de Inferencia**: Interface dual audio + visual
- **Tab de Entrenamiento**: Configuración completa multimodal
- **Tab de Comparación**: Análisis comparativo entre modalidades
- **Métricas Dashboard**: Visualización de performance combinada

---

## 🎤 04_AudioDataset.py - Generación de Dataset de Audio

### **Propósito**
Herramienta completa para generar, gestionar y validar datasets de audio usando Google Text-to-Speech (gTTS).

### **Funcionalidades Principales**

#### **🎯 Generación de Audio**
- **Text-to-Speech con gTTS**: Síntesis de alta calidad
- **Múltiples Variaciones**: Original, velocidad (0.8x-1.2x), tono, volumen
- **Idiomas Múltiples**: Español, inglés, francés, alemán, etc.
- **Conversión Automática**: MP3 → WAV para compatibilidad
- **Normalización**: Volumen y duración consistentes

#### **📚 Gestión de Vocabularios**
- **Diccionarios Predefinidos**: Kalulu (español), Phones (fonemas)
- **Vocabularios Temáticos**: Números, colores, animales, verbos
- **Vocabularios Personalizados**: Creación palabra por palabra
- **Sincronización**: Actualización automática de master_dataset_config.json

#### **✅ Validación y Control de Calidad**
- **Reproducción Automática**: Verify de cada audio generado
- **Análisis con TinyListener**: Validación de reconocimiento
- **Métricas de Calidad**: SNR, duración, consistencia
- **Reemplazo Selectivo**: Regeneración de audios problemáticos

#### **📊 Analytics y Métricas**
- **Dashboard en Tiempo Real**: Progreso de generación
- **Distribución de Datos**: Balance entre clases/palabras
- **Estadísticas de Calidad**: Tasas de éxito por configuración
- **Exportación de Reports**: Resúmenes detallados del dataset

### **Componentes UI**
- **Selector de Vocabulario**: Choose entre predefinidos o custom
- **Configuración de Síntesis**: Idioma, velocidad, variaciones
- **Progress Tracking**: Barras de progreso y status en tiempo real
- **Quality Control**: Reproducción y validación automática

---

## 🖼️ 05_VisualDataset.py - Generación de Dataset Visual

### **Propósito**
Sistema completo para generar datasets sintéticos de letras manuscritas con múltiples fuentes, estilos y augmentations.

### **Funcionalidades Principales**

#### **🎨 Generación de Letras**
- **Múltiples Fuentes**: 15+ tipografías (serif, sans-serif, script, display)
- **Tamaños Variables**: Optimización automática para 64x64px
- **Estilos Personalizables**: Bold, italic, outline, shadow
- **Colores Dinámicos**: Texto y fondo con contraste óptimo
- **Anti-aliasing**: Renderizado suave para mejor calidad

#### **🔄 Data Augmentation**
- **Transformaciones Geométricas**: Rotación (-15° a +15°), escalado
- **Efectos de Imagen**: Blur gaussiano, ruido, brillo, contraste
- **Distorsiones**: Shear, perspectiva, elastic transforms
- **Balanceo Automático**: Distribución equitativa de augmentations
- **Pipeline Configurable**: Probabilidades ajustables por transformación

#### **📁 Organización del Dataset**
- **Estructura Jerárquica**: Carpetas por letra (a-z)
- **Nomenclatura Sistemática**: Metadatos en nombres de archivo
- **Splits Automáticos**: Train (70%), validation (15%), test (15%)
- **Metadatos JSON**: Información completa de cada imagen
- **Índices de Búsqueda**: Acceso rápido por letra/estilo/fuente

#### **📊 Análisis y Visualización**
- **Galería Interactiva**: Preview de muestras generadas
- **Distribución de Clases**: Gráficos de balance del dataset
- **Quality Metrics**: Análisis de contraste, nitidez, variabilidad
- **Comparación de Estilos**: Side-by-side de diferentes fuentes

### **Componentes UI**
- **Configurador de Fuentes**: Selector múltiple con preview
- **Panel de Augmentations**: Sliders para probabilidades y intensidades
- **Generador Batch**: Configuración de cantidad por letra/estilo
- **Galería de Resultados**: Grid view con filtros y ordenamiento

---

## 📊 06_Dashboard.py - Analytics y Métricas

### **Propósito**
Dashboard centralizado para monitoreo, análisis y métricas del ecosistema completo de TinySpeak.

### **Funcionalidades Principales**

#### **📈 Métricas en Tiempo Real**
- **Estado de Datasets**: Conteos actualizados de audio y visual
- **Performance de Modelos**: Accuracy, loss, métricas por modalidad
- **Uso de Recursos**: CPU, GPU, memoria, storage
- **Health Checks**: Consistencia entre configuración y datasets reales

#### **🔍 Análisis Comparativo**
- **Benchmarking de Modelos**: TinyListener vs TinyRecognizer vs TinySpeller
- **A/B Testing**: Comparación entre versiones de modelos
- **Cross-Modal Analysis**: Correlaciones entre modalidades
- **Performance Trends**: Evolución temporal de métricas

#### **📊 Visualizaciones Interactivas**
- **Plotly Charts**: Gráficos responsive y zoom interactivo
- **Confusion Matrices**: Heatmaps detallados por modelo
- **Learning Curves**: Progreso de entrenamiento en tiempo real
- **Distribution Plots**: Análisis de balanceo de datasets

#### **⚙️ Configuración y Management**
- **Dataset Consistency**: Verificación automática de sincronización
- **Config Editor**: Interface para modificar master_dataset_config.json
- **Backup & Restore**: Snapshots de configuraciones y datasets
- **Performance Tuning**: Recomendaciones automáticas de optimización

#### **📋 Reporting y Export**
- **Summary Reports**: Informes ejecutivos de performance
- **Detailed Analytics**: Análisis técnicos profundos
- **Export Functionality**: CSV, JSON, PDF de métricas
- **Scheduling**: Reports automáticos periódicos

### **Componentes UI**
- **Main Dashboard**: Vista general con KPIs principales
- **Detailed Views**: Drill-down por modelo/dataset/métrica
- **Configuration Panel**: Editor de configuraciones
- **Export Center**: Generación y descarga de reports

---

## 🔧 Arquitectura Técnica Compartida

### **Componentes Comunes**

#### **🎨 Modern Sidebar (`components/modern_sidebar.py`)**
- **Glassmorphism Design**: Estética moderna con efectos de transparencia
- **Navegación Unificada**: Keys únicos para evitar conflictos
- **Responsive Layout**: Adaptación automática a diferentes pantallas
- **State Management**: Persistencia de navegación entre páginas

#### **🔗 Integración con Master Config**
- **Configuración Centralizada**: `master_dataset_config.json` como fuente única
- **Sincronización Automática**: Updates cross-página en tiempo real
- **Validation Layer**: Verificación de consistencia automática
- **Backup System**: Versioning de configuraciones

#### **⚡ Performance Optimizations**
- **Lazy Loading**: Carga diferida de modelos pesados
- **Caching Strategy**: `@st.cache_data` para operaciones costosas
- **Memory Management**: Liberación automática de recursos GPU
- **Batch Processing**: Operaciones vectorizadas cuando es posible

### **Patrones de Diseño**

#### **🏗️ Modular Architecture**
- **Separation of Concerns**: UI, logic, data claramente separados
- **Reusable Components**: Widgets compartidos entre páginas
- **Plugin System**: Extensibilidad para nuevos modelos/datasets
- **API Consistency**: Interfaces uniformes entre módulos

#### **📱 Responsive UI**
- **Column Layouts**: Adaptación automática a anchura de pantalla
- **Mobile-First**: Diseño optimizado para dispositivos móviles
- **Progressive Enhancement**: Funcionalidades adicionales en pantallas grandes
- **Accessibility**: Compatibilidad con screen readers y navegación por teclado

---

## 🚀 Próximos Desarrollos

### **Páginas Planificadas**
- **07_🔬_Experimentation.py**: A/B testing y experimentos controlados
- **08_🌐_API.py**: Interface REST para integración programática
- **09_📱_Mobile.py**: Versión optimizada para dispositivos móviles
- **10_🤖_AutoML.py**: Optimización automática de hyperparámetros

### **Mejoras en Páginas Existentes**
- **Real-time Training**: Streaming de métricas durante entrenamiento
- **Advanced Visualizations**: 3D plots, interactive embeddings
- **Collaborative Features**: Multi-user editing y sharing
- **Performance Profiling**: Análisis detallado de bottlenecks

---

**Documentación actualizada:** Noviembre 2025  
**Versión:** 2.0  
**Maintainer:** TinySpeak Development Team