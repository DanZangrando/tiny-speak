# 🧩 Componentes Reutilizables

Este directorio contiene módulos y componentes de interfaz de usuario (UI) que se utilizan en múltiples páginas de la aplicación para mantener la consistencia y modularidad.

## 📦 Módulos

### 📊 `analytics.py`
Módulo centralizado para visualizaciones avanzadas y métricas de Machine Learning.
*   **Funciones clave**:
    *   `plot_learning_curves`: Gráficos interactivos de pérdida y precisión (Plotly).
    *   `plot_confusion_matrix`: Mapas de calor para matrices de confusión (Seaborn/Matplotlib).
    *   `plot_probability_matrix`: Visualización de probabilidades promedio por clase, ideal para muchas clases.
    *   `plot_latent_space_pca`: Visualización 3D interactiva del espacio latente (PCA + Plotly).
    *   `display_classification_report`: Formateo elegante de métricas de precisión, recall y F1.

### 🎨 `modern_sidebar.py`
Implementación de la barra lateral de navegación con estilo moderno.
*   **Características**:
    *   Diseño con Glassmorphism.
    *   Indicadores de estado del sistema (RAM, CPU, Disco).
    *   Navegación agrupada por funcionalidad.

### 📐 `diagrams.py`
Generación de diagramas de arquitectura de redes neuronales usando Graphviz.
*   **Modelos soportados**:
    *   TinyListener (Wav2Vec2 + LSTM).
    *   TinyRecognizer (CORnet-Z).
    *   TinyReader (Encoder-Decoder Generativo).

### 💻 `code_viewer.py`
Utilidad para mostrar fragmentos de código fuente dentro de la aplicación.
*   Permite inspeccionar la implementación real de los modelos y funciones directamente desde la UI.
