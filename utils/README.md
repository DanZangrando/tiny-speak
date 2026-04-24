# 🛠️ utils/
Caja de herramientas con funciones de apoyo, procesamiento de señales y gestión de hardware.

## Contenido
- **`audio.py`**: Carga, manipulación y síntesis de formas de onda.
- **`checkpoints.py`**: Gestión de persistencia de modelos y búsqueda en directorios.
- **`device.py`**: Detección inteligente de CPU/GPU (MPS para Mac, CUDA para Linux).
- **`graphemes.py`**: Lógica de idiomas, inventarios de fonemas y tokenización de texto.
- **`plotting.py`**: Funciones para generar gráficas de señales y matrices de confusión.
- **`serialization.py`**: Utilidades para manejo seguro de JSON y tipos de NumPy.

## Uso en el Experimento
Proporciona utilidades compartidas que desacoplan la lógica compleja del procesamiento de datos de las arquitecturas de red y de la UI, facilitando el mantenimiento y la extensibilidad.
